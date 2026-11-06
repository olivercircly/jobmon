"""jobmon command line interface."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from . import collector, docker_api, service
from .collector import BUSY, DOWN, ERROR, IDLE, RUNNING, STALE, STATE_ORDER, ContainerState
from .parse import ERROR_PATTERN
from .service import JobmonError, fmt_duration, open_log_cmd

console = Console()

STATE_STYLE = {
    DOWN: "bold red",
    ERROR: "red",
    STALE: "bold yellow",
    RUNNING: "green",
    BUSY: "cyan",
    IDLE: "dim",
}


# --- helpers -------------------------------------------------------------------------------


def prefix_for(args) -> str:
    return service.container_prefix(args.env, args.prefix)


def resolve_container(args, value: str) -> str:
    return service.resolve_container(prefix_for(args), value)


# --- status --------------------------------------------------------------------------------


def status_table(states: list[ContainerState], prefix: str, show_idle: bool) -> Group:
    counts = Counter(s.state for s in states)
    summary = Text(f"{prefix}  ·  {len(states)} containers  ·  ")
    for i, state in enumerate(s for s in STATE_ORDER if counts[s]):
        if i:
            summary.append("  ")
        summary.append(f"{counts[state]} {state}", style=STATE_STYLE[state])
    summary.append(f"  ·  {time.strftime('%H:%M:%S')}", style="dim")

    table = Table(show_edge=False, pad_edge=False, expand=True, header_style="bold")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Job", no_wrap=True)
    table.add_column("Sub", no_wrap=True)
    table.add_column("CID", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Runtime", justify="right", no_wrap=True)
    table.add_column("Last log", justify="right", no_wrap=True)
    table.add_column("CPU", justify="right", no_wrap=True)
    table.add_column("Current step / detail", overflow="ellipsis", no_wrap=True, ratio=1)
    table.add_column("Last finished", no_wrap=True)

    hidden = 0
    for s in states:
        if not show_idle and s.state == IDLE:
            hidden += 1
            continue
        style = STATE_STYLE[s.state]
        if s.last_line:
            step = Text(f"{s.last_line.category or ''}: {s.last_line.message}".strip(": "))
        else:
            step = Text(s.detail, style="dim" if s.state == IDLE else style)
        if s.state == STALE and s.detail:
            step = Text(f"[{s.detail}] ", style=style) + step
        finished = Text()
        if s.last_finished:
            ok = s.last_finished.ok
            finished.append("✓ " if ok else "✗ ", style="green" if ok else "bold red")
            finished.append(f"{s.last_finished.job} ")
            finished.append(fmt_duration(s.finished_ago_s) + " ago", style="dim")
        table.add_row(
            str(s.container.number),
            Text(s.state, style=style),
            s.job or "",
            s.subjob or "",
            (s.cid or "")[:8],
            s.model or "",
            fmt_duration(s.runtime_s),
            Text(fmt_duration(s.since_last_log_s), style="yellow" if s.state == STALE else ""),
            "" if s.cpu is None else f"{s.cpu:.0f}%",
            step,
            finished,
        )
    parts = [summary, table]
    if hidden:
        parts.append(Text(f"{hidden} idle containers hidden (use --all)", style="dim"))
    return Group(*parts)


def cmd_status(args) -> int:
    prefix = prefix_for(args)

    def gather():
        states = collector.collect(prefix, with_stats=not args.no_stats, stale_after_s=args.stale_minutes * 60)
        if args.sort:
            states.sort(key=lambda s: (STATE_ORDER.index(s.state), s.container.number))
        return states

    if args.json:
        print(json.dumps([asdict(s) for s in gather()], default=str, indent=2))
        return 0
    if not args.watch:
        states = gather()
        if not states:
            console.print(f"[red]No containers named {prefix}-<N> found.[/]")
            return 1
        console.print(status_table(states, prefix, args.all))
        return 0
    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                live.update(status_table(gather(), prefix, args.all), refresh=True)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    return 0


# --- find ----------------------------------------------------------------------------------


def cmd_find(args) -> int:
    with console.status("Searching running jobs and archived logs ..."):
        result = service.find_jobs(prefix_for(args), args.ids)

    table = Table(header_style="bold", show_edge=False)
    for col in ("Search", "Container", "Result", "Job", "Sub", "Trigger", "When", "CID"):
        table.add_column(col, no_wrap=True)
    commands: list[str] = []
    now = time.time()
    for h in result.hits:
        if h.result in (RUNNING, STALE):
            label = Text(f"{h.result} now", style=STATE_STYLE[h.result])
            when = f"running {fmt_duration(h.runtime_s)}"
            commands.append(open_log_cmd(h.container, h.log_file))
        else:
            label = Text("finished OK", style="green") if h.result == "ok" else Text("ERROR", style="bold red")
            when = f"{service.utc(h.finished_epoch)} ({fmt_duration(int(now - h.finished_epoch))} ago)"
            if h.result == "error":
                commands.append(open_log_cmd(h.container, h.log_file))
        table.add_row(h.search, h.container, label, h.job, h.subjob or "", h.trigger, when, h.cid or "")

    if table.row_count:
        console.print(table)
    if result.missing:
        console.print(f"[yellow]Not found:[/] {' '.join(result.missing)}")
    if commands:
        console.print("\n[bold]Open logs:[/]")
        for c in dict.fromkeys(commands):
            console.print(f"  {c}", soft_wrap=True, highlight=False)
    return 0 if result.hits else 1


# --- errors / logs -------------------------------------------------------------------------


def cmd_errors(args) -> int:
    container = resolve_container(args, args.container)
    report = service.job_errors(
        container, args.job, context=args.context, limit=args.limit,
        scan_stdout=args.stdout, stdout_tail=args.stdout_tail,
    )
    console.rule(f"[bold]{container}[/] · {report.log.description}")
    console.print(f"[dim]{report.log.path}[/]", highlight=False)
    if report.lines:
        rx = re.compile(ERROR_PATTERN, re.IGNORECASE)
        for line in report.lines:
            text = Text(line)
            if line == "--":
                text.stylize("dim")
            else:
                text.highlight_regex(rx, "bold red")
            console.print(text, soft_wrap=True, highlight=False)
    else:
        console.print("[green]No error lines in this log file.[/]")

    if report.stdout is not None:
        console.rule("docker logs (stdout/stderr)")
        for line in report.stdout:
            console.print(Text(line.text, style="bold red" if line.hit else ""), soft_wrap=True, highlight=False)
        if not report.stdout:
            console.print(f"[green]No errors in the last {report.stdout_tail} lines of docker logs.[/]")

    console.print(f"\n[bold]Full log:[/] {open_log_cmd(container, report.log.name)}", highlight=False, soft_wrap=True)
    return 0


def cmd_logs(args) -> int:
    container = resolve_container(args, args.container)
    if args.follow:
        log = service.pick_log_file(container, args.job)
        os.execvp("docker", ["docker", "exec", "-it", container, "tail", "-n", str(args.lines), "-f", log.path])
    log, out = service.tail_log(container, args.job, args.lines)
    console.rule(f"[bold]{container}[/] · {log.description}")
    sys.stdout.write(out)
    return 0


# --- main ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jobmon", description="Monitor and debug AI-Pipeline jobs in Docker.")
    p.add_argument("--env", default=os.environ.get("JOBMON_ENV", "prod"),
                   help="container prefix <env>-ai-pipeline-gpu (default: prod, or $JOBMON_ENV)")
    p.add_argument("--prefix", help="full container prefix, overrides --env")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("status", help="what is running where (default command)")
    s.add_argument("-w", "--watch", type=int, nargs="?", const=10, metavar="SEC", help="refresh every SEC seconds (default 10)")
    s.add_argument("-a", "--all", action="store_true", help="also show idle containers")
    s.add_argument("-s", "--sort", action="store_true", help="sort by state (problems first)")
    s.add_argument("--stale-minutes", type=int, default=30, help="flag jobs without log output for this long (default 30)")
    s.add_argument("--no-stats", action="store_true", help="skip docker stats (faster)")
    s.add_argument("--json", action="store_true", help="machine readable output")
    s.set_defaults(func=cmd_status)

    f = sub.add_parser("find", help="find where job IDs (or CIDs) run / ran and how they ended")
    f.add_argument("ids", nargs="+")
    f.set_defaults(func=cmd_find)

    e = sub.add_parser("errors", help="error lines of a container's current or a specific job log")
    e.add_argument("container", help="container number (e.g. 30) or full name")
    e.add_argument("-j", "--job", help="job ID (default: current job, else last finished)")
    e.add_argument("-C", "--context", type=int, default=2, help="context lines (default 2)")
    e.add_argument("-n", "--limit", type=int, default=200, help="max output lines (default 200)")
    e.add_argument("--stdout", action="store_true", help="also scan docker logs (stdout/stderr)")
    e.add_argument("--stdout-tail", type=int, default=5000, help="docker logs lines to scan (default 5000)")
    e.set_defaults(func=cmd_errors)

    l = sub.add_parser("logs", help="tail a container's current or a specific job log")
    l.add_argument("container")
    l.add_argument("-j", "--job")
    l.add_argument("-n", "--lines", type=int, default=50)
    l.add_argument("-f", "--follow", action="store_true")
    l.set_defaults(func=cmd_logs)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args([*(argv if argv is not None else sys.argv[1:]), "status"])
    try:
        return args.func(args)
    except docker_api.DockerError as exc:
        console.print(f"[red]docker error:[/] {exc}")
        return 2
    except JobmonError as exc:
        console.print(f"[red]{exc}[/]", highlight=False)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
