"""jobmon as an MCP server (read-only tools for AI-Pipeline job monitoring).

Default transport is stdio: the MCP client starts `jobmon-mcp` itself (locally or via
`ssh <host> jobmon-mcp`). `--http` serves streamable HTTP instead, for a long-running service.

stdout carries the MCP protocol in stdio mode - nothing in here may print to it.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from . import collector, docker_api, service
from .collector import IDLE, STATE_ORDER, ContainerState
from .service import JobmonError

MAX_LINE_CHARS = 1000  # very long log lines (dumped dicts, progress bars) are clipped
MAX_ERROR_LINES = 500
MAX_TAIL_LINES = 1000

# Server-wide defaults, set from the command line in main()
CONFIG = {"env": os.environ.get("JOBMON_ENV", "prod"), "prefix": os.environ.get("JOBMON_PREFIX") or None}

INSTRUCTIONS = """\
Read-only monitoring of AI-Pipeline jobs running in Docker containers named <env>-ai-pipeline-gpu-<N>.
Containers are addressed by their number N (e.g. "30"). Each container runs at most one job at a time;
the current job logs to logger.log, finished jobs are archived as logger.log_[ERROR_]<cid>_<trigger>_<job>_<subjob>_<ts>.

Container states: running (job active), stale (job active but no log output for stale_minutes - possibly hung),
busy? (no job logged but high CPU - starting up, or logging elsewhere), idle, down (container not running),
error (container could not be inspected).

Typical flow: status for the overview -> find_jobs to locate a job ID -> job_errors / tail_log on that container.
"""

mcp = MCPServer("jobmon", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


# --- helpers -------------------------------------------------------------------------------


def _prefix(env: str | None) -> str:
    return service.container_prefix(env or CONFIG["env"], None if env else CONFIG["prefix"])


def _container(env: str | None, container: str) -> str:
    """Resolve a container number/name, restricted to the pipeline containers of the prefix."""
    prefix = _prefix(env)
    name = service.resolve_container(prefix, container)
    if not re.fullmatch(rf"{re.escape(prefix)}-\d+", name):
        raise JobmonError(f"{name!r} is not a pipeline container ({prefix}-<N>); pass the container number")
    return name


def _clip(line: str) -> str:
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + f" …[+{len(line) - MAX_LINE_CHARS} chars]"


def _run(fn, *args, **kwargs):
    """Turn anticipated failures into tool errors the model can read."""
    try:
        return fn(*args, **kwargs)
    except JobmonError as exc:
        raise ToolError(str(exc)) from exc
    except docker_api.DockerError as exc:
        raise ToolError(f"docker error: {exc}") from exc


def _duration(seconds: int | None) -> str | None:
    return service.fmt_duration(seconds) if seconds is not None else None


def _state_summary(s: ContainerState) -> dict[str, Any]:
    step = None
    if s.last_line:
        step = f"{s.last_line.category or ''}: {s.last_line.message}".strip(": ")
    out: dict[str, Any] = {
        "number": s.container.number,
        "name": s.name,
        "state": s.state,
        "detail": s.detail or None,
        "job": s.job,
        "subjob": s.subjob,
        "cid": s.cid,
        "model": s.model,
        "runtime": _duration(s.runtime_s),
        "since_last_log": _duration(s.since_last_log_s),
        "cpu_percent": None if s.cpu is None else round(s.cpu, 1),
        "mem": s.mem,
        "current_step": _clip(step) if step else None,
        "last_finished": None,
    }
    if s.last_finished:
        out["last_finished"] = {
            "job": s.last_finished.job,
            "subjob": s.last_finished.subjob,
            "result": "ok" if s.last_finished.ok else "error",
            "trigger": s.last_finished.trigger,
            "finished": service.utc(s.last_finished.finished_epoch),
            "ago": _duration(s.finished_ago_s),
        }
    return {k: v for k, v in out.items() if v is not None}


# --- tools ---------------------------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def status(
    env: str | None = None,
    include_idle: bool = False,
    stale_minutes: int = 30,
    with_cpu: bool = True,
) -> dict[str, Any]:
    """Overview of all pipeline containers: state, running job (job/subjob/CID/model), runtime,
    time since the last log line, CPU, current step and the last finished job (ok/error).

    Containers are sorted problems first (down, error, stale, running, busy?, idle).
    Idle containers are omitted unless include_idle is true (they are still counted).
    env: deployment environment, e.g. "prod" or "staging" (default: server setting).
    with_cpu=false skips `docker stats` (faster, but busy? cannot be detected).
    """
    prefix = _run(_prefix, env)
    states = _run(collector.collect, prefix, with_stats=with_cpu, stale_after_s=max(1, stale_minutes) * 60)
    if not states:
        raise ToolError(f"No containers named {prefix}-<N> found")
    states.sort(key=lambda s: (STATE_ORDER.index(s.state), s.container.number))
    counts = Counter(s.state for s in states)
    shown = [s for s in states if include_idle or s.state != IDLE]
    return {
        "prefix": prefix,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "counts": {state: counts[state] for state in STATE_ORDER if counts[state]},
        "containers": [_state_summary(s) for s in shown],
        "idle_hidden": len(states) - len(shown),
    }


@mcp.tool(annotations=READ_ONLY)
def find_jobs(ids: list[str], env: str | None = None) -> dict[str, Any]:
    """Find where job IDs (or CIDs) are running now or ran before, and whether they finished OK or
    with ERROR. Searches the current job of every container and all archived job logs.

    Returns one hit per (container, log file); a job can have several hits (retries, subjobs).
    log_file can be passed on to job_errors / tail_log via the job ID and container number.
    """
    if not ids:
        raise ToolError("pass at least one job ID or CID")
    result = _run(service.find_jobs, _run(_prefix, env), ids)
    now = time.time()
    hits = []
    for h in result.hits:
        hit: dict[str, Any] = {
            "search": h.search,
            "container": h.container,
            "result": h.result,
            "job": h.job,
            "subjob": h.subjob,
            "cid": h.cid,
            "log_file": h.log_file,
        }
        if h.trigger:
            hit["trigger"] = h.trigger
        if h.runtime_s is not None:
            hit["runtime"] = _duration(h.runtime_s)
        if h.finished_epoch is not None:
            hit["finished"] = service.utc(h.finished_epoch)
            hit["ago"] = _duration(int(now - h.finished_epoch))
        hits.append({k: v for k, v in hit.items() if v is not None})
    return {"hits": hits, "not_found": result.missing}


@mcp.tool(annotations=READ_ONLY)
def job_errors(
    container: str,
    job: str | None = None,
    env: str | None = None,
    context: int = 2,
    max_lines: int = 200,
    include_docker_logs: bool = False,
) -> dict[str, Any]:
    """Error lines (error/exception/traceback/failed/killed/out of memory) with context from a job log.

    container: container number, e.g. "30".
    job: job ID; default is the container's current job, else its last finished job.
         For a finished job with several archived logs, the newest ERROR log is used.
    max_lines: keep only the last max_lines output lines (max 500).
    include_docker_logs: also scan the last 5000 lines of `docker logs` (stdout/stderr of the
         container process) - useful when the job crashed before or outside the job logger.
    Log lines are "<line no>:<text>" for matches and "<line no>-<text>" for context; "--" separates groups.
    """
    name = _run(_container, env, container)
    report = _run(
        service.job_errors, name, job,
        context=min(max(context, 0), 20), limit=min(max(max_lines, 1), MAX_ERROR_LINES),
        scan_stdout=include_docker_logs,
    )
    out: dict[str, Any] = {
        "container": name,
        "log": report.log.description,
        "log_file": report.log.path,
        "error_lines": [_clip(line) for line in report.lines],
    }
    if not report.lines:
        out["note"] = "no error lines in this log file"
    if report.stdout is not None:
        out["docker_logs"] = [("!! " if line.hit else "   ") + _clip(line.text) for line in report.stdout]
        out["docker_logs_note"] = (
            f"scanned last {report.stdout_tail} lines; '!!' marks error lines"
            if report.stdout else f"no errors in the last {report.stdout_tail} lines of docker logs"
        )
    return out


@mcp.tool(annotations=READ_ONLY)
def tail_log(container: str, job: str | None = None, lines: int = 100, env: str | None = None) -> dict[str, Any]:
    """Last lines of a container's job log (current job by default, or a specific job ID).

    container: container number, e.g. "30". lines: max 1000.
    Line format: <time>;JobID <job>;JobID <subjob>;CID <cid>;ModelID <model>;<module>;<message>
    """
    name = _run(_container, env, container)
    log, text = _run(service.tail_log, name, job, min(max(lines, 1), MAX_TAIL_LINES))
    return {
        "container": name,
        "log": log.description,
        "log_file": log.path,
        "lines": [_clip(line) for line in text.splitlines()],
    }


# --- entry point ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jobmon-mcp", description="jobmon MCP server (stdio by default).")
    p.add_argument("--env", default=CONFIG["env"], help="default environment (default: prod, or $JOBMON_ENV)")
    p.add_argument("--prefix", default=CONFIG["prefix"], help="full container prefix, overrides --env")
    p.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1", help="HTTP bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="HTTP port (default 8000)")
    args = p.parse_args(argv)
    CONFIG.update(env=args.env, prefix=args.prefix)
    service.container_prefix(args.env, args.prefix)  # fail fast on a bad prefix

    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
