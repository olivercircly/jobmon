"""jobmon operations returning plain data - shared by the CLI and the MCP server.

Nothing in here prints: the MCP server talks over stdout, so all output is the caller's job.
"""

from __future__ import annotations

import re
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import collector, docker_api
from .collector import RUNNING, STALE
from .parse import ERROR_PATTERN, LOG_DIR, STDOUT_ERROR_RE


class JobmonError(RuntimeError):
    """An anticipated failure with a message meant for the user (no such job, no logs, bad input)."""


_JOB_RE = re.compile(r"^\d+$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")  # docker container name / prefix


# --- input handling ------------------------------------------------------------------------


def container_prefix(env: str, prefix: str | None = None) -> str:
    result = prefix or f"{env}-ai-pipeline-gpu"
    if not _NAME_RE.match(result):
        raise JobmonError(f"invalid container prefix: {result!r}")
    return result


def resolve_container(prefix: str, value: str) -> str:
    """Container number (e.g. "30") -> "<prefix>-30"; full names are passed through."""
    value = value.strip()
    name = f"{prefix}-{value}" if value.isdigit() else value
    if not _NAME_RE.match(name):
        raise JobmonError(f"invalid container name: {value!r}")
    return name


def check_job(job: str | None) -> str | None:
    if job is None:
        return None
    job = job.strip()
    if not _JOB_RE.match(job):
        raise JobmonError(f"invalid job ID {job!r}: expected digits")
    return job


def fmt_duration(seconds: int | None) -> str:
    if seconds is None:
        return ""
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes:02d}m"
    return f"{hours // 24}d{hours % 24:02d}h"


def open_log_cmd(container: str, file: str) -> str:
    """Shell command a human can run to read a log file."""
    return f"docker exec {container} cat {shlex.quote(LOG_DIR + '/' + file)} | less +G"


# --- find ----------------------------------------------------------------------------------


@dataclass
class JobHit:
    search: str  # the token that matched (job ID or CID)
    container: str
    result: str  # "running" / "stale" (job is running now), "ok" / "error" (finished)
    job: str
    subjob: str | None
    cid: str | None
    log_file: str
    trigger: str = ""
    runtime_s: int | None = None  # running jobs
    finished_epoch: int | None = None  # finished jobs, UTC


@dataclass
class FindResult:
    hits: list[JobHit]
    missing: list[str]


def find_jobs(prefix: str, ids: list[str]) -> FindResult:
    """Where do/did job IDs (or CIDs) run, and did they finish OK or with ERROR?"""
    tokens = {t.strip() for t in ids if t.strip()}
    states = collector.collect(prefix, with_stats=False)
    running = [s for s in states if s.container.state == "running"]
    hits: list[JobHit] = []

    for s in states:
        if s.job and (s.job in tokens or s.cid in tokens) and s.state in (RUNNING, STALE):
            hits.append(JobHit(
                search=s.job if s.job in tokens else s.cid, container=s.name, result=s.state,
                job=s.job, subjob=s.subjob, cid=s.cid, log_file="logger.log", runtime_s=s.runtime_s,
            ))

    with ThreadPoolExecutor(max_workers=16) as pool:
        archives = list(pool.map(lambda s: collector.list_archives(s.name), running))
    for s, files in zip(running, archives):
        for a in files:
            tok = a.job if a.job in tokens else a.cid if a.cid in tokens else None
            if tok is not None:
                hits.append(JobHit(
                    search=tok, container=s.name, result="error" if a.error else "ok",
                    job=a.job, subjob=a.subjob, cid=a.cid, log_file=a.name, trigger=a.trigger,
                    finished_epoch=a.finished_epoch,
                ))

    found = {h.search for h in hits}
    return FindResult(hits, sorted(tokens - found))


# --- errors / logs -------------------------------------------------------------------------


@dataclass
class LogFile:
    container: str
    name: str  # file name in LOG_DIR
    description: str

    @property
    def path(self) -> str:
        return f"{LOG_DIR}/{self.name}"


def pick_log_file(container: str, job: str | None) -> LogFile:
    """The log of `job` in `container`, or the current job's log, else the last finished one."""
    job = check_job(job)
    if job:
        live = docker_api.exec_sh(container, f"cd {LOG_DIR} && grep -m1 -F 'JobID {job};' logger.log")
        if live.returncode == 0 and live.stdout.strip():
            return LogFile(container, "logger.log", f"job {job} (running now)")
        matches = [a for a in collector.list_archives(container) if a.job == job]
        if not matches:
            raise JobmonError(f"No log for job {job} in {container}. Search all containers with find.")
        a = min(matches, key=lambda a: (not a.error, -a.finished_epoch))  # prefer ERROR, newest
        return LogFile(container, a.name, f"job {a.job} subjob {a.subjob} ({'ERROR' if a.error else 'OK'})")
    if docker_api.exec_sh(container, f"cd {LOG_DIR} && test -s logger.log").returncode == 0:
        return LogFile(container, "logger.log", "current job")
    archives = collector.list_archives(container)
    if not archives:
        raise JobmonError(f"No logs found in {container}")
    a = archives[0]
    return LogFile(container, a.name, f"last finished job {a.job} ({'ERROR' if a.error else 'OK'})")


@dataclass
class StdoutLine:
    text: str
    hit: bool  # the error line itself (False = context line)


@dataclass
class ErrorReport:
    log: LogFile
    lines: list[str]  # grep -n -C output of the job log; "--" separates groups
    stdout: list[StdoutLine] | None = None  # only when docker logs were scanned
    stdout_tail: int = 0


def job_errors(
    container: str,
    job: str | None = None,
    context: int = 2,
    limit: int = 200,
    scan_stdout: bool = False,
    stdout_tail: int = 5000,
) -> ErrorReport:
    context, limit, stdout_tail = max(0, int(context)), max(1, int(limit)), max(1, int(stdout_tail))
    log = pick_log_file(container, job)
    script = (
        f"cd {LOG_DIR} && grep -n -i -E -C {context} {shlex.quote(ERROR_PATTERN)} {shlex.quote(log.name)}"
        f" | tail -n {limit}"
    )
    out = docker_api.exec_sh(container, script, timeout=60).stdout
    report = ErrorReport(log, out.splitlines())

    if scan_stdout:
        lines = docker_api.logs(container, stdout_tail).splitlines()
        hits = [i for i, line in enumerate(lines) if STDOUT_ERROR_RE.search(line)]
        shown: set[int] = set()
        report.stdout, report.stdout_tail = [], stdout_tail
        for i in hits[-limit:]:
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                if j not in shown:
                    shown.add(j)
                    report.stdout.append(StdoutLine(lines[j], j == i))
    return report


def tail_log(container: str, job: str | None = None, lines: int = 50) -> tuple[LogFile, str]:
    log = pick_log_file(container, job)
    out = docker_api.exec_sh(container, f"tail -n {max(1, int(lines))} {shlex.quote(log.path)}").stdout
    return log, out


def utc(epoch: int | None) -> str | None:
    return None if epoch is None else time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(epoch))
