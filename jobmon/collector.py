"""Collect the state of every pipeline container in parallel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import docker_api
from .parse import LOG_DIR, SNAPSHOT_SCRIPT, ArchiveFile, LogLine, Snapshot, parse_archive_name, parse_snapshot

# States, ordered by how much attention they need
DOWN = "down"  # container not running
ERROR = "error"  # could not inspect the container
STALE = "stale"  # job running but logger.log not written for a long time
RUNNING = "running"
BUSY = "busy?"  # no job in logger.log but high CPU (job starting / logging elsewhere)
IDLE = "idle"

STATE_ORDER = [DOWN, ERROR, STALE, RUNNING, BUSY, IDLE]


@dataclass
class ContainerState:
    container: docker_api.Container
    state: str
    detail: str = ""
    cpu: float | None = None
    mem: str | None = None
    job: str | None = None
    subjob: str | None = None
    cid: str | None = None
    model: str | None = None
    runtime_s: int | None = None
    since_last_log_s: int | None = None
    last_line: LogLine | None = None
    last_finished: ArchiveFile | None = None
    finished_ago_s: int | None = None
    recent_archives: list[ArchiveFile] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.container.name


def _snapshot(container: docker_api.Container) -> tuple[Snapshot | None, str]:
    try:
        proc = docker_api.exec_sh(container.name, SNAPSHOT_SCRIPT, timeout=30)
    except docker_api.DockerError as exc:
        return None, str(exc)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        return None, msg.splitlines()[-1] if msg else f"exit {proc.returncode}"
    return parse_snapshot(proc.stdout), ""


def build_state(
    container: docker_api.Container,
    snap: Snapshot | None,
    err: str,
    stats: docker_api.Stats | None,
    stale_after_s: int,
    busy_cpu: float = 50.0,
) -> ContainerState:
    cs = ContainerState(container, IDLE)
    if stats:
        cs.cpu, cs.mem = stats.cpu, stats.mem
    if container.state != "running":
        cs.state, cs.detail = DOWN, container.status
        return cs
    if snap is None:
        cs.state, cs.detail = ERROR, err
        return cs
    if not snap.has_logdir:
        cs.state, cs.detail = ERROR, "log dir not found"
        return cs

    cs.recent_archives = snap.recent_archives or []
    if cs.recent_archives:
        cs.last_finished = cs.recent_archives[0]
        if snap.now_epoch is not None:
            cs.finished_ago_s = snap.now_epoch - cs.last_finished.finished_epoch

    lines = snap.last_lines or []
    if not lines:
        cs.state = BUSY if (cs.cpu or 0) >= busy_cpu else IDLE
        return cs

    cs.state = RUNNING
    cs.last_line = lines[-1]
    # Job tags: newest line that has them (some lines may lack the extra)
    for line in [*reversed(lines), snap.first_line]:
        if line and line.job:
            cs.job, cs.subjob, cs.cid, cs.model = line.job, line.subjob, line.cid, line.model
            break
    if snap.first_line and snap.first_line.time and snap.now_local:
        cs.runtime_s = int((snap.now_local - snap.first_line.time).total_seconds())
    if snap.log_mtime is not None and snap.now_epoch is not None:
        cs.since_last_log_s = snap.now_epoch - snap.log_mtime
        if cs.since_last_log_s > stale_after_s:
            cs.state = STALE
            cs.detail = "no log output for a while"
            if cs.cpu is not None and cs.cpu < 1:
                cs.detail += ", CPU idle"
    return cs


def collect(prefix: str, with_stats: bool = True, stale_after_s: int = 1800, workers: int = 16) -> list[ContainerState]:
    containers = docker_api.list_containers(prefix)
    running = [c for c in containers if c.state == "running"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        stats_future = pool.submit(docker_api.stats, [c.name for c in running]) if with_stats and running else None
        snaps = dict(zip([c.name for c in running], pool.map(_snapshot, running)))
        try:
            stats = stats_future.result() if stats_future else {}
        except docker_api.DockerError:
            stats = {}

    return [
        build_state(c, *snaps.get(c.name, (None, "")), stats.get(c.name), stale_after_s)
        for c in containers
    ]


def list_archives(container: str) -> list[ArchiveFile]:
    """All archived logger.log files of a container, newest first."""
    proc = docker_api.exec_sh(container, f"cd {LOG_DIR} && ls -1 | grep '^logger\\.log_'", timeout=60)
    files = [a for line in proc.stdout.splitlines() if (a := parse_archive_name(line))]
    return sorted(files, key=lambda a: a.finished_epoch, reverse=True)
