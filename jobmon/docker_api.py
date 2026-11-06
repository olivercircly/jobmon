"""Thin wrapper around the docker CLI. Read-only: lists, stats, exec of read commands, logs."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass


class DockerError(RuntimeError):
    pass


@dataclass
class Container:
    name: str
    number: int
    state: str  # running, exited, restarting, ...
    status: str  # human readable, e.g. "Up 3 days"


@dataclass
class Stats:
    cpu: float  # percent, can exceed 100 on multi-core
    mem: str  # e.g. "1.08GiB / 1007GiB"


def _docker(args: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout, errors="replace",
            stdin=subprocess.DEVNULL,  # never inherit stdin: in MCP stdio mode it is the protocol channel
        )
    except FileNotFoundError as exc:
        raise DockerError("docker CLI not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerError(f"docker {' '.join(args[:2])} timed out after {timeout}s") from exc


def list_containers(prefix: str) -> list[Container]:
    """All containers (running or not) named <prefix>-<N>, sorted by N."""
    proc = _docker(["ps", "-a", "--filter", f"name=^{prefix}-", "--format", "{{json .}}"])
    if proc.returncode != 0:
        raise DockerError(proc.stderr.strip())
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    containers = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        match = pattern.match(row["Names"])
        if match:
            containers.append(
                Container(row["Names"], int(match.group(1)), row["State"], row["Status"])
            )
    return sorted(containers, key=lambda c: c.number)


def _parse_percent(value: str) -> float:
    try:
        return float(value.strip().rstrip("%"))
    except ValueError:
        return 0.0


def stats(names: list[str]) -> dict[str, Stats]:
    """One `docker stats --no-stream` call for all given (running) containers."""
    if not names:
        return {}
    proc = _docker(["stats", "--no-stream", "--format", "{{json .}}", *names], timeout=60)
    result = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[row["Name"]] = Stats(_parse_percent(row["CPUPerc"]), row["MemUsage"])
    return result


def exec_sh(container: str, script: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return _docker(["exec", container, "sh", "-c", script], timeout=timeout)


def logs(container: str, tail: int) -> str:
    try:
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise DockerError("docker CLI not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerError("docker logs timed out") from exc
    return proc.stdout
