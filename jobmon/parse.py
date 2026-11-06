"""Parsing of the AI-Pipeline log format (DevAI_utils/notifications/Logging).

logger.log lines (log_config.yml, formatter "simple", extra set in AI-Pipeline/entrypoint.py):
    2026-11-05 10:12:01,123;JobID 69145;SubjobID 812;CID <uuid>;ModelID 7;<module>;<message>

Archived logs (create_historic_logs in DevAI_utils/notifications/Logging/logging.py):
    logger.log_<cid>_<trigger>_<job>_<subjob>_<YYYY_MM_DD-HH-MM-SS UTC>
    logger.log_ERROR_<cid>_<trigger>_<job>_<subjob>_<YYYY_MM_DD-HH-MM-SS UTC>
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime

LOG_DIR = "DevAI_utils/notifications/Logging"

# Job and subjob tags: "JobID 1;SubjobID 2" in the AI-Pipeline source, but deployed images
# write "JobID 1;JobID 2" - accept both, first is the job, second the subjob.
_JOB_TAGS_RE = re.compile(r"(?:^|;)(?:Subjob|Job)ID (\S+?)(?=;|$)")
_TAG_RE = {
    "cid": re.compile(r"CID (\S+?)(?:;|$)"),
    "model": re.compile(r"ModelID (\S+?)(?:;|$)"),
}
_ASCTIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

ARCHIVE_RE = re.compile(
    r"^logger\.log_(?P<error>ERROR_)?(?P<cid>[^_]+)_(?P<trigger>.+)_(?P<job>\d+)_(?P<subjob>\d+|None)"
    r"_(?P<ts>\d{4}_\d{2}_\d{2}-\d{2}-\d{2}-\d{2})$"
)


def _none(value: str | None) -> str | None:
    return None if value in (None, "", "None") else value


@dataclass
class LogLine:
    raw: str
    time: datetime | None
    job: str | None
    subjob: str | None
    cid: str | None
    model: str | None
    category: str | None
    message: str


def parse_log_line(line: str) -> LogLine:
    line = line.rstrip("\n")
    time = None
    if m := _ASCTIME_RE.match(line):
        time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    tags = {key: _none(m.group(1)) if (m := rx.search(line)) else None for key, rx in _TAG_RE.items()}
    job_tags = _JOB_TAGS_RE.findall(line)
    tags["job"] = _none(job_tags[0]) if job_tags else None
    tags["subjob"] = _none(job_tags[1]) if len(job_tags) > 1 else None

    category, message = None, line
    if time is not None:
        # asctime;JobID;[SubjobID|JobID;]CID;ModelID;category;message -> message may contain ';'.
        # The subjob tag is missing in some jobs, so anchor on the ModelID field.
        parts = line.split(";")
        model_idx = next((i for i, p in enumerate(parts[:6]) if p.startswith("ModelID ")), None)
        if model_idx is not None and len(parts) > model_idx + 2:
            category, message = parts[model_idx + 1], ";".join(parts[model_idx + 2:])
        else:
            # extra without the job tags: asctime;extra;category;message
            parts = line.split(";", 3)
            if len(parts) == 4:
                category, message = parts[2], parts[3]
    return LogLine(line, time, tags["job"], tags["subjob"], tags["cid"], tags["model"], category, message)


@dataclass
class ArchiveFile:
    name: str
    error: bool
    cid: str
    trigger: str
    job: str
    subjob: str | None
    finished_epoch: int  # UTC

    @property
    def ok(self) -> bool:
        return not self.error


def parse_archive_name(name: str) -> ArchiveFile | None:
    m = ARCHIVE_RE.match(name.strip())
    if not m:
        return None
    ts = datetime.strptime(m.group("ts"), "%Y_%m_%d-%H-%M-%S")
    return ArchiveFile(
        name=name.strip(),
        error=bool(m.group("error")),
        cid=m.group("cid"),
        trigger=m.group("trigger"),
        job=m.group("job"),
        subjob=_none(m.group("subjob")),
        finished_epoch=calendar.timegm(ts.timetuple()),
    )


# --- snapshot: one docker exec per container collecting everything `status` needs ---------

SNAPSHOT_SCRIPT = f"""
cd {LOG_DIR} 2>/dev/null || {{ echo "@@NOLOGDIR"; exit 0; }}
echo "@@NOW"; date +%s; date '+%Y-%m-%d %H:%M:%S'
echo "@@MTIME"; stat -c %Y logger.log 2>/dev/null
echo "@@HEAD"; head -n 1 logger.log 2>/dev/null
echo "@@TAIL"; tail -n 5 logger.log 2>/dev/null
echo "@@ARCHIVES"; ls -1t 2>/dev/null | grep '^logger\\.log_' | head -n 5
"""


@dataclass
class Snapshot:
    has_logdir: bool
    now_epoch: int | None = None
    now_local: datetime | None = None
    log_mtime: int | None = None
    first_line: LogLine | None = None
    last_lines: list[LogLine] | None = None
    recent_archives: list[ArchiveFile] | None = None


def split_sections(output: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in output.splitlines():
        if line.startswith("@@"):
            current = line[2:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def parse_snapshot(output: str) -> Snapshot:
    sections = split_sections(output)
    if "NOLOGDIR" in sections:
        return Snapshot(has_logdir=False)
    snap = Snapshot(has_logdir=True)
    now = sections.get("NOW", [])
    if len(now) >= 2:
        snap.now_epoch = int(now[0])
        snap.now_local = datetime.strptime(now[1], "%Y-%m-%d %H:%M:%S")
    mtime = [x for x in sections.get("MTIME", []) if x.strip()]
    if mtime:
        snap.log_mtime = int(mtime[0])
    head = [x for x in sections.get("HEAD", []) if x.strip()]
    if head:
        snap.first_line = parse_log_line(head[0])
    snap.last_lines = [parse_log_line(x) for x in sections.get("TAIL", []) if x.strip()]
    snap.recent_archives = [a for x in sections.get("ARCHIVES", []) if (a := parse_archive_name(x))]
    return snap


ERROR_PATTERN = r"error|exception|traceback|failed|killed|out of memory"
STDOUT_ERROR_RE = re.compile(
    r" - ERROR : |Traceback|Exception|Error Code: (?!None)\S+|error_\d", re.IGNORECASE
)
