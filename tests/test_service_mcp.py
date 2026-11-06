"""Service layer and MCP server against a fake `docker` CLI.

The fake runs `docker exec <c> sh -c <script>` with the real sh inside tmp/<c>/, which holds a
DevAI_utils/notifications/Logging tree - so the actual shell scripts are exercised.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap

import anyio
import pytest

from jobmon import service
from jobmon.parse import LOG_DIR

PREFIX = "test-ai-pipeline-gpu"
CID = "c2d51869-c362-4567-a4e8-2400ba63e477"

FAKE_DOCKER = textwrap.dedent("""\
    #!{python}
    import json, os, subprocess, sys
    root = os.environ["FAKE_DOCKER_ROOT"]
    args = sys.argv[1:]
    if args[0] == "ps":
        for name, state in (("{prefix}-1", "running"), ("{prefix}-2", "running"), ("{prefix}-3", "exited")):
            print(json.dumps({{"Names": name, "State": state, "Status": "Up 1 day" if state == "running" else "Exited (1)"}}))
    elif args[0] == "stats":
        for name in args[4:]:
            print(json.dumps({{"Name": name, "CPUPerc": "12.5%", "MemUsage": "1GiB / 8GiB"}}))
    elif args[0] == "exec":
        sys.exit(subprocess.run(args[2:], cwd=os.path.join(root, args[1])).returncode)
    elif args[0] == "logs":
        print("starting worker")
        print("Traceback (most recent call last):")
        print("ValueError: boom")
""")

RUNNING_LOG = (
    f"2026-11-05 08:00:01,000;JobID 69145;JobID 812;CID {CID};ModelID 7;entrypoint;Get_Job script triggered\n"
    f"2026-11-05 08:05:00,000;JobID 69145;JobID 812;CID {CID};ModelID 7;training;Exception in step 3\n"
    f"2026-11-05 08:06:00,000;JobID 69145;JobID 812;CID {CID};ModelID 7;training;epoch 1 done\n"
)
ERROR_ARCHIVE = f"logger.log_ERROR_{CID}_trigger_inference_69000_None_2026_11_05-07-00-00"
OK_ARCHIVE = f"logger.log_{CID}_trigger_inference_68000_None_2026_11_04-07-00-00"


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    bin_dir, root = tmp_path / "bin", tmp_path / "containers"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER.format(python=sys.executable, prefix=PREFIX))
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)

    busy = root / f"{PREFIX}-1" / LOG_DIR
    busy.mkdir(parents=True)
    (busy / "logger.log").write_text(RUNNING_LOG)
    (busy / ERROR_ARCHIVE).write_text("2026-11-05 06:59:00,000;JobID 69000;;x;y;inference;CUDA out of memory\n")
    idle = root / f"{PREFIX}-2" / LOG_DIR
    idle.mkdir(parents=True)
    (idle / "logger.log").write_text("")
    (idle / OK_ARCHIVE).write_text("all good\n")

    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_DOCKER_ROOT": str(root)}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


# --- service -------------------------------------------------------------------------------


def test_find_jobs_running_archived_and_missing(fake_docker):
    result = service.find_jobs(PREFIX, ["69145", "69000", "68000", "1"])
    by_job = {h.job: h for h in result.hits}
    assert by_job["69145"].result == "running" and by_job["69145"].container == f"{PREFIX}-1"
    assert by_job["69000"].result == "error" and by_job["69000"].log_file == ERROR_ARCHIVE
    assert by_job["68000"].result == "ok" and by_job["68000"].container == f"{PREFIX}-2"
    assert result.missing == ["1"]


def test_pick_log_file_and_errors(fake_docker):
    c1, c2 = f"{PREFIX}-1", f"{PREFIX}-2"
    assert service.pick_log_file(c1, None).name == "logger.log"
    assert service.pick_log_file(c1, "69000").name == ERROR_ARCHIVE
    assert service.pick_log_file(c2, None).description.startswith("last finished job 68000")
    with pytest.raises(service.JobmonError, match="No log for job 4711"):
        service.pick_log_file(c1, "4711")

    report = service.job_errors(c1, None, context=0, scan_stdout=True)
    assert len(report.lines) == 1 and "Exception in step 3" in report.lines[0]
    assert [line.text for line in report.stdout if line.hit] == ["Traceback (most recent call last):"]

    log, text = service.tail_log(c1, None, 1)
    assert text.strip().endswith("epoch 1 done")


@pytest.mark.parametrize("job", ["1; rm -rf /", "69145'", "abc"])
def test_job_ids_are_validated_before_reaching_the_shell(job):
    with pytest.raises(service.JobmonError, match="invalid job ID"):
        service.pick_log_file(f"{PREFIX}-1", job)


def test_resolve_container():
    assert service.resolve_container(PREFIX, "30") == f"{PREFIX}-30"
    with pytest.raises(service.JobmonError):
        service.resolve_container(PREFIX, "-x;ls")


# --- MCP server over real stdio ------------------------------------------------------------


def test_mcp_stdio_end_to_end(fake_docker):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "jobmon.mcp_server", "--env", "test"], env=fake_docker,
    )

    async def run():
        async with Client(params) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == {"status", "find_jobs", "job_errors", "tail_log"}
            assert all(t.annotations.read_only_hint for t in tools.values())

            status = (await client.call_tool("status", {})).structured_content
            assert status["counts"] == {"down": 1, "running": 1, "idle": 1}
            assert [c["state"] for c in status["containers"]] == ["down", "running"]
            assert status["containers"][1]["job"] == "69145" and status["idle_hidden"] == 1

            found = (await client.call_tool("find_jobs", {"ids": ["69000"]})).structured_content
            assert found["hits"][0]["result"] == "error"

            errors = (await client.call_tool("job_errors", {"container": "1", "job": "69000"})).structured_content
            assert any("out of memory" in line for line in errors["error_lines"])

            tail = (await client.call_tool("tail_log", {"container": "1", "lines": 2})).structured_content
            assert len(tail["lines"]) == 2

            # anticipated failures come back as readable tool errors, not crashes
            bad = await client.call_tool("job_errors", {"container": "other-container"})
            assert bad.is_error and "not a pipeline container" in bad.content[0].text
            missing = await client.call_tool("tail_log", {"container": "1", "job": "4711"})
            assert missing.is_error and "No log for job 4711" in missing.content[0].text

    anyio.run(run)
