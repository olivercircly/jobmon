from jobmon import collector
from jobmon.docker_api import Container, Stats
from jobmon.parse import parse_archive_name, parse_log_line, parse_snapshot

CID = "c2d51869-c362-4567-a4e8-2400ba63e477"
LINE = f"2026-11-05 10:12:01,123;JobID 69145;SubjobID 812;CID {CID};ModelID 7;02_Data_Preparation;Loading data; part 2"


def test_log_line_with_job_tags():
    line = parse_log_line(LINE)
    assert (line.job, line.subjob, line.cid, line.model) == ("69145", "812", CID, "7")
    assert line.category == "02_Data_Preparation"
    assert line.message == "Loading data; part 2"
    assert line.time.hour == 10


def test_log_line_deployed_format_two_jobid_tags():
    line = parse_log_line(f"2026-11-05 05:25:19,656;JobID 78031;JobID 37759;CID {CID};ModelID 0b59;db_insertion;Load ID file")
    assert (line.job, line.subjob, line.model) == ("78031", "37759", "0b59")
    assert line.category == "db_insertion" and line.message == "Load ID file"


def test_log_line_without_subjob_tag():
    line = parse_log_line(f"2026-11-06 09:23:58,488;JobID 80593;CID {CID};ModelID 989d;prediction;Finished saving scores!")
    assert (line.job, line.subjob, line.cid, line.model) == ("80593", None, CID, "989d")
    assert line.category == "prediction" and line.message == "Finished saving scores!"


def test_log_line_subjob_none_and_no_tags():
    line = parse_log_line(f"2026-11-05 10:12:01,123;JobID 1;SubjobID None;CID {CID};ModelID 7;entrypoint;x")
    assert line.subjob is None and line.job == "1"
    plain = parse_log_line("2026-11-05 10:12:01,123;;entrypoint;hello")
    assert plain.job is None and plain.category == "entrypoint" and plain.message == "hello"
    assert parse_log_line("Traceback (most recent call last):").time is None


def test_archive_names():
    ok = parse_archive_name(f"logger.log_{CID}_trigger_training_evaluation_53373_812_2026_11_05-09-30-00")
    assert ok and ok.ok and ok.job == "53373" and ok.subjob == "812"
    assert ok.trigger == "trigger_training_evaluation" and ok.cid == CID
    err = parse_archive_name(f"logger.log_ERROR_{CID}_trigger_inference_69145_None_2026_11_05-09-30-00")
    assert err and err.error and err.subjob is None and err.job == "69145"
    assert parse_archive_name("system.log_x_y_1_2026_11_05-09-30-00") is None
    assert parse_archive_name("logger.log") is None


SNAP = f"""@@NOW
1762337521
2026-11-05 10:12:01
@@MTIME
1762337400
@@HEAD
2026-11-05 08:00:01,000;JobID 69145;SubjobID 812;CID {CID};ModelID 7;entrypoint;Get_Job script triggered
@@TAIL
{LINE}
@@ARCHIVES
logger.log_ERROR_{CID}_trigger_inference_69000_None_2026_11_05-07-00-00
"""


def _container(state="running"):
    return Container("prod-ai-pipeline-gpu-30", 30, state, "Up 2 days")


def test_snapshot_running():
    snap = parse_snapshot(SNAP)
    cs = collector.build_state(_container(), snap, "", Stats(100.0, "1GiB"), stale_after_s=1800)
    assert cs.state == collector.RUNNING
    assert cs.job == "69145" and cs.runtime_s == 2 * 3600 + 12 * 60
    assert cs.since_last_log_s == 121
    assert cs.last_finished.error and cs.last_finished.job == "69000"


def test_snapshot_stale_idle_busy_down():
    snap = parse_snapshot(SNAP)
    stale = collector.build_state(_container(), snap, "", Stats(0.0, ""), stale_after_s=60)
    assert stale.state == collector.STALE and "CPU idle" in stale.detail

    empty = parse_snapshot("@@NOW\n1\n2026-11-05 10:00:00\n@@MTIME\n@@HEAD\n@@TAIL\n@@ARCHIVES\n")
    assert collector.build_state(_container(), empty, "", Stats(0.0, ""), 1800).state == collector.IDLE
    assert collector.build_state(_container(), empty, "", Stats(300.0, ""), 1800).state == collector.BUSY
    assert collector.build_state(_container("exited"), None, "", None, 1800).state == collector.DOWN
    assert collector.build_state(_container(), parse_snapshot("@@NOLOGDIR\n"), "", None, 1800).state == collector.ERROR
