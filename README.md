# jobmon

Monitor and debug AI-Pipeline jobs running in the `<env>-ai-pipeline-gpu-<N>` containers.
Read-only: it only lists containers, reads `docker stats`/`docker logs`, and reads files via `docker exec`.

## Install

```bash
cd ~/repos/utilities/jobmon
uv tool install -e .     # puts `jobmon` on your PATH
# or without installing: uv run jobmon ...
```

## Usage

```bash
jobmon                       # = jobmon status, prod, hides idle containers
jobmon status -w             # live view, refresh every 10s (-w 30 for 30s)
jobmon status -a -s          # all containers, problems first
jobmon --env staging status  # or: export JOBMON_ENV=staging

jobmon find 69145 69146      # where does/did a job (or CID) run, OK or ERROR?
jobmon errors 30             # error lines of container 30's current (or last) job
jobmon errors 30 -j 69145    # ... of a specific job; --stdout also scans docker logs
jobmon logs 30 -f            # follow the current job log
```

States: `running`, `stale` (no log output for `--stale-minutes`, default 30), `busy?` (no job in
`logger.log` but CPU ≥ 50%), `idle`, `down` (container not running), `error` (could not inspect).

## MCP server

`jobmon-mcp` exposes the same read-only functions as MCP tools: `status`, `find_jobs`, `job_errors`,
`tail_log`. The default transport is stdio, so the MCP client starts the server itself and nothing
runs in the background.

```bash
# Claude Code on the Docker host (after `uv tool install -e .`)
claude mcp add jobmon -- jobmon-mcp
# ... or without installing
claude mcp add jobmon -- uv run --directory ~/repos/utilities/jobmon jobmon-mcp

# Claude Code on your laptop, containers on a server: run it over SSH
claude mcp add jobmon -- ssh gpu-host jobmon-mcp
ssh gpu-host true | wc -c    # must print 0: shell startup output on stdout breaks the protocol

# staging by default (tools also take an `env` argument)
claude mcp add jobmon-staging -- jobmon-mcp --env staging

# inspect / try the tools interactively
npx @modelcontextprotocol/inspector uv run jobmon-mcp
```

`jobmon-mcp --http [--host 127.0.0.1] [--port 8000]` serves streamable HTTP at `/mcp` instead, for a
long-running shared service. It has no authentication, so keep it on localhost or put it behind an auth proxy.

In stdio mode stdout carries the protocol: server code must never print. `service.py` returns data
only, and docker subprocesses get `stdin=DEVNULL`.

## How it works

The AI-Pipeline logs the current job to `DevAI_utils/notifications/Logging/logger.log` inside the
container, every line tagged `<time>;JobID <job>;JobID <subjob>;CID <cid>;ModelID <model>;<module>;<msg>`.
When a job ends, `create_historic_logs()` (DevAI_utils) moves it to
`logger.log[_ERROR_]<cid>_<trigger>_<job>_<subjob>_<UTC timestamp>` and empties `logger.log`.

So: non-empty `logger.log` = job running; archive filename = job history including OK/ERROR.

## Tests

```bash
uv run pytest
```
