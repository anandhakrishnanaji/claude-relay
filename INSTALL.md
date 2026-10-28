# Installing claude-relay

claude-relay runs as a small HTTP server that speaks the OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) APIs and forwards requests to Claude Code. Pick one of the four setups below.

| Setup | Best for | URL |
|---|---|---|
| [1. Run in the foreground](#1-run-in-the-foreground) | Trying it out, development | `http://localhost:8000/v1` |
| [2. Always-on service (systemd)](#2-always-on-service-linux-systemd) | Daily use on your own Linux machine | `http://claude.localhost:8787/v1` |
| [3. Docker](#3-docker) | Any machine with Docker | `http://localhost:8000/v1` |
| [4. Home server on your tailnet](#4-home-server-on-your-tailnet-docker--tsdproxy) | Reaching it from all your devices | `https://claude.<tailnet>.ts.net/v1` |

## Requirements

- **Python 3.10+** (3.12 recommended)
- **[uv](https://docs.astral.sh/uv/)**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Credentials for Claude**, one of:
  - a Claude subscription login: install the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) and run `claude auth login`
  - an Anthropic API key in `ANTHROPIC_API_KEY`
  - AWS Bedrock or Google Vertex AI (see the Configuration section of the [README](./README.md))

The Claude Code CLI is bundled with the Python SDK, so Node.js is not required for API-key, Bedrock or Vertex auth. You only need the standalone CLI to run `claude auth login`.

## Get the code

```bash
git clone https://github.com/anandhakrishnanaji/claude-relay
cd claude-relay
```

## 1. Run in the foreground

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
cp .env.example .env          # optional: edit settings
.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Stop it with Ctrl+C. Add `--reload` while developing, except on Windows (see [Troubleshooting](#troubleshooting)).

Poetry works too: `poetry install`, then `poetry run uvicorn src.main:app --port 8000`.

## 2. Always-on service (Linux, systemd)

```bash
./scripts/install-service.sh
```

The script:

1. creates `.venv` and installs the project
2. writes `.env` if there isn't one (CLI auth, `127.0.0.1:8787`, rate limiting off)
3. installs and starts the `claude-wrapper` systemd user service
4. enables linger so the service survives logout and starts at boot
5. waits for `/health` and prints the URL

It requires a logged-in Claude CLI. Any `*.localhost` name resolves to your own machine, so `http://claude.localhost:8787/v1` works with no DNS setup.

```bash
systemctl --user status claude-wrapper     # check it
systemctl --user restart claude-wrapper    # after git pull or editing .env
journalctl --user -u claude-wrapper -f     # live logs
PORT=9000 ./scripts/install-service.sh     # another port (delete .env first)
./scripts/install-service.sh --uninstall   # remove it
```

## 3. Docker

Use the image in `deploy/Dockerfile` (the `Dockerfile` in the repo root is equivalent). It runs as a non-root user, which the Claude CLI requires when tools are enabled.

```bash
docker build -f deploy/Dockerfile -t claude-wrapper:local .
docker run -d --name claude-wrapper --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v ~/.claude:/home/app/.claude \
  -e CLAUDE_CONFIG_DIR=/home/app/.claude \
  claude-wrapper:local
```

- The `~/.claude` mount is read-write so the CLI can refresh its login token.
- The image runs as uid 1000. If your user has a different uid, build with `--build-arg UID=$(id -u) --build-arg GID=$(id -g)`.
- On SELinux hosts (Fedora, RHEL, rootless Podman), mount with `-v ~/.claude:/home/app/.claude:Z`.
- To use an API key instead of a login, drop the mount and pass `-e ANTHROPIC_API_KEY=...`.

## 4. Home server on your tailnet (Docker + tsdproxy)

This serves claude-relay over HTTPS to devices on your [Tailscale](https://tailscale.com) network only, via [tsdproxy](https://github.com/almeidapaulopt/tsdproxy).

**On the server you need:** Docker, a running tsdproxy attached to an external Docker network named `proxy`, and a Claude login in `~/.claude`.

> **Check the tsdproxy auth key first.** Tailscale auth keys last at most 90 days. Existing proxies keep working after the key expires, but adding a new one makes tsdproxy register a new node, and with an expired key tsdproxy 1.4.7 crashes and takes every proxy down. Generate a fresh key before deploying.

```bash
# From your machine
ssh server 'mkdir -p ~/docker/claude-wrapper/app'
rsync -a pyproject.toml README.md src deploy server:docker/claude-wrapper/app/
scp deploy/tsdproxy/compose.yml server:docker/claude-wrapper/compose.yml

# On the server
cd ~/docker/claude-wrapper && docker compose up -d --build
```

It appears as `https://claude.<your-tailnet>.ts.net/v1`. Funnel is off, so it is not reachable from the public internet.

## Verify it works

```bash
curl -s http://localhost:8000/health

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Say hi"}]}'
```

Swap in your URL from the table above. From Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Say hi"}],
)
print(reply.choices[0].message.content)
```

## Securing it

claude-relay spends your Claude subscription or API credit, and with tools enabled a request can make Claude run shell commands on the host.

- Keep it bound to `127.0.0.1` unless you need it elsewhere.
- Before exposing it on a network, set `API_KEY` in `.env`. Clients then send it as `Authorization: Bearer <key>`.
- `CORS_ORIGINS` lists the browser origins that may call the server, as a JSON list such as `["https://app.example.com"]`. The default `[]` allows none, so other websites you visit can't call it from your browser. Avoid `["*"]`: without `API_KEY` it lets any website use the server.
- `ALLOWED_HOSTS` lists the `Host` names the server answers to (comma-separated, wildcards like `*.example.com` allowed), which blocks DNS-rebinding attacks. The default covers `localhost`, `127.0.0.1`, `::1` and `*.localhost`. If clients reach it by another name, such as a LAN IP or a Docker container name, add it, or requests get `400 Invalid host header`.
- Tools are off unless you set `ALLOW_TOOLS=true`. Until then, a request with `"enable_tools": true` gets a `403`. Only turn it on if every client that can reach the server is trusted, and set `API_KEY` too.

## Updating

```bash
git pull
uv pip install --python .venv/bin/python -e .   # if dependencies changed
systemctl --user restart claude-wrapper           # setup 2
docker compose up -d --build                      # setups 3/4, in the compose directory
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Not logged in` / `401` from Claude | Run `claude auth login` (on the host whose `~/.claude` is mounted), then restart the service or container. |
| `--dangerously-skip-permissions cannot be used with root` | The container runs as root. Rebuild from this repo's `Dockerfile` or `deploy/Dockerfile`, which run as a non-root user. |
| `400 Invalid host header` | The client uses a host name not in `ALLOWED_HOSTS`. Add it, e.g. `ALLOWED_HOSTS=localhost,127.0.0.1,192.168.1.10`. |
| `403` with `tools_disabled` | The request sent `"enable_tools": true` but the server doesn't set `ALLOW_TOOLS=true`. |
| `Permission denied` on `~/.claude` in Docker | uid mismatch: rebuild with `--build-arg UID=$(id -u)`. On SELinux, add `:Z` to the mount. |
| Windows: `Failed to start Claude Code` / `claude not found` | Run uvicorn **without** `--reload`. WSL also works. |
| `address already in use` | Another process has the port. Pick another with `PORT=...` (and `--port` for uvicorn). |
| Service won't start after reboot | Check `loginctl show-user $USER -p Linger` says `yes`, then `journalctl --user -u claude-wrapper -n 50`. |
