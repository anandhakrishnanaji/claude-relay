# Installing claude-relay

claude-relay runs as a small HTTP server that speaks the OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic (`/v1/messages`) APIs and forwards requests to Claude Code. Pick one of the setups below.

| Setup | Best for | URL |
|---|---|---|
| [1. Run it when you need it](#1-run-it-when-you-need-it) | Local development (most people) | `http://localhost:8000/v1` |
| [2. Always-on service (systemd)](#2-always-on-service-linux-systemd) | Daily use on your own Linux machine | `http://claude.localhost:8787/v1` |
| [3. Docker](#3-docker) | Any machine with Docker | `http://localhost:8000/v1` |

## Requirements

- **[uv](https://docs.astral.sh/uv/)** (`curl -LsSf https://astral.sh/uv/install.sh | sh`) or pipx. uv fetches Python 3.10+ if you don't have it.
- **Credentials for Claude**, one of:
  - a Claude subscription login: install the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) and run `claude auth login`
  - an Anthropic API key in `ANTHROPIC_API_KEY`
  - AWS Bedrock or Google Vertex AI (see the Configuration section of the [README](./README.md))

The Claude Code CLI is bundled with the Python SDK, so Node.js is not required for API-key, Bedrock or Vertex auth. You only need the standalone CLI to run `claude auth login`.

## 1. Run it when you need it

Start it from a terminal while you're developing and stop it with Ctrl+C:

```bash
uvx claude-relay-server
```

Or install the `claude-relay` command once:

```bash
uv tool install claude-relay-server     # or: pipx install claude-relay-server
claude-relay
```

It listens on `127.0.0.1:8000`. Useful flags:

```bash
claude-relay --port 9000
claude-relay --generate-api-key         # require a bearer token; prints it
claude-relay --env-file ./relay.env     # settings file (default: ./.env)
claude-relay --help
```

Settings come from flags, then environment variables, then a `.env` file in the current directory. See [.env.example](./.env.example) for all of them.

### From source

To work on claude-relay itself:

```bash
git clone https://github.com/anandhakrishnanaji/claude-relay
cd claude-relay
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
.venv/bin/uvicorn claude_relay.main:app --reload --port 8000
```

Drop `--reload` on Windows (see [Troubleshooting](#troubleshooting)). Poetry works too: `poetry install`, then `poetry run claude-relay`.

## 2. Always-on service (Linux, systemd)

This runs from a source checkout:

```bash
git clone https://github.com/anandhakrishnanaji/claude-relay
cd claude-relay
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

Each release publishes a non-root image to GHCR. The Claude CLI requires a non-root user when tools are enabled.

```bash
docker run -d --name claude-relay --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v ~/.claude:/home/app/.claude \
  -e CLAUDE_CONFIG_DIR=/home/app/.claude \
  ghcr.io/anandhakrishnanaji/claude-relay:latest
```

- The `~/.claude` mount is read-write so the CLI can refresh its login token.
- The image runs as uid 1000. If your user has a different uid, build it yourself from a checkout: `docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t claude-relay .`
- On SELinux hosts (Fedora, RHEL, rootless Podman), mount with `-v ~/.claude:/home/app/.claude:Z`.
- To use an API key instead of a login, drop the mount and pass `-e ANTHROPIC_API_KEY=...`.
- For a headless subscription login, use `CLAUDE_CODE_OAUTH_TOKEN` instead of mounting `~/.claude`: run `claude setup-token` on any machine where you can log in, then drop the mount and pass `-e CLAUDE_CODE_OAUTH_TOKEN=...`.
- To send the CLI's API traffic through a proxy or gateway, pass `-e ANTHROPIC_BASE_URL=https://...`.

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

- It binds to `127.0.0.1` by default. Keep it that way unless you need it elsewhere.
- Before exposing it on a network (`--host 0.0.0.0`), set an API key with `--api-key`, `--generate-api-key` or `API_KEY`. Clients then send it as `Authorization: Bearer <key>`. claude-relay warns at startup if you don't.
- `CORS_ORIGINS` lists the browser origins that may call the server, as a JSON list such as `["https://app.example.com"]`. The default `[]` allows none, so other websites you visit can't call it from your browser. Avoid `["*"]`: without `API_KEY` it lets any website use the server.
- `ALLOWED_HOSTS` lists the `Host` names the server answers to (comma-separated, wildcards like `*.example.com` allowed), which blocks DNS-rebinding attacks. The default covers `localhost`, `127.0.0.1`, `::1` and `*.localhost`. If clients reach it by another name, such as a LAN IP or a Docker container name, add it, or requests get `400 Invalid host header`.
- Tools are off unless you set `ALLOW_TOOLS=true`. Until then, a request with `"enable_tools": true` gets a `403`. Only turn it on if every client that can reach the server is trusted, and set `API_KEY` too.

## Updating

```bash
uv tool upgrade claude-relay-server               # setup 1 (uvx always runs the latest)
git pull && ./scripts/install-service.sh          # setup 2
docker pull ghcr.io/anandhakrishnanaji/claude-relay:latest   # setup 3, then recreate the container
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` with `claude_cli_not_authenticated` (or `Not logged in`) | The CLI's login is missing or expired. Run `claude auth login` (on the host whose `~/.claude` is mounted), or renew `CLAUDE_CODE_OAUTH_TOKEN`, then restart the service or container. `GET /v1/auth/status` shows the last failure under `last_auth_failure` until a request succeeds. |
| `--dangerously-skip-permissions cannot be used with root` | The container runs as root. Rebuild from this repo's `Dockerfile`, which runs as a non-root user. |
| `400 Invalid host header` | The client uses a host name not in `ALLOWED_HOSTS`. Add it, e.g. `ALLOWED_HOSTS=localhost,127.0.0.1,192.168.1.10`. |
| `403` with `tools_disabled` | The request sent `"enable_tools": true` but the server doesn't set `ALLOW_TOOLS=true`. |
| `Permission denied` on `~/.claude` in Docker | uid mismatch: rebuild with `--build-arg UID=$(id -u)`. On SELinux, add `:Z` to the mount. |
| Windows: `Failed to start Claude Code` / `claude not found` | Run uvicorn **without** `--reload`. WSL also works. |
| `port 8000 on 127.0.0.1 is already in use` | Another process has the port. Pick another with `--port`. |
| Service won't start after reboot | Check `loginctl show-user $USER -p Linger` says `yes`, then `journalctl --user -u claude-wrapper -n 50`. |
