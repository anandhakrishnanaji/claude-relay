<!--
  claude-relay: Claude Code API server. A self-hosted OpenAI-compatible and Anthropic-compatible API server for Claude Code.
  Use your Claude subscription (Pro/Max) or an Anthropic API key from any OpenAI SDK or Anthropic SDK client.
  Python, FastAPI, Claude Agent SDK. MIT licensed.
-->

# claude-relay: Claude Code API server

<p>
  <a href="https://github.com/anandhakrishnanaji/claude-relay/actions/workflows/ci.yml"><img src="https://github.com/anandhakrishnanaji/claude-relay/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python 3.10 to 3.13" />
  <img src="https://img.shields.io/badge/API-OpenAI%20%2B%20Anthropic%20compatible-8A2BE2" alt="OpenAI and Anthropic compatible API" />
</p>

**claude-relay is a self-hosted Claude Code API server. It exposes Claude Code through the OpenAI Chat Completions API (`/v1/chat/completions`) and the Anthropic Messages API (`/v1/messages`).** Point any OpenAI or Anthropic client library at it by changing the base URL, and your requests run through Claude Code using your Claude subscription login or an Anthropic API key.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
reply = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello, Claude"}],
)
print(reply.choices[0].message.content)
```

**Run it:**

```bash
claude auth login          # or: export ANTHROPIC_API_KEY=...
uvx claude-relay-server    # serves http://localhost:8000/v1
```

See [INSTALL.md](./INSTALL.md) for pipx/pip, an always-on service and Docker.

## Highlights

- **Two APIs, one server.** OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages`. Chat Completions supports streaming (SSE).
- **Honours system prompts.** Your system message replaces Claude Code's own prompt on every endpoint, and `/v1/messages` also accepts `system` as a list of text blocks, as the Anthropic SDK sends it.
- **Tools off by default.** Both endpoints run a single turn without tools, so untrusted text in a prompt can't run commands. Send `"enable_tools": true` to give Claude its Bash/Read/Write tools.
- **Isolated from host settings.** Requests don't load the host's `~/.claude/CLAUDE.md`, settings, skills or plugins. Set `CLAUDE_SETTING_SOURCES=user,project,local` to opt back in.
- **Flexible auth.** Claude CLI login (subscription), `ANTHROPIC_API_KEY`, AWS Bedrock or Google Vertex AI, auto-detected.
- **Ships deploy tooling.** `scripts/install-service.sh` sets up an always-on systemd user service, and `deploy/Dockerfile` is a non-root production image. See [Always-on local service](#always-on-local-service-linux-systemd) and [Docker](#docker).

## What is claude-relay for?

claude-relay is for developers who want to call Claude Code from tools and code written against the OpenAI or Anthropic APIs. Typical uses:

- Using an existing **OpenAI Python or JavaScript SDK** app with Claude, with only a base-URL change.
- Connecting **chat UIs, editors and agent frameworks** that accept a custom OpenAI-compatible endpoint.
- Scripting Claude Code over **plain HTTP/curl** instead of the CLI.

It runs on the official [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/), which bundles the Claude Code CLI, so no separate Node.js install is needed.

## Version

**Current Version:** 3.0.0
- **Published on PyPI** as `claude-relay-server`: run it with `uvx claude-relay-server`, or install the `claude-relay` command with `uv tool install` / `pipx install`
- **New `claude-relay` command** with `--host`, `--port`, `--api-key`, `--generate-api-key` and `--env-file`
- **Binds to `127.0.0.1` by default**, and warns if you expose it without an API key
- **No startup prompt**: the interactive "Enable API key protection?" question is gone, so it runs cleanly from scripts and services
- **Docker image** at `ghcr.io/anandhakrishnanaji/claude-relay`

**Upgrading from 2.x:** the Python package moved from `src` to `claude_relay`. Replace `uvicorn src.main:app` with `claude-relay` (or `uvicorn claude_relay.main:app`). If you used the systemd service, rerun `./scripts/install-service.sh`. If you relied on the server listening on all interfaces, pass `--host 0.0.0.0` and set an API key.

**2.3.0:**
- **Live `/v1/models` discovery:** When `ANTHROPIC_API_KEY` is set, the wrapper fetches Anthropic's live model list (cached) instead of serving the static fallback
- **Dynamic default Sonnet:** `DEFAULT_MODEL` resolves to the latest Sonnet at startup when `ANTHROPIC_API_KEY` is configured; falls back to `claude-sonnet-4-6` otherwise
- **Operator overrides:** New `CLAUDE_MODELS_OVERRIDE`, `FAST_MODEL`, and `MODEL_LIST_*` env vars
- **Updated catalog:** Claude 4.6 family added to the static fallback list

**Migration Resources:**
- [docs/MIGRATION_STATUS.md](./docs/MIGRATION_STATUS.md) - Detailed v2.0.0 migration status
- [docs/UPGRADE_PLAN.md](./docs/UPGRADE_PLAN.md) - Comprehensive migration strategy and technical details

## Features

### 🔥 **Core API Compatibility**
- OpenAI-compatible `/v1/chat/completions` endpoint
- Anthropic-compatible `/v1/messages` endpoint
- Support for both streaming and non-streaming responses
- Compatible with OpenAI Python SDK and all OpenAI client libraries
- Automatic model validation and selection

### 🛠 **Claude Agent SDK Integration**
- **Official Claude Agent SDK** integration
- **Real-time cost tracking** - actual costs from SDK metadata
- **Accurate token counting** - input/output tokens from SDK
- **Session management** - proper session IDs and continuity
- **Enhanced error handling** with detailed authentication diagnostics
- **Modern SDK features** - Latest capabilities and improvements

### 🔐 **Multi-Provider Authentication**
- **Automatic detection** of authentication method
- **Claude CLI auth** - works with existing `claude auth` setup
- **Direct API key** - `ANTHROPIC_API_KEY` environment variable
- **AWS Bedrock** - enterprise authentication with AWS credentials
- **Google Vertex AI** - GCP authentication support

### ⚡ **Advanced Features**
- **System prompt support** via SDK options
- **Optional tool usage** - Enable Claude Code tools (Read, Write, Bash, etc.) when needed
- **Fast default mode** - Tools disabled by default for OpenAI API compatibility
- **Optional API key protection** - `--api-key` or `--generate-api-key`
- **Comprehensive logging** and debugging capabilities

### 🌐 **Interactive Landing Page**
- **API Explorer** at root URL (`http://localhost:8000/`)
- **Live endpoint testing** - Expandable accordions fetch real-time data
- **Light/dark theme toggle** - Persists preference in localStorage
- **Copy-to-clipboard** - One-click copy for Quick Start commands
- **Version badge** and GitHub link

## Quick Start

You need Python 3.10+ (uv fetches it for you) and credentials for Claude:

- a Claude subscription: install the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) and run `claude auth login`, or
- an Anthropic API key in `ANTHROPIC_API_KEY`, or
- AWS Bedrock / Google Vertex AI (see [Configuration](#configuration)).

```bash
uvx claude-relay-server
```

That downloads and starts the server on `http://localhost:8000`. Point your client at `http://localhost:8000/v1` (OpenAI) or `http://localhost:8000` (Anthropic) and stop it with Ctrl+C when you're done.

> **Note:** The Claude Code CLI is bundled with the Claude Agent SDK, so Node.js is only needed if you want the standalone CLI for `claude auth login`.

## Installation

To keep the `claude-relay` command installed instead of using `uvx`:

```bash
uv tool install claude-relay-server   # or: pipx install claude-relay-server
claude-relay                          # start it
uv tool upgrade claude-relay-server   # later, to update
```

`pip install claude-relay-server` into a virtualenv works too.

**From source** (to work on claude-relay itself):

```bash
git clone https://github.com/anandhakrishnanaji/claude-relay
cd claude-relay
uv venv .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/uvicorn claude_relay.main:app --reload --port 8000   # auto-reloads on code changes
```

## Configuration

Settings come from command-line flags, then environment variables, then a `.env` file in the directory you run `claude-relay` from (or the file given with `--env-file`). See [.env.example](./.env.example) for every option. The common ones:

```env
# Claude CLI path (usually just "claude")
CLAUDE_CLI_PATH=claude

# Explicit authentication method (optional)
# Options: cli, api_key, bedrock, vertex
# If not set, auto-detects based on available credentials
# CLAUDE_AUTH_METHOD=cli

# Optional API key clients must send as "Authorization: Bearer <key>"
# API_KEY=your-optional-api-key

# Interface and port (same as --host / --port)
CLAUDE_WRAPPER_HOST=127.0.0.1
PORT=8000

# Timeout in milliseconds
MAX_TIMEOUT=600000

# CORS origins
CORS_ORIGINS=["*"]

# Working directory for Claude Code (optional)
# If not set, uses an isolated temporary directory for security
# CLAUDE_CWD=/path/to/your/workspace
```

### 📁 **Working Directory Configuration**

By default, Claude Code runs in an **isolated temporary directory** to prevent it from accessing the wrapper's source code. This enhances security by ensuring Claude Code only has access to the workspace you intend.

**Configuration Options:**

1. **Default (Recommended)**: Automatically creates a temporary isolated workspace
   ```bash
   # No configuration needed - secure by default
   claude-relay
   ```

2. **Custom Directory**: Set a specific workspace directory
   ```bash
   export CLAUDE_CWD=/path/to/your/project
   claude-relay
   ```

3. **Via .env file**: Add to your `.env` file
   ```env
   CLAUDE_CWD=/home/user/my-workspace
   ```

**Important Notes:**
- The temporary directory is automatically cleaned up when the server stops
- This prevents Claude Code from accidentally modifying the wrapper's own code
- Cross-platform compatible (Windows, macOS, Linux)

### 🔐 **API Security Configuration**

By default the server listens on `127.0.0.1` with no API key, which is what you want for local development: only programs on your machine can reach it.

To require a key, pass one or let claude-relay generate one:

```bash
claude-relay --api-key my-secret          # or set API_KEY in the environment / .env
claude-relay --generate-api-key           # prints a random key for this run
```

Clients then send `Authorization: Bearer <key>` (with the OpenAI SDK, pass it as `api_key`).

If you listen on another interface (`--host 0.0.0.0`), set an API key. claude-relay warns at startup if you don't, because anyone who can reach the port can use your Claude account.

### 🛡️ **Rate Limiting**

Built-in rate limiting protects against abuse and ensures fair usage:

- **Chat Completions** (`/v1/chat/completions`): 10 requests/minute
- **Debug Requests** (`/v1/debug/request`): 2 requests/minute
- **Auth Status** (`/v1/auth/status`): 10 requests/minute
- **Health Check** (`/health`): 30 requests/minute

Rate limits are applied per IP address using a fixed window algorithm. When exceeded, the API returns HTTP 429 with a structured error response:

```json
{
  "error": {
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "type": "rate_limit_exceeded",
    "code": "too_many_requests",
    "retry_after": 60
  }
}
```

Configure rate limiting through environment variables:

```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_CHAT_PER_MINUTE=10
RATE_LIMIT_DEBUG_PER_MINUTE=2
RATE_LIMIT_AUTH_PER_MINUTE=10
RATE_LIMIT_HEALTH_PER_MINUTE=30
```

## Running the Server

```bash
claude-relay                              # http://localhost:8000
claude-relay --port 9000                  # another port
claude-relay --host 0.0.0.0 --api-key KEY # reachable from your network
claude-relay --env-file ~/claude-relay.env
claude-relay --debug                      # verbose logging
claude-relay --help
```

`python -m claude_relay` runs the same command. If the port is taken, claude-relay exits with an error instead of silently picking another port, so your clients' base URL stays correct.

To check your Claude credentials before starting:

```bash
claude --version
claude --print --model claude-haiku-4-5-20251001 "Hello"
```

### Always-on local service (Linux, systemd)

Run the wrapper in the background at a fixed URL, starting at boot and restarting on crashes. It uses your Claude subscription login, so no `ANTHROPIC_API_KEY` is needed.

**Requirements:** [uv](https://docs.astral.sh/uv/) and a logged-in Claude CLI (`claude auth login`).

```bash
./scripts/install-service.sh
```

The script:
- creates `.venv` (Python 3.12) and installs the project
- writes `.env` if it doesn't exist (CLI auth, `127.0.0.1:8787`, rate limiting off)
- installs and starts the `claude-wrapper` systemd user service
- enables linger so the service keeps running after you log out
- waits for `/health` and prints the URL

Your fixed URL is **`http://claude.localhost:8787/v1`**. Any `*.localhost` name resolves to your own machine, so no DNS or `/etc/hosts` setup is needed.

```python
from openai import OpenAI
client = OpenAI(base_url="http://claude.localhost:8787/v1", api_key="not-needed")
```

The service listens on `127.0.0.1` only, so other devices can't reach it. To expose it on your network, set `HOST=0.0.0.0` and add an `API_KEY` to `.env` so nobody else can use your subscription.

**Managing the service:**
```bash
systemctl --user status claude-wrapper     # check it
systemctl --user restart claude-wrapper    # after git pull or editing .env
journalctl --user -u claude-wrapper -f     # live logs
PORT=9000 ./scripts/install-service.sh     # reinstall on another port (delete .env first)
./scripts/install-service.sh --uninstall   # remove the service
```

After a `git pull` that changes dependencies, rerun `./scripts/install-service.sh`. If your Claude login expires, run `claude auth login` and restart the service.

## Docker

Each release publishes a multi-arch image (amd64, arm64) to GHCR. It runs as a non-root user (uid 1000), which the Claude CLI requires when tools are enabled.

```bash
docker run -d --name claude-relay --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v ~/.claude:/home/app/.claude \
  -e CLAUDE_CONFIG_DIR=/home/app/.claude \
  ghcr.io/anandhakrishnanaji/claude-relay:latest
```

- The `~/.claude` mount is read-write so the CLI can refresh its login token. To use an API key instead, drop the mount and pass `-e ANTHROPIC_API_KEY=...`.
- If your uid isn't 1000, build the image yourself: `docker build -f deploy/Dockerfile --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t claude-relay .`
- To give Claude a workspace, add `-v /path/to/project:/workspace -e CLAUDE_CWD=/workspace`.
- `-p 127.0.0.1:8000:8000` keeps it local. If you publish the port more widely, add `-e API_KEY=...`.

The `Dockerfile` in the repo root is a development image (runs as root with auto-reload) that the included `docker-compose.yml` builds. Use `deploy/Dockerfile` for anything long-running.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_WRAPPER_HOST` | Interface to bind (`--host`) | `127.0.0.1` (`0.0.0.0` in the Docker image) |
| `PORT` | Server port (`--port`) | `8000` |
| `API_KEY` | Key clients must send as a bearer token (`--api-key`) | - |
| `MAX_TIMEOUT` | Request timeout (seconds) | `300` |
| `CLAUDE_CWD` | Working directory | temp dir |
| `CLAUDE_SETTING_SOURCES` | Comma-separated filesystem settings the CLI loads per request (`user`, `project`, `local`). Empty isolates requests from the host's `~/.claude/CLAUDE.md`, settings, skills and plugins. | empty |
| `CLAUDE_AUTH_METHOD` | Auth method: `cli`, `api_key`, `bedrock`, `vertex` | auto-detect |
| `ANTHROPIC_API_KEY` | Direct Anthropic API key. Optional — also unlocks live `/v1/models` discovery and dynamic latest-Sonnet default. Not required when using Bedrock, Vertex, or Claude CLI subscription auth. | - |
| `API_KEYS` | Comma-separated client API keys | - |
| `DEFAULT_MODEL` | Override the default model. When unset and `ANTHROPIC_API_KEY` is configured, the wrapper resolves the latest Sonnet at startup; otherwise falls back to `claude-sonnet-4-6`. | auto |
| `FAST_MODEL` | Speed/cost-optimized model alias. | `claude-haiku-4-5-20251001` |
| `CLAUDE_MODELS_OVERRIDE` | Comma-separated model IDs to advertise via `/v1/models`. Takes precedence over both live and static lists. | - |
| `MODEL_LIST_CACHE_TTL_SECONDS` | Cache TTL for live `/v1/models` results. | `3600` |
| `MODEL_LIST_ERROR_TTL_SECONDS` | Short cache TTL applied when the live fetch fails, so transient outages don't suppress live discovery for the full hour. | `60` |
| `MODEL_LIST_REQUEST_TIMEOUT_SECONDS` | HTTP timeout for the live model fetch. | `5` |

### Management

```bash
docker logs -f claude-relay        # View logs
docker stop claude-relay           # Stop
docker start claude-relay          # Start
docker rm claude-relay             # Remove
```

### Test

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

## Usage Examples

### Using curl

```bash
# Basic chat completion (no auth)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'

# With API key protection (when enabled)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-generated-api-key" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [
      {"role": "user", "content": "Write a Python hello world script"}
    ],
    "stream": true
  }'
```

### Using OpenAI Python SDK

```python
from openai import OpenAI

# Configure client (automatically detects auth requirements)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key-if-required"  # Only needed if protection enabled
)

# Alternative: Let examples auto-detect authentication
# The wrapper's example files automatically check server auth status

# Basic chat completion
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What files are in the current directory?"}
    ]
)

print(response.choices[0].message.content)
# Output: Fast response without tool usage (default behaviour)

# Enable tools when you need them (e.g., to read files)
response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "user", "content": "What files are in the current directory?"}
    ],
    extra_body={"enable_tools": True}  # Enable tools for file access
)
print(response.choices[0].message.content)
# Output: Claude will actually read your directory and list the files!

# Check real costs and tokens
print(f"Cost: ${response.usage.total_tokens * 0.000003:.6f}")  # Real cost tracking
print(f"Tokens: {response.usage.total_tokens} ({response.usage.prompt_tokens} + {response.usage.completion_tokens})")

# Streaming
stream = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Supported Models

The wrapper exposes Claude's full model catalog. When `ANTHROPIC_API_KEY` is set, `/v1/models` returns Anthropic's live list (cached for 1 hour) and the wrapper picks the latest Sonnet as `DEFAULT_MODEL` at startup. When the key is absent — for example, when running with Bedrock, Vertex, or Claude CLI subscription auth — the static list below is served and `claude-sonnet-4-6` is used as the fallback default. Operators who want a curated list regardless of auth can set `CLAUDE_MODELS_OVERRIDE`.

### Claude 4.6 Family (Latest)
- **`claude-opus-4-6`** 🎯 Most capable
- **`claude-sonnet-4-6`** ⭐ Recommended — best coding model

### Claude 4.5 Family (Fall 2025)
- `claude-opus-4-5-20250929` — deep reasoning and coding
- `claude-sonnet-4-5-20250929` — agents and coding
- **`claude-haiku-4-5-20251001`** ⚡ Fast & cheap

### Claude 4.1 Family
- `claude-opus-4-1-20250805` — upgraded Opus 4

**Note:** Claude 3.x models are not supported by the Claude Agent SDK. The model parameter is passed to Claude Code via the SDK's model selection.

## Session Continuity 🆕

The wrapper now supports **session continuity**, allowing you to maintain conversation context across multiple requests. This is a powerful feature that goes beyond the standard OpenAI API.

### How It Works

- **Stateless Mode** (default): Each request is independent, just like the standard OpenAI API
- **Session Mode**: Include a `session_id` to maintain conversation history across requests

### Using Sessions with OpenAI SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Start a conversation with session continuity
response1 = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "user", "content": "Hello! My name is Alice and I'm learning Python."}
    ],
    extra_body={"session_id": "my-learning-session"}
)

# Continue the conversation - Claude remembers the context
response2 = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "user", "content": "What's my name and what am I learning?"}
    ],
    extra_body={"session_id": "my-learning-session"}  # Same session ID
)
# Claude will remember: "Your name is Alice and you're learning Python."
```

### Using Sessions with curl

```bash
# First message (add -H "Authorization: Bearer your-key" if auth enabled)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "My favourite color is blue."}],
    "session_id": "my-session"
  }'

# Follow-up message - context is maintained
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "What's my favourite color?"}],
    "session_id": "my-session"
  }'
```

### Session Management

The wrapper provides endpoints to manage active sessions:

- `GET /v1/sessions` - List all active sessions
- `GET /v1/sessions/{session_id}` - Get session details
- `DELETE /v1/sessions/{session_id}` - Delete a session
- `GET /v1/sessions/stats` - Get session statistics

```bash
# List active sessions
curl http://localhost:8000/v1/sessions

# Get session details
curl http://localhost:8000/v1/sessions/my-session

# Delete a session
curl -X DELETE http://localhost:8000/v1/sessions/my-session
```

### Session Features

- **Automatic Expiration**: Sessions expire after 1 hour of inactivity
- **Streaming Support**: Session continuity works with both streaming and non-streaming requests
- **Memory Persistence**: Full conversation history is maintained within the session
- **Efficient Storage**: Only active sessions are kept in memory

### Examples

See `examples/session_continuity.py` for comprehensive Python examples and `examples/session_curl_example.sh` for curl examples.

## API Endpoints

### Core Endpoints
- `GET /` - Interactive landing page with API explorer
- `POST /v1/chat/completions` - OpenAI-compatible chat completions (supports `session_id`)
- `POST /v1/messages` - Anthropic-compatible messages endpoint
- `GET /v1/models` - List available models
- `GET /v1/auth/status` - Check authentication status and configuration
- `GET /version` - Get API version
- `GET /health` - Health check endpoint

### Session Management Endpoints 🆕
- `GET /v1/sessions` - List all active sessions
- `GET /v1/sessions/{session_id}` - Get detailed session information
- `DELETE /v1/sessions/{session_id}` - Delete a specific session
- `GET /v1/sessions/stats` - Get session manager statistics

## Limitations & Roadmap

### 🚫 **Current Limitations**
- **Images in messages** are converted to text placeholders
- **Function calling** not supported (tools work automatically based on prompts)
- **OpenAI parameters** not yet mapped: `temperature`, `top_p`, `max_tokens`, `logit_bias`, `presence_penalty`, `frequency_penalty`
- **Multiple responses** (`n > 1`) not supported

### 🛣 **Planned Enhancements** 
- [ ] **Tool configuration** - allowed/disallowed tools endpoints  
- [ ] **OpenAI parameter mapping** - temperature, top_p, max_tokens support
- [ ] **Enhanced streaming** - better chunk handling
- [ ] **MCP integration** - Model Context Protocol server support

### ✅ **Recent Improvements (v2.2.0)**
- **Interactive Landing Page**: API explorer with live endpoint testing
- **Anthropic Messages API**: Native `/v1/messages` endpoint
- **Explicit Auth Selection**: `CLAUDE_AUTH_METHOD` env var
- **Tool Execution Fix**: `enable_tools: true` now works correctly

### ✅ **v2.0.0 - v2.1.0 Features**
- Claude Agent SDK v0.1.18 with bundled CLI
- Multi-provider auth (CLI, API key, Bedrock, Vertex AI)
- Session continuity and management
- Real-time cost and token tracking
- System prompt support

## Troubleshooting

1. **Claude CLI not found**:
   ```bash
   # Check Claude is in PATH
   which claude
   # Update CLAUDE_CLI_PATH in .env if needed
   ```

2. **Authentication errors**:
   ```bash
   # Test authentication with fastest model
   claude --print --model claude-haiku-4-5-20251001 "Hello"
   # If this fails, re-authenticate if needed
   ```

3. **Timeout errors**:
   - Increase `MAX_TIMEOUT` in `.env`
   - Note: Claude Code can take time for complex requests

## Testing

### 🧪 **Quick Test Suite**
Test all endpoints with a simple script:
```bash
# Make sure server is running first
poetry run python tests/test_endpoints.py
```

### 📝 **Basic Test Suite**
Run the comprehensive test suite:
```bash
# Make sure server is running first  
poetry run python test_basic.py

# With API key protection enabled, set TEST_API_KEY:
TEST_API_KEY=your-generated-key poetry run python test_basic.py
```

The test suite automatically detects whether API key protection is enabled and provides helpful guidance for providing the necessary authentication.

### 🔍 **Authentication Test**
Check authentication status:
```bash
curl http://localhost:8000/v1/auth/status | python -m json.tool
```

### ⚙️ **Development Tools**
```bash
# Install development dependencies
poetry install --with dev

# Format code
poetry run black .

# Run full tests (when implemented)
poetry run pytest tests/
```

### ✅ **Expected Results**
All tests should show:
- **4/4 endpoint tests passing**
- **4/4 basic tests passing** 
- **Authentication method detected** (claude_cli, anthropic, bedrock, or vertex)
- **Real cost tracking** (e.g., $0.001-0.005 per test call)
- **Accurate token counts** from SDK metadata

## FAQ

### Can I use my Claude Pro or Max subscription with the OpenAI SDK?
Yes. Run `claude auth login` on the host, start claude-relay, and point the OpenAI SDK's `base_url` at `http://<host>:8000/v1`. Check the [Terms Compliance](#terms-compliance) section first: subscription use is for your own personal use.

### Does it support streaming?
On `/v1/chat/completions`, yes: pass `"stream": true` to get server-sent events. `/v1/messages` currently returns non-streaming responses only.

### Does it support OpenAI function calling?
No. OpenAI-style `tools`/function calling is not mapped. Claude Code's own tools (Bash, Read, Write and others) can be turned on per request with `"enable_tools": true`.

### Which models can I use?
Any model Claude Code accepts, such as `claude-sonnet-4-6`, `claude-opus-4-6` and `claude-haiku-4-5-20251001`. With `ANTHROPIC_API_KEY` set, `/v1/models` returns Anthropic's live list. See [Supported Models](#supported-models).

### Is it safe to expose on a network?
Set `API_KEY` so clients must send a bearer token, keep tools off unless you need them, and keep it on a private network. See [API Security Configuration](#-api-security-configuration).

### How is this different from calling the Anthropic API directly?
Requests go through Claude Code, so you can use a Claude subscription login instead of per-token API billing, and optionally give Claude its coding tools. If you already have an API key and don't need Claude Code, the Anthropic API is simpler.

## Terms Compliance

This wrapper is designed to be compliant with [Anthropic's Terms of Service](https://www.anthropic.com/legal).

### Requirements for Users

> **Important:** You must have your own valid Claude subscription or API access to use this wrapper.

- **Claude Pro or Max subscription** - For CLI authentication (`claude auth login`)
- **Anthropic API key** - Available at [platform.claude.com](https://platform.claude.com)
- **AWS Bedrock or Google Vertex AI** - For enterprise cloud authentication

This wrapper does not provide Claude access - it provides an OpenAI-compatible interface to Claude services you already have access to.

### How This Wrapper Works

- **Uses the official Claude Agent SDK** - The same SDK Anthropic provides for developers
- **Each user authenticates individually** - No credential sharing or pooling
- **Format translation only** - Converts OpenAI-format requests to Claude SDK calls
- **No reselling** - Users access Claude through their own subscriptions/API keys

### Personal vs Commercial Use

| Use Case | Recommended Authentication | Notes |
|----------|---------------------------|-------|
| Personal projects | CLI Auth (Pro/Max) or API Key | Acceptable at moderate scale |
| Business/Commercial | API Key, Bedrock, or Vertex AI | Use [platform.claude.com](https://platform.claude.com) |
| High-scale applications | Bedrock or Vertex AI | Enterprise authentication recommended |

**Note on Consumer Plans:** Claude Pro and Max subscriptions are primarily designed for individual, interactive use. Using them through wrappers or automated implementations is acceptable for personal projects at moderate scale. For business use or applications that scale significantly, Anthropic's commercial API offerings at [platform.claude.com](https://platform.claude.com) are more appropriate.

### Authentication Methods

| Method | Terms | Compliance |
|--------|-------|------------|
| `ANTHROPIC_API_KEY` | Commercial Terms | Explicitly allowed for programmatic access |
| AWS Bedrock | Commercial Terms | Explicitly allowed for programmatic access |
| Google Vertex AI | Commercial Terms | Explicitly allowed for programmatic access |
| CLI Auth (Pro/Max) | Consumer Terms | Uses official SDK with official auth methods |

### CLI Authentication Note

Using CLI auth (`claude auth login`) with this wrapper is functionally equivalent to using Claude Code directly - both use the Claude Agent SDK with your personal subscription. Anthropic provides the SDK with CLI auth support, and this wrapper simply provides an alternative interface format.

### What This Wrapper Does NOT Do

- Does not share or pool credentials between users
- Does not include or expose API keys or credentials
- Does not resell API access
- Does not train competing AI models
- Does not scrape or harvest data
- Does not bypass authentication or rate limits

### User Responsibilities

By using this wrapper, you agree to:
- Comply with [Anthropic's Terms of Service](https://www.anthropic.com/legal/consumer-terms)
- Comply with [Anthropic's Usage Policy](https://www.anthropic.com/legal/aup)
- Use your own valid Claude subscription or API access
- Not share your credentials with others
- Use commercial API access for business applications

### Disclaimer

This is an independent open-source project, not affiliated with or endorsed by Anthropic. Users are responsible for ensuring their own usage complies with Anthropic's terms. Anthropic reserves the right to modify their Terms of Service at any time.

When in doubt, use `ANTHROPIC_API_KEY` authentication which is explicitly permitted for programmatic access under the Commercial Terms.

For Anthropic's official terms, see:
- [Usage Policy](https://www.anthropic.com/legal/aup)
- [Consumer Terms](https://www.anthropic.com/legal/consumer-terms)
- [Commercial Terms](https://www.anthropic.com/legal/commercial-terms)

## Licence

MIT Licence

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
