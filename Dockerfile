# Production image: non-root, no auto-reload, uses a mounted Claude CLI login.
# Runs as a non-root user because the Claude CLI refuses
# --dangerously-skip-permissions as root, which tool use needs.
FROM python:3.12-slim

ARG UID=1000
ARG GID=1000

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app

# Note: Claude Code CLI is bundled with claude-agent-sdk >= 0.1.8
# No separate Node.js/npm installation required
WORKDIR /app

# Install the dependency versions pinned in poetry.lock (with their hashes),
# then the package itself without re-resolving.
COPY pyproject.toml poetry.lock ./
RUN uv tool run --no-cache --from poetry==2.1.3 --with poetry-plugin-export \
        poetry export --only main --format requirements.txt --output /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY README.md ./
COPY claude_relay ./claude_relay
RUN uv pip install --system --no-cache --no-deps .

USER app
ENV HOME=/home/app \
    PORT=8000 \
    CLAUDE_AUTH_METHOD=cli \
    CLAUDE_WRAPPER_HOST=0.0.0.0

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["claude-relay"]
