# Production image: non-root, no auto-reload, uses a mounted Claude CLI login.
FROM python:3.12-slim

ARG UID=1000
ARG GID=1000

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

USER app
ENV HOME=/home/app \
    PORT=8000 \
    CLAUDE_AUTH_METHOD=cli \
    CLAUDE_WRAPPER_HOST=0.0.0.0

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
