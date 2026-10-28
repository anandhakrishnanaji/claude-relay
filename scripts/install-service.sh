#!/usr/bin/env bash
# Install the wrapper as an always-on systemd user service (Linux).
#
#   ./scripts/install-service.sh            # default: http://claude.localhost:8787/v1
#   PORT=9000 ./scripts/install-service.sh  # custom port
#   ./scripts/install-service.sh --uninstall
#
# Uses your existing Claude login (`claude auth login`); no API key needed.
set -euo pipefail

SERVICE=claude-wrapper
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/$SERVICE.service"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8787}"
HOST="${HOST:-127.0.0.1}"

if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl --user disable --now "$SERVICE" 2>/dev/null || true
    rm -f "$UNIT"
    systemctl --user daemon-reload
    echo "Removed $SERVICE service."
    exit 0
fi

command -v uv >/dev/null || { echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v claude >/dev/null || echo "warning: 'claude' CLI not on PATH; the bundled SDK CLI will be used"

if ! claude auth status 2>/dev/null | grep -q '"loggedIn": true'; then
    echo "Not logged in to Claude. Run: claude auth login"
    exit 1
fi

echo "==> Creating virtualenv and installing dependencies"
cd "$REPO"
[[ -d .venv ]] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -q -e .

if [[ ! -f .env ]]; then
    echo "==> Writing .env"
    cat > .env <<EOF
CLAUDE_AUTH_METHOD=cli
CLAUDE_WRAPPER_HOST=$HOST
PORT=$PORT
MAX_TIMEOUT=600000
CORS_ORIGINS=["*"]
RATE_LIMIT_ENABLED=false
EOF
else
    echo "==> Keeping existing .env"
fi

echo "==> Writing $UNIT"
mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=Claude Code OpenAI-compatible wrapper (claude.localhost:$PORT)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO
EnvironmentFile=$REPO/.env
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$REPO/.venv/bin/uvicorn src.main:app --host $HOST --port $PORT
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE"
systemctl --user restart "$SERVICE"
# Keep running after logout and start at boot without logging in
loginctl enable-linger "$USER" || echo "warning: could not enable linger; service will only run while logged in"

echo "==> Waiting for health check"
for _ in $(seq 1 20); do
    if curl -sf "http://claude.localhost:$PORT/health" >/dev/null; then
        echo "Ready: http://claude.localhost:$PORT/v1"
        exit 0
    fi
    sleep 1
done
echo "Service did not become healthy; check: journalctl --user -u $SERVICE -n 50"
exit 1
