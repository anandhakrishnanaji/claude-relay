# Terminal demo for the README: change one base URL and the OpenAI SDK talks to Claude Code.
# Render from the repo root (needs vhs, ttyd, ffmpeg, uv, and a logged-in Claude Code):
#   vhs docs/demo.tape
# Port 8000 must be free.

Output docs/demo.gif

Set Shell bash
Set FontSize 18
Set Width 1100
Set Height 680
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 15ms
Set Framerate 20

# Setup, hidden: plain prompt, empty dir (no stray .env), `uvx claude-relay-server` runs this
# checkout (same as the PyPI release once published) with uvicorn access-log lines filtered,
# and `python` gets the openai SDK.
Hide
Type "export PS1='$ ' PS2='' REPO=$(git rev-parse --show-toplevel); unset API_KEY; cd $(mktemp -d)" Enter
Type `uvx() { command uvx -q --from "$REPO" "$@" </dev/null | grep --line-buffered -v "^INFO"; }; python() { uv run -q --no-project --with openai python "$@"; }` Enter
Type "uvx claude-relay-server --version >/dev/null; python -c 'import openai' && clear" Enter
Sleep 15s
Show

# 1. Start the server
Type "uvx claude-relay-server 2>/dev/null &" Sleep 300ms Enter
Wait+Screen@20s /Anthropic base URL/
Sleep 2s
Hide
Type "clear" Enter
Show

# 2. Streaming curl
Type `curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \` Enter
Type `  -d '{"model": "claude-sonnet-4-6", "stream": true, "messages": [{"role": "user", "content": "A haiku about terminals"}]}'` Sleep 300ms Enter
Sleep 6s

Hide
Type "clear" Enter
Show

# 3. The OpenAI SDK, pointed at the relay
Type "python - <<'EOF'" Enter
Type "from openai import OpenAI" Enter
Type `client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")` Enter
Type `r = client.chat.completions.create(model="claude-sonnet-4-6",` Enter
Type `    messages=[{"role": "user", "content": "Greet the OpenAI SDK in one sentence."}])` Enter
Type "print(r.choices[0].message.content)" Enter
Type "EOF" Sleep 300ms Enter
Sleep 7s

Hide
Type "kill %1" Enter
Sleep 1s
