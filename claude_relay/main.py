"""Entry point kept for `uvicorn claude_relay.main:app`; the app lives in claude_relay.app."""

from claude_relay.app import app

__all__ = ["app"]

if __name__ == "__main__":
    from claude_relay.cli import main

    main()
