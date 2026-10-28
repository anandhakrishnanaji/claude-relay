"""Command-line entry point: `claude-relay` (or `python -m claude_relay`)."""

import argparse
import ipaddress
import os
import secrets
import socket
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

# Wildcard addresses, used only to pick the URL to print; nothing binds to these here.
_ALL_INTERFACES = ("0.0.0.0", "::")  # nosec B104


def _version() -> str:
    try:
        return version("claude-relay-server")
    except PackageNotFoundError:
        from claude_relay import __version__

        return __version__


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _port_in_use(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-relay",
        description=(
            "Run an OpenAI- and Anthropic-compatible API server backed by Claude Code. "
            "Settings come from flags, then environment variables, then a .env file "
            "in the current directory."
        ),
    )
    parser.add_argument(
        "--host",
        help="interface to bind (env CLAUDE_WRAPPER_HOST, default 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, help="port to listen on (env PORT, default 8000)")
    parser.add_argument(
        "--api-key",
        help="require clients to send 'Authorization: Bearer <key>' (env API_KEY)",
    )
    parser.add_argument(
        "--generate-api-key",
        action="store_true",
        help="generate a random API key for this run and print it",
    )
    parser.add_argument(
        "--env-file",
        help="load settings from this file instead of ./.env",
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging (env DEBUG_MODE)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.env_file:
        if not os.path.isfile(args.env_file):
            sys.exit(f"claude-relay: env file not found: {args.env_file}")
        load_dotenv(args.env_file)
    else:
        load_dotenv(find_dotenv(usecwd=True))

    # Flags override the environment. The app reads its settings at import
    # time, so they must be in os.environ before claude_relay.main is imported.
    if args.generate_api_key:
        os.environ["API_KEY"] = secrets.token_urlsafe(24)
    elif args.api_key:
        os.environ["API_KEY"] = args.api_key
    if args.debug:
        os.environ["DEBUG_MODE"] = "true"

    host = args.host or os.getenv("CLAUDE_WRAPPER_HOST") or "127.0.0.1"
    port = args.port if args.port is not None else int(os.getenv("PORT", "8000"))

    if _port_in_use(host, port):
        sys.exit(f"claude-relay: port {port} on {host} is already in use; pick another with --port")

    display_host = "localhost" if _is_loopback(host) or host in _ALL_INTERFACES else host
    api_key = os.getenv("API_KEY")
    print(f"claude-relay {_version()}")
    print(f"  OpenAI base URL:    http://{display_host}:{port}/v1")
    print(f"  Anthropic base URL: http://{display_host}:{port}")
    if args.generate_api_key:
        print(f"  API key:            {api_key}")
    elif api_key:
        print("  API key:            required (from --api-key or API_KEY)")
    if not _is_loopback(host) and not api_key:
        print(
            f"  WARNING: listening on {host} without an API key. Anyone who can reach this "
            "port can use your Claude account. Set --api-key or bind to 127.0.0.1."
        )
    print(flush=True)

    import uvicorn

    uvicorn.run("claude_relay.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
