"""
Unit tests for claude_relay/cli.py
"""

import os
import socket
from unittest.mock import patch

import pytest

from claude_relay import cli


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """Run from an empty directory with no relay settings in the environment."""
    for key in ("API_KEY", "PORT", "CLAUDE_WRAPPER_HOST", "DEBUG_MODE", "FORWARDED_ALLOW_IPS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestIsLoopback:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
    def test_loopback(self, host):
        assert cli._is_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
    def test_not_loopback(self, host):
        assert not cli._is_loopback(host)


class TestPortInUse:
    def test_free_port(self):
        assert not cli._port_in_use("127.0.0.1", _free_port())

    def test_bound_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            assert cli._port_in_use("127.0.0.1", sock.getsockname()[1])


class TestMain:
    def test_defaults_to_localhost(self, clean_env):
        port = _free_port()
        with patch.dict(os.environ, {"PORT": str(port)}), patch("uvicorn.run") as run:
            cli.main([])
        run.assert_called_once_with(
            "claude_relay.main:app", host="127.0.0.1", port=port, forwarded_allow_ips=""
        )

    def test_flags_override_env(self, clean_env):
        port = _free_port()
        with patch.dict(os.environ, {"PORT": "1", "CLAUDE_WRAPPER_HOST": "0.0.0.0"}):
            with patch("uvicorn.run") as run:
                cli.main(["--host", "127.0.0.1", "--port", str(port)])
        run.assert_called_once_with(
            "claude_relay.main:app", host="127.0.0.1", port=port, forwarded_allow_ips=""
        )

    def test_forwarded_allow_ips_from_env(self, clean_env):
        port = _free_port()
        with patch.dict(os.environ, {"PORT": str(port), "FORWARDED_ALLOW_IPS": "10.0.0.1"}):
            with patch("uvicorn.run") as run:
                cli.main([])
        assert run.call_args.kwargs["forwarded_allow_ips"] == "10.0.0.1"

    def test_api_key_flag_sets_env(self, clean_env):
        with patch.dict(os.environ, {}), patch("uvicorn.run"):
            cli.main(["--port", str(_free_port()), "--api-key", "secret"])
            assert os.environ["API_KEY"] == "secret"

    def test_generate_api_key_prints_key(self, clean_env, capsys):
        with patch.dict(os.environ, {}), patch("uvicorn.run"):
            cli.main(["--port", str(_free_port()), "--generate-api-key"])
            key = os.environ["API_KEY"]
        assert len(key) >= 32
        assert key in capsys.readouterr().out

    def test_reads_dotenv_from_cwd(self, clean_env):
        port = _free_port()
        (clean_env / ".env").write_text(f"PORT={port}\n")
        with patch.dict(os.environ, {}), patch("uvicorn.run") as run:
            cli.main([])
        assert run.call_args.kwargs["port"] == port

    def test_missing_env_file_exits(self, clean_env):
        with pytest.raises(SystemExit):
            cli.main(["--env-file", "does-not-exist.env"])

    def test_port_in_use_exits(self, clean_env):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            with patch("uvicorn.run") as run, pytest.raises(SystemExit):
                cli.main(["--port", str(sock.getsockname()[1])])
        run.assert_not_called()

    def test_warns_when_exposed_without_key(self, clean_env, capsys):
        with patch.dict(os.environ, {}), patch("uvicorn.run"):
            cli.main(["--host", "0.0.0.0", "--port", str(_free_port())])
        assert "WARNING" in capsys.readouterr().out
