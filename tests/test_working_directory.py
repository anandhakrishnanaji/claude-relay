#!/usr/bin/env python3
"""ClaudeCodeCLI's working directory: a temp dir by default, else CLAUDE_CWD or cwd."""

import shutil
import tempfile
from pathlib import Path

import pytest

from claude_relay.claude_cli import ClaudeCodeCLI


def test_default_temp_directory(monkeypatch):
    monkeypatch.delenv("CLAUDE_CWD", raising=False)
    cli = ClaudeCodeCLI()
    try:
        assert cli.temp_dir is not None
        assert cli.temp_dir.startswith(tempfile.gettempdir())
        assert "claude_code_workspace_" in cli.temp_dir
        assert isinstance(cli.cwd, Path) and cli.cwd.exists()
        assert str(cli.cwd) == cli.temp_dir
    finally:
        shutil.rmtree(cli.temp_dir, ignore_errors=True)


def test_explicit_cwd(tmp_path):
    cli = ClaudeCodeCLI(cwd=str(tmp_path))
    assert cli.temp_dir is None
    assert cli.cwd == tmp_path


def test_nonexistent_directory_error():
    with pytest.raises(ValueError, match="does not exist"):
        ClaudeCodeCLI(cwd="/this/directory/does/not/exist/12345")
