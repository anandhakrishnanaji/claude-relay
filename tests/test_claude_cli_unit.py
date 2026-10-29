#!/usr/bin/env python3
"""
Unit tests for claude_relay/claude_cli.py

Tests the ClaudeCodeCLI class methods.
These are pure unit tests that don't require a running server or Claude SDK.
"""

import pytest
import os
import tempfile
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path


class TestClaudeCodeCLIParseMessage:
    """Test ClaudeCodeCLI.parse_claude_message()"""

    @pytest.fixture
    def cli_class(self):
        """Get the ClaudeCodeCLI class without instantiating."""
        from claude_relay.claude_cli import ClaudeCodeCLI

        return ClaudeCodeCLI

    def test_parse_result_message(self, cli_class):
        """Parses result message with 'result' field."""
        # Use classmethod-like approach - create minimal mock instance
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [{"subtype": "success", "result": "The final answer is 42."}]
        result = cli.parse_claude_message(messages)
        assert result == "The final answer is 42."

    def test_parse_assistant_message_with_content_list(self, cli_class):
        """Parses assistant message with content list."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "World!"},
                ]
            }
        ]
        result = cli.parse_claude_message(messages)
        assert result == "Hello \nWorld!"

    def test_parse_assistant_message_with_textblock_objects(self, cli_class):
        """Parses assistant message with TextBlock objects."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        # Mock TextBlock object
        text_block = MagicMock()
        text_block.text = "Response text"

        messages = [{"content": [text_block]}]
        result = cli.parse_claude_message(messages)
        assert result == "Response text"

    def test_parse_assistant_message_with_string_content(self, cli_class):
        """Parses assistant message with string content blocks."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [{"content": ["Part 1", "Part 2"]}]
        result = cli.parse_claude_message(messages)
        assert result == "Part 1\nPart 2"

    def test_parse_old_format_assistant_message(self, cli_class):
        """Parses old format assistant message."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Old format response"}]},
            }
        ]
        result = cli.parse_claude_message(messages)
        assert result == "Old format response"

    def test_parse_old_format_string_content(self, cli_class):
        """Parses old format with string content."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {
                "type": "assistant",
                "message": {"content": "Simple string content"},
            }
        ]
        result = cli.parse_claude_message(messages)
        assert result == "Simple string content"

    def test_parse_empty_messages_returns_none(self, cli_class):
        """Empty messages list returns None."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        result = cli.parse_claude_message([])
        assert result is None

    def test_parse_no_matching_messages_returns_none(self, cli_class):
        """No matching messages returns None."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [{"type": "system", "content": "System message"}]
        result = cli.parse_claude_message(messages)
        assert result is None

    def test_parse_uses_last_text(self, cli_class):
        """When multiple messages, uses the last one with text."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {"content": [{"type": "text", "text": "First response"}]},
            {"content": [{"type": "text", "text": "Second response"}]},
        ]
        result = cli.parse_claude_message(messages)
        assert result == "Second response"

    def test_result_takes_priority(self, cli_class):
        """ResultMessage.result takes priority over AssistantMessage."""
        cli = MagicMock()
        cli.parse_claude_message = cli_class.parse_claude_message.__get__(cli, cli_class)

        messages = [
            {"content": [{"type": "text", "text": "Some response"}]},
            {"subtype": "success", "result": "Final result"},
        ]
        result = cli.parse_claude_message(messages)
        assert result == "Final result"


class TestClaudeCodeCLICleanupTempDir:
    """Test ClaudeCodeCLI._cleanup_temp_dir()"""

    def test_cleanup_removes_existing_dir(self):
        """Cleanup removes existing temp directory."""
        from claude_relay.claude_cli import ClaudeCodeCLI

        # Create a mock instance
        cli = MagicMock(spec=ClaudeCodeCLI)

        # Create an actual temp directory
        temp_dir = tempfile.mkdtemp(prefix="test_cleanup_")
        cli.temp_dir = temp_dir

        # Bind the method
        cli._cleanup_temp_dir = ClaudeCodeCLI._cleanup_temp_dir.__get__(cli, ClaudeCodeCLI)

        assert os.path.exists(temp_dir)

        cli._cleanup_temp_dir()

        assert not os.path.exists(temp_dir)

    def test_cleanup_handles_missing_dir(self):
        """Cleanup handles already-deleted directory gracefully."""
        from claude_relay.claude_cli import ClaudeCodeCLI

        cli = MagicMock(spec=ClaudeCodeCLI)
        cli.temp_dir = "/nonexistent/test/dir/12345"

        cli._cleanup_temp_dir = ClaudeCodeCLI._cleanup_temp_dir.__get__(cli, ClaudeCodeCLI)

        # Should not raise
        cli._cleanup_temp_dir()

    def test_cleanup_no_temp_dir_set(self):
        """Cleanup does nothing when temp_dir is None."""
        from claude_relay.claude_cli import ClaudeCodeCLI

        cli = MagicMock(spec=ClaudeCodeCLI)
        cli.temp_dir = None

        cli._cleanup_temp_dir = ClaudeCodeCLI._cleanup_temp_dir.__get__(cli, ClaudeCodeCLI)

        # Should not raise
        cli._cleanup_temp_dir()


class TestClaudeCodeCLIInit:
    """Test ClaudeCodeCLI.__init__() initialization logic."""

    def test_timeout_conversion(self):
        """Timeout is converted from milliseconds to seconds."""
        # Test the conversion logic directly
        timeout_ms = 120000
        timeout_seconds = timeout_ms / 1000
        assert timeout_seconds == 120.0

    def test_path_handling_with_valid_dir(self):
        """Valid directory path is handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            assert path.exists()

    def test_path_handling_with_invalid_dir(self):
        """Invalid directory path is detected."""
        path = Path("/nonexistent/path/12345")
        assert not path.exists()

    def test_init_with_cwd(self):
        """ClaudeCodeCLI initializes with provided cwd."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("claude_relay.auth.validate_claude_code_auth") as mock_validate:
                with patch("claude_relay.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {}

                    from claude_relay.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI(cwd=temp_dir)

                    assert cli.cwd == Path(temp_dir)
                    assert cli.temp_dir is None
                    assert cli.timeout == 600.0  # 600000ms / 1000

    def test_init_with_invalid_cwd_raises(self):
        """ClaudeCodeCLI raises ValueError for non-existent cwd."""
        with patch("claude_relay.auth.validate_claude_code_auth") as mock_validate:
            with patch("claude_relay.auth.auth_manager") as mock_auth:
                mock_validate.return_value = (True, {"method": "anthropic"})
                mock_auth.get_claude_code_env_vars.return_value = {}

                from claude_relay.claude_cli import ClaudeCodeCLI

                with pytest.raises(ValueError, match="Working directory does not exist"):
                    ClaudeCodeCLI(cwd="/nonexistent/path/12345")

    def test_init_without_cwd_creates_temp(self):
        """ClaudeCodeCLI creates temp directory when no cwd provided."""
        with patch("claude_relay.auth.validate_claude_code_auth") as mock_validate:
            with patch("claude_relay.auth.auth_manager") as mock_auth:
                with patch("atexit.register"):  # Don't actually register cleanup
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {}

                    from claude_relay.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI()

                    assert cli.temp_dir is not None
                    assert cli.cwd == Path(cli.temp_dir)
                    assert "claude_code_workspace_" in cli.temp_dir

                    # Cleanup
                    if cli.temp_dir and os.path.exists(cli.temp_dir):
                        import shutil

                        shutil.rmtree(cli.temp_dir)

    def test_init_with_custom_timeout(self):
        """ClaudeCodeCLI uses custom timeout."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("claude_relay.auth.validate_claude_code_auth") as mock_validate:
                with patch("claude_relay.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {}

                    from claude_relay.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI(timeout=120000, cwd=temp_dir)

                    assert cli.timeout == 120.0

    def test_init_auth_validation_failure(self):
        """ClaudeCodeCLI handles auth validation failure gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("claude_relay.auth.validate_claude_code_auth") as mock_validate:
                with patch("claude_relay.auth.auth_manager") as mock_auth:
                    # Auth fails
                    mock_validate.return_value = (False, {"errors": ["Missing API key"]})
                    mock_auth.get_claude_code_env_vars.return_value = {}

                    from claude_relay.claude_cli import ClaudeCodeCLI

                    # Should not raise, just log warning
                    cli = ClaudeCodeCLI(cwd=temp_dir)
                    assert cli.cwd == Path(temp_dir)


class TestClaudeCodeCLIVerifyCLI:
    """Test ClaudeCodeCLI.verify_cli()"""

    @pytest.fixture
    def cli_instance(self):
        """Create a CLI instance with mocked auth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("claude_relay.auth.validate_claude_code_auth") as mock_validate:
                with patch("claude_relay.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {}

                    from claude_relay.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI(cwd=temp_dir)
                    yield cli

    def _fake_cli(self, tmp_path, script):
        cli = tmp_path / "claude"
        cli.write_text("#!/bin/sh\n" + script)
        cli.chmod(0o755)
        return patch("claude_relay.claude_cli._find_cli", return_value=str(cli))

    @pytest.mark.asyncio
    async def test_runs_auth_status_without_a_query(self, cli_instance, tmp_path):
        """verify_cli runs `claude auth status` with the auth env and never calls query()."""
        cli_instance.claude_env_vars = {"ANTHROPIC_API_KEY": "test-key"}
        out = tmp_path / "args"
        script = f'echo "$@ $ANTHROPIC_API_KEY" > {out}\necho \'{{"loggedIn": true}}\'\n'
        with (
            self._fake_cli(tmp_path, script),
            patch("claude_relay.claude_cli.query", side_effect=AssertionError("billable")),
        ):
            assert await cli_instance.verify_cli() is True
        assert out.read_text().strip() == "auth status test-key"

    @pytest.mark.asyncio
    async def test_not_logged_in(self, cli_instance, tmp_path):
        with self._fake_cli(tmp_path, "echo '{\"loggedIn\": false}'\nexit 1\n"):
            assert await cli_instance.verify_cli() is False

    @pytest.mark.asyncio
    async def test_timeout_kills_the_process(self, cli_instance, tmp_path):
        with self._fake_cli(tmp_path, "exec sleep 30\n"):
            assert await cli_instance.verify_cli(timeout=0.2) is False

    @pytest.mark.asyncio
    async def test_cli_missing(self, cli_instance):
        with patch("claude_relay.claude_cli._find_cli", return_value=None):
            assert await cli_instance.verify_cli() is False


class TestClaudeCodeCLIRunCompletion:
    """Test ClaudeCodeCLI.run_completion()"""

    @pytest.fixture
    def cli_instance(self):
        """Create a CLI instance with mocked auth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("claude_relay.auth.validate_claude_code_auth") as mock_validate:
                with patch("claude_relay.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {
                        "ANTHROPIC_API_KEY": "test-key"
                    }

                    from claude_relay.claude_cli import ClaudeCodeCLI

                    cli = ClaudeCodeCLI(cwd=temp_dir)
                    yield cli

    @pytest.mark.asyncio
    async def test_run_completion_basic(self, cli_instance):
        """run_completion yields messages from SDK."""
        mock_message = {"type": "assistant", "content": [{"type": "text", "text": "Hello"}]}

        async def mock_query(*args, **kwargs):
            yield mock_message

        with patch("claude_relay.claude_cli.query", mock_query):
            messages = []
            async for msg in cli_instance.run_completion("Hello"):
                messages.append(msg)

            assert len(messages) == 1
            assert messages[0] == mock_message

    @pytest.mark.asyncio
    async def test_run_completion_with_system_prompt(self, cli_instance):
        """run_completion sets system_prompt option."""
        mock_message = {"type": "assistant", "content": "Response"}
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield mock_message

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello", system_prompt="You are helpful"):
                pass

            assert len(captured_options) == 1
            opts = captured_options[0]
            # Must be a plain string: the SDK drops unknown dict shapes silently
            assert opts.system_prompt == "You are helpful"

    @pytest.mark.asyncio
    async def test_run_completion_isolates_filesystem_settings(self, cli_instance):
        """run_completion loads no host settings (CLAUDE.md, skills) by default."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "assistant", "content": "Response"}

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello"):
                pass

        assert captured_options[0].setting_sources == []

    @pytest.mark.asyncio
    async def test_run_completion_with_model(self, cli_instance):
        """run_completion sets model option."""
        mock_message = {"type": "assistant"}
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield mock_message

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello", model="claude-3-opus"):
                pass

            assert captured_options[0].model == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_run_completion_with_tool_restrictions(self, cli_instance):
        """run_completion sets allowed/disallowed tools."""
        mock_message = {"type": "assistant"}
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield mock_message

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion(
                "Hello",
                allowed_tools=["Bash", "Read"],
                disallowed_tools=["Task"],
            ):
                pass

            assert captured_options[0].allowed_tools == ["Bash", "Read"]
            assert captured_options[0].disallowed_tools == ["Task"]

    @pytest.mark.asyncio
    async def test_run_completion_with_permission_mode(self, cli_instance):
        """run_completion sets permission_mode."""
        mock_message = {"type": "assistant"}
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield mock_message

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello", permission_mode="acceptEdits"):
                pass

            assert captured_options[0].permission_mode == "acceptEdits"

    @pytest.mark.asyncio
    async def test_run_completion_with_thinking(self, cli_instance):
        """run_completion passes the thinking config to the SDK."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "assistant"}

        thinking = {"type": "enabled", "budget_tokens": 2048}
        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello", thinking=thinking):
                pass

        assert captured_options[0].thinking == thinking
        assert captured_options[0].max_thinking_tokens is None

    @pytest.mark.asyncio
    async def test_run_completion_effort_and_fallback(self, cli_instance):
        """effort and fallback_model reach the SDK; FALLBACK_MODEL is the default."""
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "assistant"}

        with (
            patch("claude_relay.claude_cli.query", mock_query),
            patch("claude_relay.constants.FALLBACK_MODEL", "haiku"),
        ):
            async for _ in cli_instance.run_completion("Hi", model="opus", effort="low"):
                pass
            # The CLI rejects a fallback equal to the main model
            async for _ in cli_instance.run_completion("Hi", model="haiku"):
                pass
            async for _ in cli_instance.run_completion("Hi", model="opus", fallback_model="sonnet"):
                pass

        assert captured_options[0].effort == "low"
        assert captured_options[0].fallback_model == "haiku"
        assert captured_options[1].fallback_model is None
        assert captured_options[2].fallback_model == "sonnet"

    @pytest.mark.asyncio
    async def test_run_completion_converts_objects_to_dicts(self, cli_instance):
        """run_completion converts message objects to dicts."""
        # Create a mock object with attributes
        mock_obj = MagicMock()
        mock_obj.type = "assistant"
        mock_obj.content = "Hello"

        async def mock_query(*args, **kwargs):
            yield mock_obj

        with patch("claude_relay.claude_cli.query", mock_query):
            messages = []
            async for msg in cli_instance.run_completion("Hello"):
                messages.append(msg)

            assert len(messages) == 1
            # Should be converted to dict
            assert isinstance(messages[0], dict)
            assert "type" in messages[0]

    @pytest.mark.asyncio
    async def test_run_completion_exception_yields_error(self, cli_instance):
        """run_completion yields error message on exception."""

        async def mock_query(*args, **kwargs):
            raise RuntimeError("SDK failed")
            yield  # Make it a generator

        with patch("claude_relay.claude_cli.query", mock_query):
            messages = []
            async for msg in cli_instance.run_completion("Hello"):
                messages.append(msg)

            assert len(messages) == 1
            assert messages[0]["type"] == "result"
            assert messages[0]["subtype"] == "error_during_execution"
            assert messages[0]["is_error"] is True
            assert "SDK failed" in messages[0]["error_message"]

    @pytest.mark.asyncio
    async def test_run_completion_passes_env_via_options(self, cli_instance):
        """run_completion passes auth env vars via options.env, not os.environ."""
        captured_options = []
        env_during_query = []
        env_before = dict(os.environ)

        async def mock_query(prompt, options):
            captured_options.append(options)
            env_during_query.append(dict(os.environ))
            yield {"type": "assistant"}

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello"):
                pass

        assert captured_options[0].env == {"ANTHROPIC_API_KEY": "test-key"}
        # os.environ is untouched, both during and after the query
        assert env_during_query[0] == env_before
        assert dict(os.environ) == env_before

    @pytest.mark.asyncio
    async def test_run_completion_timeout_yields_error_and_closes(self, cli_instance):
        """run_completion yields a timeout error and closes the SDK generator."""
        import asyncio

        cli_instance.timeout = 0.05
        state = {"closed": False}

        async def mock_query(prompt, options):
            try:
                yield {"type": "assistant"}
                await asyncio.sleep(10)
                yield {"type": "never"}
            finally:
                state["closed"] = True

        with patch("claude_relay.claude_cli.query", mock_query):
            messages = []
            async for msg in cli_instance.run_completion("Hello"):
                messages.append(msg)

        assert messages[0] == {"type": "assistant"}
        assert len(messages) == 2
        assert messages[1]["type"] == "result"
        assert messages[1]["subtype"] == "error_during_execution"
        assert messages[1]["is_error"] is True
        assert "timed out" in messages[1]["error_message"]
        assert state["closed"] is True

    @pytest.mark.asyncio
    async def test_run_completion_timeout_covers_whole_iteration(self, cli_instance):
        """The timeout is a deadline for the whole query, not per message."""
        import asyncio

        cli_instance.timeout = 0.2

        async def mock_query(prompt, options):
            # Each message arrives well within the timeout, but the total exceeds it
            for _ in range(20):
                await asyncio.sleep(0.05)
                yield {"type": "assistant"}

        with patch("claude_relay.claude_cli.query", mock_query):
            messages = []
            async for msg in cli_instance.run_completion("Hello"):
                messages.append(msg)

        assert len(messages) < 20
        assert messages[-1]["is_error"] is True
        assert "timed out" in messages[-1]["error_message"]

    @pytest.mark.asyncio
    async def test_run_completion_closes_sdk_stream_on_early_exit(self, cli_instance):
        """Closing run_completion early also closes the SDK generator."""
        state = {"closed": False}

        async def mock_query(prompt, options):
            try:
                yield {"type": "assistant"}
                yield {"type": "assistant"}
            finally:
                state["closed"] = True

        with patch("claude_relay.claude_cli.query", mock_query):
            gen = cli_instance.run_completion("Hello")
            await gen.__anext__()
            await gen.aclose()

        assert state["closed"] is True

    @pytest.mark.asyncio
    async def test_run_completion_large_system_prompt_uses_file(self, cli_instance):
        """A system prompt above the argv-safe threshold is passed as a file."""
        from claude_relay.claude_cli import SYSTEM_PROMPT_FILE_THRESHOLD_BYTES

        big_prompt = "é" * SYSTEM_PROMPT_FILE_THRESHOLD_BYTES  # 2 UTF-8 bytes each
        captured = []

        async def mock_query(prompt, options):
            sp = options.system_prompt
            with open(sp["path"], encoding="utf-8") as f:
                captured.append((sp, f.read()))
            yield {"type": "assistant"}

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello", system_prompt=big_prompt):
                pass

        sp, contents = captured[0]
        assert sp["type"] == "file"
        assert set(sp) == {"type", "path"}
        assert contents == big_prompt
        # Temp file is removed afterwards
        assert not os.path.exists(sp["path"])

    @pytest.mark.asyncio
    async def test_run_completion_large_system_prompt_file_removed_on_error(self, cli_instance):
        """The system prompt temp file is removed even when the SDK fails."""
        from claude_relay.claude_cli import SYSTEM_PROMPT_FILE_THRESHOLD_BYTES

        paths = []

        async def mock_query(prompt, options):
            paths.append(options.system_prompt["path"])
            raise RuntimeError("SDK failed")
            yield  # Make it a generator

        with patch("claude_relay.claude_cli.query", mock_query):
            messages = []
            async for msg in cli_instance.run_completion(
                "Hello", system_prompt="x" * (SYSTEM_PROMPT_FILE_THRESHOLD_BYTES + 1)
            ):
                messages.append(msg)

        assert messages[0]["is_error"] is True
        assert not os.path.exists(paths[0])

    @pytest.mark.asyncio
    async def test_run_completion_small_system_prompt_stays_string(self, cli_instance):
        """A system prompt at or below the threshold is passed as a plain string."""
        from claude_relay.claude_cli import SYSTEM_PROMPT_FILE_THRESHOLD_BYTES

        prompt = "x" * SYSTEM_PROMPT_FILE_THRESHOLD_BYTES
        captured_options = []

        async def mock_query(prompt, options):
            captured_options.append(options)
            yield {"type": "assistant"}

        with patch("claude_relay.claude_cli.query", mock_query):
            async for _ in cli_instance.run_completion("Hello", system_prompt=prompt):
                pass

        assert captured_options[0].system_prompt == prompt


class TestClaudeCodeCLICleanupException:
    """Test ClaudeCodeCLI._cleanup_temp_dir() exception handling."""

    def test_cleanup_exception_is_caught(self):
        """Cleanup catches exceptions during rmtree."""
        from claude_relay.claude_cli import ClaudeCodeCLI

        cli = MagicMock(spec=ClaudeCodeCLI)
        temp_dir = tempfile.mkdtemp(prefix="test_cleanup_exc_")
        cli.temp_dir = temp_dir

        # Bind the real method
        cli._cleanup_temp_dir = ClaudeCodeCLI._cleanup_temp_dir.__get__(cli, ClaudeCodeCLI)

        with patch("shutil.rmtree", side_effect=PermissionError("Cannot delete")):
            # Should not raise
            cli._cleanup_temp_dir()

        # Clean up manually
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
