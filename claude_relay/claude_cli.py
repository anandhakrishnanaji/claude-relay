import asyncio
import os
import tempfile
import atexit
import shutil
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path
import logging

import claude_agent_sdk
from claude_agent_sdk import query, ClaudeAgentOptions

from claude_relay import constants
from claude_relay.constants import CLAUDE_SETTING_SOURCES

logger = logging.getLogger(__name__)

# Linux caps a single argv string at MAX_ARG_STRLEN (131072 bytes) and the SDK
# passes a string system prompt as one --system-prompt argument. Prompts larger
# than this many UTF-8 bytes are written to a temp file instead.
SYSTEM_PROMPT_FILE_THRESHOLD_BYTES = 100_000


def _find_cli() -> Optional[str]:
    """The CLI the SDK runs: its bundled binary, else ``claude`` on PATH."""
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return str(bundled) if bundled.is_file() else shutil.which("claude")


class _QueryTimeout(Exception):
    """Raised internally when a query exceeds ClaudeCodeCLI.timeout."""


class ClaudeCodeCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None):
        self.timeout = timeout / 1000  # Convert ms to seconds
        self.temp_dir = None

        # If cwd is provided (from CLAUDE_CWD env var), use it
        # Otherwise create an isolated temp directory
        if cwd:
            self.cwd = Path(cwd)
            # Check if the directory exists
            if not self.cwd.exists():
                logger.error(f"ERROR: Specified working directory does not exist: {self.cwd}")
                logger.error(
                    "Please create the directory first or unset CLAUDE_CWD to use a temporary directory"
                )
                raise ValueError(f"Working directory does not exist: {self.cwd}")
            else:
                logger.info(f"Using CLAUDE_CWD: {self.cwd}")
        else:
            # Create isolated temp directory (cross-platform)
            self.temp_dir = tempfile.mkdtemp(prefix="claude_code_workspace_")
            self.cwd = Path(self.temp_dir)
            logger.info(f"Using temporary isolated workspace: {self.cwd}")

            # Register cleanup function to remove temp dir on exit
            atexit.register(self._cleanup_temp_dir)

        # Import auth manager
        from claude_relay.auth import auth_manager, validate_claude_code_auth

        # Validate authentication
        is_valid, auth_info = validate_claude_code_auth()
        if not is_valid:
            logger.warning(f"Claude Code authentication issues detected: {auth_info['errors']}")
        else:
            logger.info(f"Claude Code authentication method: {auth_info.get('method', 'unknown')}")

        # Store auth environment variables for SDK
        self.claude_env_vars = auth_manager.get_claude_code_env_vars()

    async def verify_cli(self, timeout: float = 30.0) -> bool:
        """Check the Claude Code CLI runs and is logged in, without a model call.

        Runs ``claude auth status``, which reads local credentials only and exits
        non-zero when not logged in. It costs no tokens.
        """
        cli = _find_cli()
        if not cli:
            logger.error("Claude Code CLI not found (neither bundled with the SDK nor on PATH)")
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                cli,
                "auth",
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self.claude_env_vars or {})},
                cwd=self.cwd,
            )
        except OSError as e:
            logger.error(f"Could not run Claude Code CLI {cli}: {e}")
            return False
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(f"`claude auth status` timed out after {timeout:g}s")
            return False
        if proc.returncode != 0:
            detail = (stderr or stdout).decode(errors="replace").strip()[:500]
            logger.warning(f"Claude Code CLI is not logged in: {detail}")
            logger.warning("Run `claude auth login` on this host, or set CLAUDE_CODE_OAUTH_TOKEN")
            return False
        return True

    async def run_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        permission_mode: Optional[str] = None,
        thinking: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run Claude Agent using the Python SDK and yield response chunks.

        The whole query iteration is bounded by ``self.timeout`` seconds.
        """
        system_prompt_file = None
        sdk_stream = None

        try:
            # Build SDK options. Auth env vars go to the CLI subprocess via
            # options.env, so os.environ is never touched (it is shared by
            # concurrent requests).
            options = ClaudeAgentOptions(
                max_turns=max_turns,
                cwd=self.cwd,
                setting_sources=list(CLAUDE_SETTING_SOURCES),
                env=dict(self.claude_env_vars or {}),
            )

            # Set model if specified
            if model:
                options.model = model

            # Set system prompt. The SDK takes a plain string (or a typed
            # preset/custom/file dict); any other dict shape is silently
            # dropped and the CLI falls back to its own Claude Code prompt.
            if system_prompt:
                if len(system_prompt.encode("utf-8")) > SYSTEM_PROMPT_FILE_THRESHOLD_BYTES:
                    # A string prompt is passed as a single argv entry, which
                    # Linux caps at MAX_ARG_STRLEN (128 KiB). Hand the CLI a
                    # file instead.
                    system_prompt_file = self._write_system_prompt_file(system_prompt)
                    options.system_prompt = {"type": "file", "path": system_prompt_file}
                else:
                    options.system_prompt = system_prompt
            else:
                # Use Claude Code preset to maintain expected behavior
                options.system_prompt = {"type": "preset", "preset": "claude_code"}

            # Set tool restrictions
            if allowed_tools:
                options.allowed_tools = allowed_tools
            if disallowed_tools:
                options.disallowed_tools = disallowed_tools

            # Set permission mode (needed for tool execution in API context)
            if permission_mode:
                options.permission_mode = permission_mode

            # SDK ThinkingConfig; replaces the deprecated max_thinking_tokens
            if thinking:
                options.thinking = thinking
            if effort:
                options.effort = effort

            # The CLI refuses a fallback that equals the main model
            fallback_model = fallback_model or constants.FALLBACK_MODEL
            if fallback_model and fallback_model != model:
                options.fallback_model = fallback_model

            # Run the query and yield messages, enforcing the overall timeout.
            # asyncio.timeout() needs Python 3.11, so bound each step with
            # wait_for() against a single deadline.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout
            sdk_stream = query(prompt=prompt, options=options)
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise _QueryTimeout()
                try:
                    message = await asyncio.wait_for(sdk_stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    # Only our deadline counts; a TimeoutError raised inside
                    # the SDK well before it is an ordinary SDK error. (The
                    # event loop may fire timers up to clock_resolution early.)
                    if deadline - loop.time() > 0.5:
                        raise
                    raise _QueryTimeout() from None

                # Debug logging
                logger.debug(f"Raw SDK message type: {type(message)}")
                logger.debug(f"Raw SDK message: {message}")

                # Convert message object to dict if needed
                if hasattr(message, "__dict__") and not isinstance(message, dict):
                    # Convert object to dict for consistent handling
                    message_dict = {}

                    # Get all attributes from the object
                    for attr_name in dir(message):
                        if not attr_name.startswith("_"):  # Skip private attributes
                            try:
                                attr_value = getattr(message, attr_name)
                                if not callable(attr_value):  # Skip methods
                                    message_dict[attr_name] = attr_value
                            except:
                                pass

                    logger.debug(f"Converted message dict: {message_dict}")
                    yield message_dict
                else:
                    yield message

        except _QueryTimeout:
            logger.error(f"Claude Agent SDK query timed out after {self.timeout:g}s")
            # Close the SDK generator now so the CLI subprocess is terminated
            # before the error is reported.
            await self._close_sdk_stream(sdk_stream)
            sdk_stream = None
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": f"Claude Agent SDK query timed out after {self.timeout:g} seconds",
            }

        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            # Yield error message in the expected format
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": str(e),
            }

        finally:
            # Also runs when the caller stops iterating early (aclose).
            await self._close_sdk_stream(sdk_stream)
            if system_prompt_file:
                try:
                    os.unlink(system_prompt_file)
                except OSError as e:
                    logger.warning(f"Failed to remove system prompt file {system_prompt_file}: {e}")

    @staticmethod
    def _write_system_prompt_file(system_prompt: str) -> str:
        """Write a system prompt to a private temp file and return its path."""
        fd, path = tempfile.mkstemp(prefix="claude_system_prompt_", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(system_prompt)
        except BaseException:
            os.unlink(path)
            raise
        return path

    @staticmethod
    async def _close_sdk_stream(sdk_stream) -> None:
        """Close the SDK query generator, which terminates the CLI subprocess."""
        if sdk_stream is None:
            return
        try:
            await sdk_stream.aclose()
        except Exception as e:
            logger.warning(f"Error closing Claude Agent SDK stream: {e}")

    def parse_claude_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the assistant message from Claude Agent SDK messages.

        Prioritizes ResultMessage.result for multi-turn conversations,
        falls back to last AssistantMessage content.
        """
        # First, check for ResultMessage with 'result' field (multi-turn completion)
        for message in messages:
            if message.get("subtype") == "success" and "result" in message:
                return message["result"]

        # Collect all text from AssistantMessages (take the last one with text)
        last_text = None
        for message in messages:
            # Look for AssistantMessage type (new SDK format)
            if "content" in message and isinstance(message["content"], list):
                text_parts = []
                for block in message["content"]:
                    # Handle TextBlock objects
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)

                if text_parts:
                    last_text = "\n".join(text_parts)

            # Fallback: look for old format
            elif message.get("type") == "assistant" and "message" in message:
                sdk_message = message["message"]
                if isinstance(sdk_message, dict) and "content" in sdk_message:
                    content = sdk_message["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Handle content blocks (Anthropic SDK format)
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            last_text = "\n".join(text_parts)
                    elif isinstance(content, str):
                        last_text = content

        return last_text

    def extract_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata like costs, tokens, and session info from SDK messages."""
        metadata = {
            "session_id": None,
            "total_cost_usd": 0.0,
            "duration_ms": 0,
            "num_turns": 0,
            "model": None,
        }

        for message in messages:
            # New SDK format - ResultMessage
            if message.get("subtype") == "success" and "total_cost_usd" in message:
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            # New SDK format - SystemMessage
            elif message.get("subtype") == "init" and "data" in message:
                data = message["data"]
                metadata.update({"session_id": data.get("session_id"), "model": data.get("model")})
            # Old format fallback
            elif message.get("type") == "result":
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            elif message.get("type") == "system" and message.get("subtype") == "init":
                metadata.update(
                    {"session_id": message.get("session_id"), "model": message.get("model")}
                )

        return metadata

    def estimate_token_usage(
        self, prompt: str, completion: str, model: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Estimate token usage based on character count.

        Uses rough approximation: ~4 characters per token for English text.
        This is approximate and may not match actual tokenization.
        """
        # Rough approximation: 1 token ≈ 4 characters
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _cleanup_temp_dir(self):
        """Clean up temporary directory on exit."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary workspace: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {self.temp_dir}: {e}")
