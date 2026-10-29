"""One Claude Code request, shared by every endpoint.

The endpoints (chat completions, Anthropic messages, Responses) only translate
their request into prepare() and the result back into their own format. This
module owns the rest: model resolution, the tools gate, the SDK options, the
Claude auth check, running the query, and turning SDK errors into one
CompletionError type.

Tests patch the two names the module looks up at call time:
``completion.validate_claude_code_auth`` and ``completion.claude_cli``.
"""

import logging
import os
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Union

from claude_relay import constants
from claude_relay import usage as sdk_usage
from claude_relay.auth import record_auth_failure, validate_claude_code_auth
from claude_relay.claude_cli import ClaudeCodeCLI
from claude_relay.constants import CLAUDE_TOOLS, DEFAULT_ALLOWED_TOOLS
from claude_relay.message_adapter import MessageAdapter
from claude_relay.models import Message
from claude_relay.parameter_validator import ParameterValidator
from claude_relay.session_manager import session_manager

logger = logging.getLogger(__name__)

claude_cli = ClaudeCodeCLI(
    timeout=int(os.getenv("MAX_TIMEOUT", "600000")), cwd=os.getenv("CLAUDE_CWD")
)

TOOLS_DISABLED_MESSAGE = (
    "enable_tools is disabled on this server. Set ALLOW_TOOLS=true on the server to allow "
    "requests to run Claude Code tools (Bash, Write, Edit) with permissions bypassed."
)


class CompletionError(Exception):
    """A request that failed, with its HTTP status and what kind of failure it was.

    kind is one of: claude_cli_not_authenticated (401), model_not_found (400),
    tools_disabled (403), auth_not_configured (503), rate_limit (429),
    upstream_error (502), internal_error (500).
    """

    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.message = message


def claude_login_error(failure: str) -> CompletionError:
    """The Claude Code CLI on this host is not logged in, or its login expired."""
    return CompletionError(
        401,
        "claude_cli_not_authenticated",
        f"Claude Code on the server is not authenticated ({failure}). The server operator "
        "must run `claude auth login` on the host (or set CLAUDE_CODE_OAUTH_TOKEN), then retry.",
    )


def unknown_model_message(model: str) -> str:
    return (
        f"The model '{model}' is not a Claude model. Use a Claude model ID or alias "
        "(see /v1/models), or map it with MODEL_ALIASES on the server."
    )


def classify_sdk_error(chunks: List[Dict[str, Any]]) -> Optional[CompletionError]:
    """The run's error as a CompletionError, or None; also tracks the last login failure."""
    error = sdk_usage.result_error(chunks)
    if not error:
        record_auth_failure(None)
        return None
    if error.get("auth_failure"):
        logger.warning(f"Claude Code login failure: {error['message']}")
        record_auth_failure(error["auth_failure"])
        return claude_login_error(error["auth_failure"])
    if error.get("code") == "model_not_found":
        return CompletionError(400, "model_not_found", error["message"])
    kind = "rate_limit" if error["status"] == 429 else "upstream_error"
    return CompletionError(error["status"], kind, error["message"])


@dataclass
class Completion:
    """A finished run: its text plus what the SDK reported about it."""

    text: Optional[str]
    model: str  # the model that actually ran
    prompt: str
    chunks: List[Dict[str, Any]] = field(repr=False)

    @property
    def stop_reason(self) -> Optional[str]:
        """The SDK's native stop reason (end_turn, max_tokens, ...)."""
        return sdk_usage.extract_stop_reason(self.chunks)

    @property
    def usage(self) -> Optional[Dict[str, int]]:
        """Real token counts from the SDK, or None if it reported none."""
        return sdk_usage.extract_usage(self.chunks)

    def openai_usage(self) -> Dict[str, int]:
        """prompt/completion/total tokens; estimated when the SDK reported none."""
        return sdk_usage.openai_usage_counts(
            self.chunks,
            MessageAdapter.estimate_tokens(self.prompt),
            MessageAdapter.estimate_tokens(self.text),
        )

    def anthropic_usage(self) -> Dict[str, int]:
        return self.usage or {
            "input_tokens": MessageAdapter.estimate_tokens(self.prompt),
            "output_tokens": MessageAdapter.estimate_tokens(self.text),
        }


class Run:
    """A validated request, ready to run once with complete() or stream()."""

    def __init__(self, args: Dict[str, Any], model: str, session_id: Optional[str]):
        self.args = args  # run_completion keyword arguments
        self.model = model  # the requested name until the SDK reports the real one
        self.session_id = session_id

    async def stream(self) -> AsyncIterator[Union[str, Completion]]:
        """Yield text deltas, then the Completion; raise CompletionError on failure."""
        chunks: List[Dict[str, Any]] = []
        try:
            async with aclosing(claude_cli.run_completion(**self.args, stream=True)) as sdk:
                async for chunk in sdk:
                    chunks.append(chunk)
                    self.model = sdk_usage.extract_model([chunk], self.model)
                    # Skip the CLI's own error text (reported below) and the ResultMessage,
                    # which repeats the whole answer
                    if chunk.get("error") or sdk_usage.find_result_message([chunk]):
                        continue
                    text = claude_cli.parse_claude_message([chunk])
                    if text:
                        yield text
        except Exception as e:
            logger.error(f"Claude Code run failed: {e}")
            raise CompletionError(500, "internal_error", str(e)) from e
        error = classify_sdk_error(chunks)
        if error:
            raise error
        text = claude_cli.parse_claude_message(chunks)
        if self.session_id and text:
            session_manager.add_assistant_response(
                self.session_id, Message(role="assistant", content=text)
            )
        yield Completion(text=text, model=self.model, prompt=self.args["prompt"], chunks=chunks)

    async def complete(self) -> Completion:
        """Run to the end; raise CompletionError on failure or an empty answer."""
        async for item in self.stream():
            if isinstance(item, Completion):
                if not item.text:
                    raise CompletionError(500, "internal_error", "No response from Claude Code")
                return item
        raise AssertionError("stream() always ends with a Completion or an error")


def check_claude_auth() -> None:
    auth_valid, auth_info = validate_claude_code_auth()
    if not auth_valid:
        errors = "; ".join(auth_info.get("errors", [])) or auth_info.get("method", "none")
        raise CompletionError(
            503,
            "auth_not_configured",
            f"Claude Code authentication failed ({errors}). "
            "Check /v1/auth/status for detailed authentication information.",
        )


def prepare(
    *,
    model: str,
    messages: List[Message],
    enable_tools: bool = False,
    headers: Optional[Mapping[str, str]] = None,
    thinking: Optional[Dict[str, Any]] = None,
    effort: Optional[str] = None,
    system_suffix: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Run:
    """Validate a request and build its Run; raise CompletionError if it can't run.

    headers: the request's X-Claude-* headers, which override thinking and effort.
    system_suffix: appended to the system prompt (chat's sampling instructions).
    session_id: merge with the stored session and store the reply.
    """
    if enable_tools and not constants.ALLOW_TOOLS:
        raise CompletionError(403, "tools_disabled", TOOLS_DISABLED_MESSAGE)
    check_claude_auth()

    # Apply MODEL_ALIASES and reject clearly non-Claude names
    resolved = ParameterValidator.resolve_model(model)
    if resolved is None:
        raise CompletionError(400, "model_not_found", unknown_model_message(model))

    options = ParameterValidator.extract_claude_headers(dict(headers or {}))
    if "max_thinking_tokens" in options:
        thinking = ParameterValidator.thinking_from_budget(options["max_thinking_tokens"])

    if enable_tools:
        tools = {
            "max_turns": options.get("max_turns", 10),
            "allowed_tools": DEFAULT_ALLOWED_TOOLS,
            # X-Claude-Disallowed-Tools narrows the allowed set
            "disallowed_tools": options.get("disallowed_tools"),
            # Headless: nobody is there to answer permission prompts
            "permission_mode": "bypassPermissions",
        }
        logger.info(f"Tools enabled by user request: {DEFAULT_ALLOWED_TOOLS}")
    else:
        # The prompt may carry untrusted text; with tools on it would drive an agent
        # that can run Bash with permissions bypassed
        tools = {
            "max_turns": 1,
            "allowed_tools": None,
            "disallowed_tools": CLAUDE_TOOLS,
            "permission_mode": None,
        }

    messages, session_id = session_manager.process_messages(messages, session_id)
    prompt, system_prompt = MessageAdapter.messages_to_prompt(messages)
    if system_suffix:
        system_prompt = f"{system_prompt}\n\n{system_suffix}" if system_prompt else system_suffix

    args = dict(
        prompt=prompt,
        system_prompt=system_prompt,
        model=resolved,
        thinking=thinking,
        effort=options.get("effort", effort),
        fallback_model=options.get("fallback_model"),
        **tools,
    )
    return Run(args, model, session_id)
