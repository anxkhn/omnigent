"""Reasoning-effort validation helpers shared across client/runtime paths."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable

from omnigent.llms.errors import PermanentLLMError

EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
EFFORT_CLEAR_VALUES = frozenset({"default", "off", "reset"})

OPENAI_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
CLAUDE_EFFORTS = ANTHROPIC_EFFORTS
CODEX_EFFORTS = OPENAI_EFFORTS
OPENAI_AGENTS_EFFORTS = OPENAI_EFFORTS
GEMINI_EFFORTS = frozenset({"low", "medium", "high"})
ANTIGRAVITY_EFFORTS = GEMINI_EFFORTS
# The GitHub Copilot SDK's ``create_session(reasoning_effort=...)`` accepts
# exactly these levels (``copilot.session.ReasoningEffort`` literal); per-model
# support is gated by the Copilot backend (``list_models()``).
COPILOT_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})

# xAI / Grok accepts the OpenAI-compatible ``reasoning_effort`` parameter on only
# a subset of models. Sending it to the others (``grok-3``, ``grok-4``,
# ``grok-code-fast-1``, ``grok-4-fast-reasoning``, ...) is rejected with HTTP 400.
#
# Rather than hand-maintain an allow-list -- which silently drops the parameter
# for every new reasoning-capable Grok until someone edits the tuple, and which
# the model-id space actively fights (``grok-4`` rejects it but ``grok-4.3``
# accepts it; ``grok-3`` rejects it but ``grok-3-mini`` accepts it) -- send it
# optimistically and self-heal. The Chat Completions caller catches an HTTP 400
# that names the parameter, strips it, retries once, and records the rejection
# via :func:`record_reasoning_effort_rejection` so the rest of the process skips
# the wasted round trip. See ``omnigent/llms/client.py``.

# Seed set of ``(provider, model)`` pairs known not to accept ``reasoning_effort``.
# Purely an optimization + safety net: it skips the first wasted round trip for
# models we already know reject the parameter, and it is the ONLY line of defense
# for a model that *silently* accepts and ignores it (no 400 to learn from).
# Entries are exact, lower-cased model ids -- never prefixes -- because the Grok
# id space collides (``grok-4`` vs ``grok-4.3``, ``grok-3`` vs ``grok-3-mini``).
# A stale or missing entry costs at most one extra round trip, never a wrong
# answer. Refs: docs.x.ai reasoning docs; issue #1686.
_KNOWN_UNSUPPORTED_REASONING_EFFORT: frozenset[tuple[str, str]] = frozenset(
    ("xai", model)
    for model in (
        "grok-3",
        "grok-4",
        "grok-4-fast-reasoning",
        "grok-code-fast-1",
    )
)

# Rejections learned at runtime from a 400. Process-lifetime, guarded by a lock
# because ``Client`` calls run concurrently across asyncio tasks and threads.
_learned_unsupported: set[tuple[str, str]] = set()
_learned_lock = threading.Lock()

# xAI reports an unsupported reasoning-effort parameter with an HTTP 400 whose
# body names the parameter, e.g.
# ``{"error": "Model grok-4 does not support parameter reasoningEffort."}``
# (note the camel-cased ``reasoningEffort``). Match either spelling, plus an
# unsupported-ish phrase, so unrelated 400s (context overflow, bad schema) do
# not trip the fallback.
_REASONING_EFFORT_NAMED_RE = re.compile(r"reasoning[_\s-]?effort", re.IGNORECASE)
_PARAM_UNSUPPORTED_RE = re.compile(
    r"does not support|not supported|unsupported|unknown|unrecognized"
    r"|not a valid|is not valid|isn't supported|is not permitted",
    re.IGNORECASE,
)


def should_send_reasoning_effort(provider: str, model: str) -> bool:
    """Return whether ``reasoning_effort`` should be sent for *provider*/*model*.

    Optimistic by default: returns ``True`` unless the pair is in the seed set of
    known-unsupported models or has been learned unsupported this process (via
    :func:`record_reasoning_effort_rejection`). Replaces the old hand-maintained
    allow-list so a newly released reasoning-capable model is never silently
    denied the parameter; a wrong guess is corrected by the caller's one-shot
    retry rather than by editing this file.

    :param provider: The routed provider id, e.g. ``"xai"``.
    :param model: The model name without provider prefix, e.g. ``"grok-4"``.
    :returns: ``True`` if ``reasoning_effort`` may be sent for this model.
    """
    key = (provider, model.lower())
    if key in _KNOWN_UNSUPPORTED_REASONING_EFFORT:
        return False
    with _learned_lock:
        return key not in _learned_unsupported


def record_reasoning_effort_rejection(provider: str, model: str) -> None:
    """Remember that *provider*/*model* rejected ``reasoning_effort``.

    Called after a 400 that named the parameter so the next call for the same
    model skips it. Process-lifetime only; not persisted.

    :param provider: The routed provider id, e.g. ``"xai"``.
    :param model: The model name without provider prefix, e.g. ``"grok-4"``.
    """
    with _learned_lock:
        _learned_unsupported.add((provider, model.lower()))


def reset_reasoning_effort_rejections() -> None:
    """Clear the learned-rejection cache. Test seam."""
    with _learned_lock:
        _learned_unsupported.clear()


def is_unsupported_reasoning_effort_error(status_code: int, body: str) -> bool:
    """Return whether a provider error means ``reasoning_effort`` is unsupported.

    Conservative: only an HTTP 400 whose body both names the parameter (in either
    ``reasoning_effort`` or ``reasoningEffort`` spelling) and carries an
    unsupported-ish phrase returns ``True``. Everything else returns ``False`` so
    the error propagates unchanged (e.g. a context-overflow 400 still reaches the
    compaction path).

    :param status_code: The HTTP status code from the provider, e.g. ``400``.
    :param body: The raw HTTP response body string.
    :returns: ``True`` if the parameter should be stripped and the call retried.
    """
    if status_code != 400:
        return False
    if not _REASONING_EFFORT_NAMED_RE.search(body):
        return False
    return bool(_PARAM_UNSUPPORTED_RE.search(body))


def format_supported(values: Iterable[str]) -> str:
    """Return a stable comma-separated supported-values string."""
    order = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    values_set = set(values)
    return ", ".join(value for value in order if value in values_set)


def unsupported_effort_message(effort: str, provider: str, supported: Iterable[str]) -> str:
    """Build a clear unsupported-effort error message."""
    return (
        f"Effort {effort!r} is not supported by {provider}; "
        f"supported values: {format_supported(supported)}"
    )


def validate_effort(effort: object, provider: str, supported: Iterable[str]) -> str | None:
    """Validate *effort* against *supported*, returning a string or None."""
    if effort is None or effort == "":
        return None
    effort_str = str(effort)
    if effort_str not in set(supported):
        raise ValueError(unsupported_effort_message(effort_str, provider, supported))
    return effort_str


def validate_effort_or_llm_error(
    effort: object,
    provider: str,
    supported: Iterable[str],
) -> str | None:
    """Validate for native LLM paths, raising non-retryable PermanentLLMError."""
    try:
        return validate_effort(effort, provider, supported)
    except ValueError as exc:
        raise PermanentLLMError(str(exc), code="unsupported_reasoning_effort") from exc
