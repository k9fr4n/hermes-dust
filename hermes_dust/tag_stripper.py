"""Removing the delimiters Dust leaves in the raw token stream.

Ported from opencode-dust ``src/dust/tag-stripper.ts``. Some Dust agents
split reasoning from answer on ``<thinking>``/``<response>`` tags the model
emits. Dust honours them — the stored message has content and chainOfThought
as separate clean fields — but the generation_tokens events still carry the
tags, so a client rendering tokens verbatim shows them as prose.

Stripping survives chunk boundaries: tokens do not respect tag boundaries,
so this holds back only the tail that could still become a delimiter.
"""

from __future__ import annotations

_TAGS = ("<thinking>", "</thinking>", "<response>", "</response>")
_LONGEST = max(len(tag) for tag in _TAGS)


def _could_become_tag(text: str) -> bool:
    return any(tag.startswith(text) for tag in _TAGS)


class TagStripper:
    """Stateful stripper for one stream."""

    def __init__(self) -> None:
        self._carry = ""

    def push(self, chunk: str) -> str:
        """Feed one delta; return the text safe to render now."""
        pending = self._carry + chunk
        out: list[str] = []
        while pending != "":
            open_at = pending.find("<")
            if open_at == -1:
                out.append(pending)
                pending = ""
                break
            out.append(pending[:open_at])
            pending = pending[open_at:]
            matched = next((tag for tag in _TAGS if pending.startswith(tag)), None)
            if matched is not None:
                pending = pending[len(matched) :]
                continue
            # Still short enough to be a tag prefix: hold it and wait.
            if len(pending) < _LONGEST and _could_become_tag(pending):
                break
            # Long enough, or ruled out: this "<" is the model's own.
            out.append("<")
            pending = pending[1:]
        self._carry = pending
        return "".join(out)

    def flush(self) -> str:
        """End the stream; release held-back text that never became a tag."""
        rest = self._carry
        self._carry = ""
        return rest


def create_tag_stripper() -> TagStripper:
    """Build a stripper for one generation."""
    return TagStripper()
