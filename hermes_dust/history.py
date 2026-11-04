"""Mapping the host's prompt onto a Dust conversation.

Ported from opencode-dust ``src/provider/history.ts``. The host sends the
whole history on every call and owns it (compaction, revert, fork). Dust
owns an append-only transcript no client may rewrite. This module decides,
per call, whether a Dust conversation already holds a faithful prefix of the
prompt — in which case only the new user message is sent — or whether a new
conversation must be opened.

A call carries no session id, so threads are keyed by a digest of the
history Dust has seen, answer included. A miss is never wrong, only dearer:
it replays the transcript into a fresh conversation.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

DEFAULT_CAPACITY = 256


@dataclass
class RenderedMessage:
    role: Literal["user", "assistant"]
    text: str


@dataclass
class TurnPlan:
    kind: Literal["create", "append"]
    content: str
    conversation_id: str | None = None


class DustThreadStore:
    """Which Dust conversation holds which history. Bounded LRU."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self._threads: OrderedDict[str, str] = OrderedDict()
        self._capacity = capacity

    def get(self, digest: str) -> str | None:
        conversation_id = self._threads.get(digest)
        if conversation_id is None:
            return None
        self._threads.move_to_end(digest)
        return conversation_id

    def set(self, digest: str, conversation_id: str) -> None:
        self._threads[digest] = conversation_id
        self._threads.move_to_end(digest)
        while len(self._threads) > self._capacity:
            self._threads.popitem(last=False)


def _render_text(content: list[dict]) -> str:
    return "\n".join(part.get("text", "") for part in content if part.get("type") == "text" and part.get("text"))


def render_all(messages: list[dict]) -> list[RenderedMessage]:
    """Render a prompt's conversation, leaving out system and tool parts.

    Files, reasoning and tool parts are dropped: Dust keeps its own
    reasoning and runs its own tools, so echoing either back would duplicate
    its transcript.
    """
    rendered: list[RenderedMessage] = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _render_text(message.get("content", []))
        if role == "assistant" and rendered and rendered[-1].role == "assistant":
            # One Dust answer can come back as several assistant messages:
            # the host splits it around bridged tool calls. Dust holds it as
            # one, so it is digested as one.
            previous = rendered[-1]
            joined = "\n".join(t for t in (previous.text, text) if t)
            rendered[-1] = RenderedMessage(role="assistant", text=joined)
            continue
        rendered.append(RenderedMessage(role=role, text=text))
    return rendered


def digest_history(messages: list[RenderedMessage]) -> str:
    """Digest a rendered history.

    Each field is length-prefixed so no regrouping of the same characters
    can produce the same digest and let a rewritten history pass as a
    prefix. The system prompt is deliberately left out: the host rebuilds it
    per call, and Dust only ever received it once, at the top.
    """
    digest = hashlib.sha256()
    for message in messages:
        # Whitespace-blind: the stream and the host transcript disagree on
        # breaks between text parts, and that difference is not a rewrite.
        text = re.sub(r"\s+", "", message.text)
        digest.update(f"{len(message.role)}:{message.role}".encode())
        digest.update(f"{len(text)}:{text}".encode())
    return digest.hexdigest()


def _with_system_prompt(messages: list[dict], body: str) -> str:
    system = "\n\n".join(
        part.get("text", "")
        for message in messages
        if message.get("role") == "system"
        for part in message.get("content", [])
        if part.get("type") == "text"
    )
    if system.strip() == "":
        return body
    return f"{system}\n\n---\n\n{body}"


def _transcript(messages: list[RenderedMessage]) -> str:
    return "\n".join(m.text for m in messages if m.text != "")


def plan_turn(threads: DustThreadStore, prompt: list[dict]) -> TurnPlan:
    """Decide how to send a prompt to Dust."""
    messages = render_all(prompt)
    last_answer = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "assistant"), -1)
    if last_answer != -1:
        conversation_id = threads.get(digest_history(messages[: last_answer + 1]))
        # Only the human's own messages are new to Dust; an append carrying
        # nothing from the human leaves Dust nothing to answer.
        delta = [m for m in messages[last_answer + 1 :] if m.role == "user"]
        if conversation_id is not None and _transcript(delta) != "":
            return TurnPlan(kind="append", content=_transcript(delta), conversation_id=conversation_id)
    return TurnPlan(kind="create", content=_with_system_prompt(prompt, _transcript(messages)))


def remember_turn(threads: DustThreadStore, prompt: list[dict], answer: str, conversation_id: str) -> None:
    """Record that a conversation now holds a prompt and Dust's answer."""
    threads.set(digest_history([*render_all(prompt), RenderedMessage(role="assistant", text=answer)]), conversation_id)
