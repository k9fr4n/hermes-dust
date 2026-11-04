"""Minimal ``text/event-stream`` reader.

Ported from opencode-dust ``src/dust/sse.ts``. Dust serves both the agent
answer stream and the reverse MCP request stream as SSE. The framing is
parsed here so no EventSource dependency is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class SseEvent:
    """One SSE record: the fields this plugin reads, decoded."""

    data: str
    id: str | None = None
    event: str | None = None


def _field_value(rest: str) -> str:
    return rest[1:] if rest.startswith(" ") else rest


def parse_record(record: str) -> SseEvent | None:
    """Parse one complete record's lines, or None when it carries no data."""
    data: list[str] = []
    event_id: str | None = None
    event_type: str | None = None
    for line in record.split("\n"):
        if line == "" or line.startswith(":"):
            continue
        colon = line.find(":")
        if colon == -1:
            field, value = line, ""
        else:
            field, value = line[:colon], _field_value(line[colon + 1 :])
        if field == "data":
            data.append(value)
        elif field == "id":
            event_id = value
        elif field == "event":
            event_type = value
    if not data:
        return None
    return SseEvent(data="\n".join(data), id=event_id, event=event_type)


def iter_sse_events(chunks: Iterable[bytes | str]) -> Iterator[SseEvent]:
    """Decode a response body as a sequence of SSE records.

    A record the server left unterminated when it closed the body is still
    yielded: Dust ends a completed turn by closing the connection rather than
    by writing a final blank line, and dropping that record would lose the
    last event of every stream.
    """
    buffer = ""
    for chunk in chunks:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        buffer += text.replace("\r\n", "\n")
        split = buffer.find("\n\n")
        while split != -1:
            event = parse_record(buffer[:split])
            buffer = buffer[split + 2 :]
            if event is not None:
                yield event
            split = buffer.find("\n\n")
    trailing = parse_record(buffer.replace("\r\n", "\n"))
    if trailing is not None:
        yield trailing
