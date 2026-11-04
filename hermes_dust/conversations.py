"""Dust conversations: open one, continue one, follow the agent's answer.

Ported from opencode-dust ``src/dust/conversations.ts`` (pure protocol
helpers; the streaming reconnect loop lives in ``client.py`` where the host
owns cancellation). HTTP is injected through ``DustRequestContext``-like
callables so tests never touch the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hermes_dust.constants import USER_MESSAGE_ORIGIN
from hermes_dust.http import DustApiError

# Agent events after which no further events will arrive for a turn.
TERMINAL_EVENT_TYPES = frozenset(
    {
        "agent_message_success",
        "agent_message_gracefully_stopped",
        "agent_error",
        "agent_generation_cancelled",
        "user_message_error",
    }
)


class DustConversationGoneError(Exception):
    """A conversation Dust will not accept further messages for."""

    def __init__(self, conversation_id: str, cause: Exception) -> None:
        super().__init__(f"Dust conversation {conversation_id} is no longer available.")
        self.conversation_id = conversation_id
        self.__cause__ = cause


@dataclass(frozen=True)
class DustTurn:
    agent_s_id: str
    content: str
    username: str = "hermes"
    full_name: str = "hermes"
    email: str = ""
    timezone: str = "UTC"
    model_provider: str | None = None
    model_id: str | None = None
    reasoning_effort: str | None = None
    space_id: str | None = None
    content_fragments: tuple[dict[str, str], ...] = ()
    client_side_mcp_server_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DustTurnHandle:
    conversation_id: str
    user_message_id: str
    # The reply Dust already created, when the send response carried it.
    # The send routes are read-your-writes (strongly consistent) and return
    # the agent message as soon as it exists; the separate conversation GET
    # is served from lagging read replicas and can hide it for minutes.
    agent_message_id: str | None = None


def message_payload(turn: DustTurn) -> dict[str, Any]:
    """Build the message payload shared by both send routes.

    ``origin`` is stamped here and nowhere else, and ``DustTurn`` has no
    field that could change it. That is deliberate: Dust bills a turn as
    programmatic unless the origin is one it classifies as interactive, so
    an override would be a way to accidentally pay API prices.
    """
    payload: dict[str, Any] = {
        "content": turn.content,
        "mentions": [{"configurationId": turn.agent_s_id}],
        "context": {
            "username": turn.username,
            "fullName": turn.full_name,
            "email": turn.email,
            "timezone": turn.timezone,
            "origin": USER_MESSAGE_ORIGIN,
            "clientSideMCPServerIds": list(turn.client_side_mcp_server_ids),
        },
    }
    if turn.model_provider is not None and turn.model_id is not None:
        selection: dict[str, str] = {"providerId": turn.model_provider, "modelId": turn.model_id}
        if turn.reasoning_effort is not None:
            selection["reasoningEffort"] = turn.reasoning_effort
        payload["modelSelection"] = selection
    return payload


def _require_id(value: Any, route: str, what: str) -> str:
    if isinstance(value, str) and value != "":
        return value
    raise ValueError(f"Dust {route} returned no usable {what} id.")


def last_messages(conversation: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The live version of every message in a conversation."""
    content = (conversation or {}).get("content", [])
    live: list[dict[str, Any]] = []
    for versions in content:
        if versions:
            live.append(versions[-1])
    return live


def find_agent_message_id(conversation: dict[str, Any], user_message_id: str) -> str | None:
    """Find the agent message replying to a user message."""
    for message in last_messages(conversation):
        if message.get("type") == "agent_message" and message.get("parentMessageId") == user_message_id:
            s_id = message.get("sId")
            return s_id if isinstance(s_id, str) else None
    return None


def find_agent_message_id_in(agent_messages: list[dict[str, Any]], user_message_id: str) -> str | None:
    """Find the agent message replying to a user message among a flat list.

    The continue route (``POST …/messages``) answers with ``agentMessages``,
    a flat list of the agent messages it created for that user message, rather
    than a whole conversation. Matching on ``parentMessageId`` keeps this the
    same reply lookup as :func:`find_agent_message_id`.
    """
    for message in agent_messages:
        if message.get("type") == "agent_message" and message.get("parentMessageId") == user_message_id:
            s_id = message.get("sId")
            return s_id if isinstance(s_id, str) else None
    return None


_FINISHED_STATES = {
    "succeeded": "succeeded",
    "gracefully_stopped": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "interrupted": "cancelled",
}


def message_state(message: dict[str, Any]) -> str:
    """Where a Dust agent message has got to; unknown reads as running."""
    status = message.get("status")
    return _FINISHED_STATES.get(status, "running") if isinstance(status, str) else "running"


@dataclass(frozen=True)
class DustAgentMessage:
    s_id: str
    state: str
    content: str
    error: str | None = None


def read_agent_message(conversation: dict[str, Any], agent_message_id: str) -> DustAgentMessage | None:
    """Read an agent message's state and stored answer."""
    for message in last_messages(conversation):
        if message.get("type") != "agent_message" or message.get("sId") != agent_message_id:
            continue
        raw = message.get("error")
        error = raw if isinstance(raw, dict) else {}
        detail = error.get("message") if isinstance(error.get("message"), str) and error.get("message") else None
        detail = detail or (error.get("code") if isinstance(error.get("code"), str) and error.get("code") else None)
        content = message.get("content")
        return DustAgentMessage(
            s_id=agent_message_id,
            state=message_state(message),
            content=content if isinstance(content, str) else "",
            error=detail,
        )
    return None


def find_unfinished_agent_message_ids(conversation: dict[str, Any]) -> list[str]:
    """Name every generation still running in a conversation."""
    ids: list[str] = []
    for message in last_messages(conversation):
        if message.get("type") == "agent_message" and message_state(message) == "running":
            s_id = message.get("sId")
            if isinstance(s_id, str):
                ids.append(s_id)
    return ids


def create_conversation(
    dust_json: Callable[..., Any], turn: DustTurn, space_id: str | None = None
) -> DustTurnHandle:
    """Open a new conversation and send its first message."""
    body = dust_json(
        "assistant/conversations",
        method="POST",
        body={
            "visibility": "unlisted",
            **({} if (space_id or turn.space_id) is None else {"spaceId": space_id or turn.space_id}),
            **(
                {}
                if not turn.content_fragments
                else {"contentFragments": list(turn.content_fragments)}
            ),
            "message": message_payload(turn),
        },
    )
    conversation = body.get("conversation", {}) if isinstance(body, dict) else {}
    user_message_id = next(
        (
            m.get("sId")
            for m in last_messages(conversation)
            if m.get("type") == "user_message"
        ),
        None,
    )
    resolved_user_id = _require_id(user_message_id, "assistant/conversations", "user message")
    # The create response already holds the agent reply Dust just queued, read
    # from the primary. Prefer it over the conversation GET, which is served
    # from read replicas that can lag the write for minutes.
    return DustTurnHandle(
        conversation_id=_require_id(conversation.get("sId"), "assistant/conversations", "conversation"),
        user_message_id=resolved_user_id,
        agent_message_id=find_agent_message_id(conversation, resolved_user_id),
    )


def post_message(
    dust_json: Callable[..., Any], conversation_id: str, turn: DustTurn
) -> DustTurnHandle:
    """Send a further message to an existing conversation."""
    try:
        body = dust_json(
            f"assistant/conversations/{conversation_id}/messages",
            method="POST",
            body=message_payload(turn),
        )
    except DustApiError as exc:
        if exc.kind == "gone":
            raise DustConversationGoneError(conversation_id, exc) from exc
        raise
    message = body.get("message", {}) if isinstance(body, dict) else {}
    user_message_id = _require_id(message.get("sId"), "assistant/conversations/messages", "user message")
    agent_messages = body.get("agentMessages") if isinstance(body, dict) else None
    return DustTurnHandle(
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        agent_message_id=(
            find_agent_message_id_in(agent_messages, user_message_id)
            if isinstance(agent_messages, list)
            else None
        ),
    )
