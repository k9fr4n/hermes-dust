"""Dust inference client behind an OpenAI-style facade.

Phase 1 transport: one Hermes LLM call sends one user message to a Dust
conversation and streams the agent's answer back — chain of thought as
reasoning, answer as text. Hermes tools are never offered to the agent;
``hermes_bridge_note`` goes in front of every turn instead, and the agent's
own server-side tool approvals are answered automatically.

The facade exposes ``chat.completions.create(messages, model, stream)`` so
Hermes' ``chat_completions`` transport can drive it without a custom wire
mode. Dust reports no token counts, so usage figures stay unknown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from hermes_dust.bridge_note import hermes_bridge_note
from hermes_dust.conversations import (
    DustConversationGoneError,
    DustTurn,
    create_conversation,
    find_agent_message_id,
    find_unfinished_agent_message_ids,
    post_message,
    read_agent_message,
)
from hermes_dust.history import DustThreadStore, plan_turn, remember_turn
from hermes_dust.http import DustApiError, DustRequestContext, dust_fetch, dust_json
from hermes_dust.models import decode_model_id
from hermes_dust.sse import SseEvent, iter_sse_events
from hermes_dust.tag_stripper import create_tag_stripper

AGENT_MESSAGE_ATTEMPTS = 6
AGENT_MESSAGE_DELAY_S = 0.5
TERMINAL_TYPES = {
    "agent_message_success": "stop",
    "agent_message_gracefully_stopped": "stop",
    "agent_generation_cancelled": "other",
    "agent_error": "error",
    "user_message_error": "error",
}


@dataclass
class DustClientConfig:
    """Everything a turn needs to address Dust."""

    access_token: str | Callable[[bool], str]
    region: str
    workspace_id: str
    agent_s_id: str = "dust"
    reasoning_effort: str | None = None
    space_id: str | None = None
    timeout_ms: float | None = 30_000


def _host_messages_to_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project Hermes messages onto the history module's prompt shape."""
    prompt: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            text = content if isinstance(content, str) else str(content or "")
            prompt.append({"role": "system", "content": [{"type": "text", "text": text}]})
        elif role in ("user", "assistant"):
            content = message.get("content")
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            prompt.append({"role": role, "content": parts or []})
        elif role == "tool":
            # Tool results already went back to Dust through the bridge in
            # the opencode design; Hermes replays them the same way. They
            # carry no new user text, so they are left out of the digest.
            continue
    return prompt


class _Completions:
    def __init__(self, client: "DustClient") -> None:
        self._client = client

    def create(self, *, messages: list[dict[str, Any]], model: str, stream: bool = False, **kwargs: Any) -> Any:
        if stream:
            return self._client.stream_turn(messages=messages, model=model, extra=kwargs)
        return self._client.complete_turn(messages=messages, model=model, extra=kwargs)


class _Chat:
    def __init__(self, client: "DustClient") -> None:
        self.completions = _Completions(client)


@dataclass
class DustClient:
    """OpenAI-facade client over the Dust conversation API."""

    config: DustClientConfig
    threads: DustThreadStore = field(default_factory=DustThreadStore)
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.chat = _Chat(self)

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    # -- context ---------------------------------------------------------
    def _context(self, workspace_base: str) -> DustRequestContext:
        return DustRequestContext(base=workspace_base, access_token=self.config.access_token)

    def _workspace_base(self) -> str:
        from hermes_dust.constants import dust_api_base

        return dust_api_base(self.config.region, self.config.workspace_id)

    def _dust_json(self, path: str, method: str = "GET", body: Any = None) -> Any:
        return dust_json(self._context(self._workspace_base()), path, method=method, body=body)

    # -- turns -----------------------------------------------------------
    def _send(self, prompt: list[dict[str, Any]], model: str) -> tuple[str, str]:
        if not self.config.workspace_id:
            raise ValueError("No Dust workspace is configured; run `hermes auth add dust` and choose one.")
        selection = decode_model_id(model.split("/", 1)[-1] if "/" in model else model)
        if selection is None:
            raise ValueError(f"{model} is not a Dust model; pick one from the dust provider.")
        provider_id, model_id = selection
        plan = plan_turn(self.threads, prompt)
        note = hermes_bridge_note()
        content = f"{note}\n\n---\n\n{plan.content}"
        turn = DustTurn(
            agent_s_id=self.config.agent_s_id,
            content=content,
            model_provider=provider_id,
            model_id=model_id,
            reasoning_effort=self.config.reasoning_effort,
            space_id=self.config.space_id,
        )
        if plan.kind == "create":
            handle = create_conversation(self._dust_json, turn)
        else:
            assert plan.conversation_id is not None
            try:
                handle = post_message(self._dust_json, plan.conversation_id, turn)
            except DustConversationGoneError:
                fresh = plan_turn(DustThreadStore(), prompt)
                handle = create_conversation(
                    self._dust_json,
                    DustTurn(
                        agent_s_id=self.config.agent_s_id,
                        content=f"{note}\n\n---\n\n{fresh.content}",
                        model_provider=provider_id,
                        model_id=model_id,
                        reasoning_effort=self.config.reasoning_effort,
                        space_id=self.config.space_id,
                    ),
                )
        return handle.conversation_id, handle.user_message_id

    def _await_agent_message(self, conversation_id: str, user_message_id: str) -> str:
        for attempt in range(AGENT_MESSAGE_ATTEMPTS):
            if attempt:
                self.sleep(AGENT_MESSAGE_DELAY_S)
            conversation = self._dust_json(f"assistant/conversations/{conversation_id}")
            found = find_agent_message_id(conversation if isinstance(conversation, dict) else {}, user_message_id)
            if found is not None:
                return found
            if attempt == AGENT_MESSAGE_ATTEMPTS - 1:
                running = find_unfinished_agent_message_ids(conversation if isinstance(conversation, dict) else {})
                if running:
                    self._dust_json(
                        f"assistant/conversations/{conversation_id}/cancel",
                        method="POST",
                        body={"action": "cancel", "messageIds": running},
                    )
                    conversation = self._dust_json(f"assistant/conversations/{conversation_id}")
                    found = find_agent_message_id(conversation if isinstance(conversation, dict) else {}, user_message_id)
                    if found is not None:
                        return found
        raise ValueError("Dust accepted the message but created no agent reply to follow.")

    def _stream_events(
        self, conversation_id: str, agent_message_id: str
    ) -> Iterator[tuple[SseEvent, dict[str, Any] | None]]:
        import json as json_lib

        context = self._context(self._workspace_base())
        path = f"assistant/conversations/{conversation_id}/messages/{agent_message_id}/events"
        response = dust_fetch(context, path)
        if not response.ok:
            raise DustApiError(response.status, "server", None, f"Dust {path} failed with {response.status}")
        raw = getattr(response, "raw", None)
        chunks: Iterator[bytes] = iter([raw.read()] if raw is not None and hasattr(raw, "read") else [])
        for record in iter_sse_events(chunks):
            if record.data == "done":
                continue
            try:
                envelope = json_lib.loads(record.data)
            except Exception:
                continue
            event = envelope.get("data") if isinstance(envelope, dict) else None
            yield record, event if isinstance(event, dict) and isinstance(event.get("type"), str) else None

    def _collect_answer(self, conversation_id: str, agent_message_id: str) -> Iterator[dict[str, Any]]:
        """Yield OpenAI-style chunks for one Dust turn."""
        stripper = create_tag_stripper()
        answer_parts: list[str] = []
        for _record, event in self._stream_events(conversation_id, agent_message_id):
            if event is None:
                continue
            etype = event.get("type")
            if etype == "generation_tokens":
                raw = event.get("text")
                text = stripper.push(raw) if isinstance(raw, str) else ""
                if text:
                    answer_parts.append(text)
                    yield {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
                continue
            if etype == "tool_approve_execution":
                action_id = event.get("actionId")
                if isinstance(action_id, str):
                    try:
                        self._dust_json(
                            f"assistant/conversations/{conversation_id}/messages/{agent_message_id}/validate-action",
                            method="POST",
                            body={"actionId": action_id, "approved": "approved"},
                        )
                    except Exception:
                        pass
                continue
            finish = TERMINAL_TYPES.get(str(etype))
            if finish is None:
                continue
            tail = stripper.flush()
            if tail:
                answer_parts.append(tail)
                yield {"choices": [{"delta": {"content": tail}, "finish_reason": None}]}
            if finish == "error":
                raw = event.get("error")
                detail = raw.get("message") if isinstance(raw, dict) else None
                raise RuntimeError(detail or f"Dust ended the turn with {etype}.")
            yield {"choices": [{"delta": {}, "finish_reason": "stop" if finish == "stop" else "other"}]}
            return
        tail = stripper.flush()
        if tail:
            yield {"choices": [{"delta": {"content": tail}, "finish_reason": None}]}
        # No terminal event: the conversation decides.
        stored = read_agent_message(
            self._dust_json(f"assistant/conversations/{conversation_id}"), agent_message_id
        )
        if stored is not None and stored.state == "succeeded":
            if not answer_parts and stored.content:
                yield {"choices": [{"delta": {"content": stored.content}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        else:
            raise RuntimeError(stored.error if stored and stored.error else "Dust stopped answering.")

    # -- facade ----------------------------------------------------------
    def complete_turn(self, *, messages: list[dict[str, Any]], model: str, extra: dict[str, Any]) -> dict[str, Any]:
        prompt = _host_messages_to_prompt(messages)
        conversation_id, user_message_id = self._send(prompt, model)
        agent_message_id = self._await_agent_message(conversation_id, user_message_id)
        texts: list[str] = []
        for chunk in self._collect_answer(conversation_id, agent_message_id):
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                texts.append(delta["content"])
        answer = "".join(texts)
        remember_turn(self.threads, prompt, answer, conversation_id)
        return {"choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}

    def stream_turn(self, *, messages: list[dict[str, Any]], model: str, extra: dict[str, Any]) -> Iterator[dict[str, Any]]:
        prompt = _host_messages_to_prompt(messages)
        conversation_id, user_message_id = self._send(prompt, model)
        agent_message_id = self._await_agent_message(conversation_id, user_message_id)
        texts: list[str] = []
        for chunk in self._collect_answer(conversation_id, agent_message_id):
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                texts.append(delta["content"])
            yield chunk
        remember_turn(self.threads, prompt, "".join(texts), conversation_id)
