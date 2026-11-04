"""Dust inference client behind an OpenAI-style facade, with the pod bridge.

Each Hermes LLM call sends one user message to a Dust conversation and
streams the agent's answer back. Hermes tools are never sent to Dust; the
bridge note goes in front of every turn, and while the turn streams the
client polls the exchange pod:

- A request the agent left (JSON batch, or shell script) is emitted as a
  single OpenAI ``tool_calls`` round — one call at a time, requests in
  order. Hermes executes it in its own loop, with its own approvals.
- The results come back in the next client call as ``role: tool`` messages.
  The client writes the ``.out`` the sandbox helper waits on, then resumes
  the same Dust event stream from its cursor and yields the rest.
- A request nobody waits on (``.detached`` marker) is queued as an
  automatic message for the next turn.

Dust reports no token counts, so usage figures stay unknown.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from hermes_dust.bridge.protocol import NOT_BRIDGED, pod_name_for, tool_signatures
from hermes_dust.bridge.session import BridgeSession
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
from hermes_dust.http import DustRequestContext, dust_json
from hermes_dust.models import REASONING_EFFORTS, decode_model_id
from hermes_dust.sse import iter_sse_events
from hermes_dust.tag_stripper import create_tag_stripper

AGENT_MESSAGE_DELAY_S = 2.0
# Poll schedule while waiting for Dust to expose the agent reply. Dust's
# conversation reads are served live (no CDN cache) but EU read replicas
# have been observed lagging the write by over a minute: short windows fail
# turns whose reply already exists. ~38 polls over ~4 minutes.
AGENT_MESSAGE_DELAYS = tuple([2.0] * 10 + [5.0] * 10 + [10.0] * 18)
POST_CANCEL_ATTEMPTS = 5
TERMINAL_TYPES = {
    "agent_message_success": "stop",
    "agent_message_gracefully_stopped": "stop",
    "agent_generation_cancelled": "other",
    "agent_error": "error",
    "user_message_error": "error",
}

# Grace polls for a request left in the same breath as the turn's end.
GRACE_POLLS = 3

# How long one event-stream connection may stay silent before reopening.
REOPEN_DELAY_S = 0.5
MAX_REOPENS = 5


@dataclass
class DustClientConfig:
    """Everything a turn needs to address Dust."""

    access_token: str | Any
    region: str
    workspace_id: str
    agent_s_id: str = "dust"
    reasoning_effort: str | None = None
    space_id: str | None = None
    project_dir: str = ""
    timeout_ms: float | None = 30_000


@dataclass
class _TurnState:
    conversation_id: str
    agent_message_id: str
    send_prompt: list[dict[str, Any]] = field(default_factory=list)
    last_event_id: str | None = None
    answer_parts: list[str] = field(default_factory=list)
    session: BridgeSession | None = None


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
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
            text_parts = [
                {"type": "text", "text": p.get("text", "")}
                for p in parts
                if isinstance(p, dict) and p.get("type") != "tool_call"
            ]
            prompt.append({"role": role, "content": text_parts})
        # role == "tool": results already went back to Dust through the pod;
        # they carry no new user text, so they stay out of the digest.
    return prompt


def _tool_results(messages: list[dict[str, Any]]) -> list[tuple[str, str, bool]]:
    """Trailing ``role: tool`` messages as (tool_call_id, text, is_error)."""
    results: list[tuple[str, str, bool]] = []
    for message in reversed(messages):
        if message.get("role") != "tool":
            break
        tool_call_id = message.get("tool_call_id") or message.get("id") or ""
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content)
        results.append((str(tool_call_id), str(content or ""), bool(message.get("is_error", False))))
    results.reverse()
    return results


def split_model_variant(model: str) -> tuple[str, str | None]:
    """Split ``provider:model+effort`` into the Dust model id and effort.

    The ``+effort`` suffix mirrors opencode's reasoning variants
    (``none``/``light``/``medium``/``high``). An unknown suffix is kept as
    part of the model id rather than silently dropped, so a typo fails at
    Dust with the id visible instead of running the wrong configuration.
    """
    base, plus, suffix = model.partition("+")
    if plus and suffix in REASONING_EFFORTS:
        return base, suffix
    return model, None


def _openai_tools(extra: dict[str, Any]) -> list[dict[str, Any]]:
    tools = extra.get("tools") or []
    normalized = []
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            normalized.append({"name": fn.get("name", ""), "parameters": fn.get("parameters", {})})
        elif isinstance(tool, dict) and "name" in tool:
            normalized.append({"name": tool.get("name", ""), "parameters": tool.get("parameters", {})})
    return [t for t in normalized if t["name"] and t["name"] not in NOT_BRIDGED]


def _shell_tool_name(tools: list[dict[str, Any]]) -> str | None:
    names = [t["name"] for t in tools]
    for candidate in ("terminal", "bash", "shell"):
        if candidate in names:
            return candidate
    return None


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
    sleep: Any = time.sleep

    def __post_init__(self) -> None:
        self.chat = _Chat(self)
        self._turns: dict[str, _TurnState] = {}
        self._cancel_event = threading.Event()
        self._pod_cache: dict[str, dict[str, Any]] = {}
        self._reads = 0

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def cancel(self) -> None:
        """Interrupt the running turn: stop streaming, cancel Dust's generation."""
        self._cancel_event.set()
        for turn in list(self._turns.values()):
            if turn.session is not None:
                turn.session.interrupt()
            try:
                self._dust_json(
                    f"assistant/conversations/{turn.conversation_id}/cancel",
                    method="POST",
                    body={"action": "cancel", "messageIds": [turn.agent_message_id]},
                )
            except Exception:
                pass

    # -- contexts ------------------------------------------------------------
    def _context(self, base: str) -> DustRequestContext:
        return DustRequestContext(base=base, access_token=self.config.access_token)

    def _workspace_base(self) -> str:
        from hermes_dust.constants import dust_api_base

        return dust_api_base(self.config.region, self.config.workspace_id)

    def _private_base(self) -> str:
        from hermes_dust.constants import dust_private_api_base

        return dust_private_api_base(self.config.region, self.config.workspace_id)

    def _dust_json(self, path: str, method: str = "GET", body: Any = None) -> Any:
        return dust_json(self._context(self._workspace_base()), path, method=method, body=body)

    def _pod_ops(self) -> dict[str, Any]:
        """Exchange pod file operations, ensured once per workspace."""
        import logging

        from hermes_dust.bridge.pod import ensure_pod, make_pod_ops
        from hermes_dust.bridge.protocol import HELPER_PATH, helper_script

        key = f"{self.config.region}/{self.config.workspace_id}"
        cached = self._pod_cache.get(key)
        if cached is not None:
            return cached

        def context_fn() -> DustRequestContext:
            return self._context(self._private_base())

        def json_fn(path: str, method: str = "GET", body: Any = None) -> Any:
            return dust_json(context_fn(), path, method=method, body=body)

        pod, _created = ensure_pod(json_fn, pod_name_for(self.config.project_dir))
        ops = make_pod_ops(context_fn, pod.s_id)
        try:
            present: bool | None = None
            try:
                present = ops["read"](HELPER_PATH) == helper_script()
            except Exception:
                present = False
            if not present:
                ops["write"](HELPER_PATH, helper_script())
                present = ops["read"](HELPER_PATH) == helper_script()
            if not present:
                raise RuntimeError(f"helper {HELPER_PATH} not readable after upload")
        except Exception as exc:
            # Loud, not fatal: the turn still runs, but the agent cannot
            # reach this machine — every request would wait out the sandbox
            # timeout. The next turn retries the upload.
            logging.getLogger(__name__).warning("dust: exchange pod %s unusable: %s", pod.s_id, exc)
            return {**ops, "space_id": pod.s_id, "bridge_usable": False}
        self._pod_cache[key] = {**ops, "space_id": pod.s_id, "bridge_usable": True}
        return self._pod_cache[key]

    # -- turns -----------------------------------------------------------------
    def _send(self, prompt: list[dict[str, Any]], model: str, tools: list[dict[str, Any]]) -> _TurnState:
        if not self.config.workspace_id:
            raise ValueError("No Dust workspace is configured; run `hermes auth add dust` and choose one.")
        raw_model = model.split("/", 1)[-1] if "/" in model else model
        bare_model, variant_effort = split_model_variant(raw_model)
        selection = decode_model_id(bare_model)
        if selection is None:
            raise ValueError(f"{model} is not a Dust model; pick one from the dust provider.")
        provider_id, model_id = selection
        reasoning_effort = variant_effort or self.config.reasoning_effort
        plan = plan_turn(self.threads, prompt)
        ops = self._pod_ops()
        note = hermes_bridge_note(tool_signatures(tools) if tools else None)
        deliveries = [d for turn in self._turns.values() if turn.session for d in turn.session.deliveries]
        for turn in self._turns.values():
            if turn.session is not None:
                turn.session.deliveries.clear()
        prefix = "".join(f"{d}\n\n" for d in deliveries)

        def turn_body(content: str) -> DustTurn:
            return DustTurn(
                agent_s_id=self.config.agent_s_id,
                content=content,
                model_provider=provider_id,
                model_id=model_id,
                reasoning_effort=reasoning_effort,
                space_id=ops.get("space_id"),
            )

        def json_fn(path: str, method: str = "GET", body: Any = None) -> Any:
            return self._dust_json(path, method=method, body=body)

        content = f"{prefix}{note}\n\n---\n\n{plan.content}"
        if plan.kind == "create":
            handle = create_conversation(json_fn, turn_body(content))
        else:
            assert plan.conversation_id is not None
            try:
                handle = post_message(json_fn, plan.conversation_id, turn_body(content))
            except DustConversationGoneError:
                fresh = plan_turn(DustThreadStore(), prompt)
                handle = create_conversation(json_fn, turn_body(f"{prefix}{note}\n\n---\n\n{fresh.content}"))
        agent_message_id = self._await_agent_message(handle.conversation_id, handle.user_message_id)
        bridge = BridgeSession(handle.conversation_id, ops, str(ops.get("space_id", "")))
        state = _TurnState(
            conversation_id=handle.conversation_id,
            agent_message_id=agent_message_id,
            send_prompt=prompt,
            session=bridge,
        )
        self._turns[handle.conversation_id] = state
        return state

    def _conversation_read_path(self, conversation_id: str) -> str:
        """Conversation read that no cache may serve stale.

        Dust serves API GETs through Cloudflare: a first read taken before
        the agent row is visible can be answered from cache for every later
        poll, which then never sees the reply. A unique query parameter
        makes each poll a MISS.
        """
        self._reads += 1
        return f"assistant/conversations/{conversation_id}?_={self._reads:x}"

    def _await_agent_message(self, conversation_id: str, user_message_id: str) -> str:
        from hermes_dust.conversations import DustConversationGoneError

        gone_reads = 0
        empty_reads = 0

        def poll() -> tuple[str | None, list[str]]:
            nonlocal gone_reads, empty_reads
            try:
                conversation = self._dust_json(self._conversation_read_path(conversation_id))
            except DustConversationGoneError:
                # A just-created conversation may not be readable yet;
                # bounded by the attempts below, this is a retry, not a loss.
                gone_reads += 1
                return None, []
            if not isinstance(conversation, dict):
                empty_reads += 1
                return None, []
            found = find_agent_message_id(conversation, user_message_id)
            if found is not None:
                return found, []
            empty_reads += 1
            return None, find_unfinished_agent_message_ids(conversation)

        agent_message_id: str | None = None
        running: list[str] = []
        for index, delay in enumerate(AGENT_MESSAGE_DELAYS):
            if index:
                self.sleep(delay)
            agent_message_id, running = poll()
            if agent_message_id is not None:
                return agent_message_id
        if running:
            await_cancel = True
            try:
                self._dust_json(
                    f"assistant/conversations/{conversation_id}/cancel",
                    method="POST",
                    body={"action": "cancel", "messageIds": running},
                )
            except Exception:
                await_cancel = False
            if await_cancel:
                for _ in range(POST_CANCEL_ATTEMPTS):
                    self.sleep(AGENT_MESSAGE_DELAY_S)
                    agent_message_id, _ = poll()
                    if agent_message_id is not None:
                        return agent_message_id
        raise ValueError(
            "Dust accepted the message but created no agent reply to follow "
            f"(conversation {conversation_id}; {gone_reads} unreadable reads, "
            f"{empty_reads} reads without a reply)."
        )

    def _find_follow_up(self, results: list[tuple[str, str, bool]]) -> _TurnState | None:
        if not results:
            return None
        ids = {tool_call_id for tool_call_id, _, _ in results if tool_call_id}
        for turn in self._turns.values():
            if turn.session is None:
                continue
            if ids and ids <= turn.session.pending_tool_ids():
                return turn
        return None

    # -- streaming ---------------------------------------------------------------
    def _open_sse_response(self, path: str):
        """Open the event stream; tests override this to avoid the network."""
        import urllib.request

        from hermes_dust.http import dust_auth_headers, resolve_token

        token = resolve_token(self.config.access_token)
        request = urllib.request.Request(
            f"{self._workspace_base()}/{path}",
            headers={**dust_auth_headers(token), "Accept": "text/event-stream"},
            method="GET",
        )
        return urllib.request.urlopen(request, timeout=70)

    def _pump_records(self, turn: _TurnState, out: "queue.Queue[Any]", stop: threading.Event) -> None:
        """Read the agent event stream in the background, reopening from cursor."""
        reopens = 0
        while not stop.is_set():
            query = f"?lastEventId={turn.last_event_id}" if turn.last_event_id else ""
            path = (
                f"assistant/conversations/{turn.conversation_id}"
                f"/messages/{turn.agent_message_id}/events{query}"
            )
            try:
                response = self._open_sse_response(path)
            except Exception as exc:
                if reopens >= MAX_REOPENS:
                    out.put(("error", exc))
                    return
                reopens += 1
                stop.wait(REOPEN_DELAY_S)
                continue
            try:
                chunks: Iterator[bytes] = iter(lambda: response.read(65536), b"")
                for record in iter_sse_events(chunks):
                    if stop.is_set():
                        return
                    if record.data == "done":
                        continue
                    try:
                        envelope = json.loads(record.data)
                    except Exception:
                        continue
                    if isinstance(envelope, dict):
                        event_id = envelope.get("eventId")
                        event = envelope.get("data")
                        if isinstance(event, dict) and isinstance(event.get("type"), str):
                            out.put(("event", event, event_id if isinstance(event_id, str) else None))
                            reopens = 0
                out.put(("closed", None))
                return
            except Exception as exc:
                if reopens >= MAX_REOPENS:
                    out.put(("error", exc))
                    return
                reopens += 1
                stop.wait(REOPEN_DELAY_S)
            finally:
                try:
                    response.close()
                except Exception:
                    pass

    def _poll_once(self, session: BridgeSession, shell_tool: str | None) -> tuple[str, str, dict] | None:
        """One pod poll; returns the next (tool_call_id, name, args), if any."""
        try:
            session.poll()
        except Exception:
            return None
        nxt = session.next_tool_call(shell_tool)
        if nxt is None:
            return None
        tool_call_id, call = nxt
        return tool_call_id, call.tool, call.arguments

    def _collect_answer(self, turn: _TurnState, shell_tool: str | None) -> Iterator[dict[str, Any]]:
        """Yield OpenAI-style chunks for one Dust turn, bridging pod requests."""
        stripper = create_tag_stripper()
        events: queue.Queue[Any] = queue.Queue()
        stop = threading.Event()
        pump = threading.Thread(target=self._pump_records, args=(turn, events, stop), daemon=True)
        pump.start()
        assert turn.session is not None
        session = turn.session

        def text_chunk(delta: str) -> dict[str, Any]:
            if delta:
                turn.answer_parts.append(delta)
            return {"choices": [{"delta": {"content": delta}}]}

        def tool_chunk(tool_call_id: str, name: str, arguments: dict) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ]
                        }
                    }
                ]
            }

        def finish_chunk(reason: str) -> dict[str, Any]:
            return {"choices": [{"delta": {}, "finish_reason": reason}]}

        def bridged_round() -> Iterator[dict[str, Any]]:
            nxt = self._poll_once(session, shell_tool)
            if nxt is None:
                return
            tool_call_id, name, arguments = nxt
            yield tool_chunk(tool_call_id, name, arguments)
            tail = stripper.flush()
            if tail:
                yield text_chunk(tail)
            yield finish_chunk("tool_calls")

        try:
            while True:
                if self._cancel_event.is_set():
                    yield finish_chunk("other")
                    return
                try:
                    item = events.get(timeout=1.0)
                except queue.Empty:
                    pending = list(bridged_round())
                    if pending:
                        yield from pending
                        return
                    continue
                kind = item[0]
                if kind == "error":
                    raise RuntimeError(f"Dust event stream failed: {item[1]}")
                if kind == "closed":
                    break
                _, event, event_id = item
                if event_id:
                    turn.last_event_id = event_id
                etype = event.get("type")
                if etype == "generation_tokens":
                    raw = event.get("text")
                    if event.get("classification") == "chain_of_thought":
                        stripped = stripper.push(raw if isinstance(raw, str) else "")
                        if stripped:
                            yield {"choices": [{"delta": {"reasoning_content": stripped}}]}
                    else:
                        stripped = stripper.push(raw if isinstance(raw, str) else "")
                        if stripped:
                            yield text_chunk(stripped)
                    continue
                if etype == "tool_approve_execution":
                    action_id = event.get("actionId")
                    if isinstance(action_id, str):
                        try:
                            self._dust_json(
                                f"assistant/conversations/{turn.conversation_id}"
                                f"/messages/{turn.agent_message_id}/validate-action",
                                method="POST",
                                body={"actionId": action_id, "approved": "approved"},
                            )
                        except Exception:
                            pass
                    continue
                finish_reason = TERMINAL_TYPES.get(str(etype))
                if finish_reason is None:
                    continue
                if finish_reason == "error":
                    raw = event.get("error")
                    detail = raw.get("message") if isinstance(raw, dict) else None
                    raise RuntimeError(detail or f"Dust ended the turn with {etype}.")
                break

            # Terminal event, or the stream closed: grace polls for a request
            # left in the same breath as the turn's end.
            for _ in range(GRACE_POLLS):
                if self._cancel_event.is_set():
                    break
                pending = list(bridged_round())
                if pending:
                    yield from pending
                    return
                self.sleep(1.0)

            if not self._cancel_event.is_set():
                stored = read_agent_message(
                    self._dust_json(self._conversation_read_path(turn.conversation_id)),
                    turn.agent_message_id,
                )
                if stored is not None and stored.state == "succeeded" and not turn.answer_parts and stored.content:
                    yield text_chunk(stored.content)
            yield finish_chunk("stop" if not self._cancel_event.is_set() else "other")
        finally:
            stop.set()

    # -- facade --------------------------------------------------------------------
    def _prepare(self, messages: list[dict[str, Any]], model: str, extra: dict[str, Any]) -> tuple[_TurnState, str | None]:
        tools = _openai_tools(extra)
        shell_tool = _shell_tool_name(tools) or "terminal"
        results = _tool_results(messages)
        follow_up = self._find_follow_up(results)
        if follow_up is not None and follow_up.session is not None:
            for tool_call_id, output, is_error in results:
                follow_up.session.record_outcome(tool_call_id, output, is_error)
            return follow_up, shell_tool
        prompt = _host_messages_to_prompt(messages)
        return self._send(prompt, model, tools), shell_tool

    def complete_turn(self, *, messages: list[dict[str, Any]], model: str, extra: dict[str, Any]) -> dict[str, Any]:
        turn, shell_tool = self._prepare(messages, model, extra)
        round_texts: list[str] = []
        finish_reason = "stop"
        for chunk in self._collect_answer(turn, shell_tool):
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    round_texts.append(delta["content"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
        if finish_reason == "stop":
            remember_turn(self.threads, turn.send_prompt, "".join(turn.answer_parts), turn.conversation_id)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "".join(round_texts)},
                    "finish_reason": finish_reason,
                }
            ]
        }

    def stream_turn(
        self, *, messages: list[dict[str, Any]], model: str, extra: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        turn, shell_tool = self._prepare(messages, model, extra)
        finish_reason = "stop"
        for chunk in self._collect_answer(turn, shell_tool):
            for choice in chunk.get("choices", []):
                if (choice.get("delta") or {}).get("finish_reason") or choice.get("finish_reason"):
                    finish_reason = (choice.get("delta") or {}).get("finish_reason") or choice.get("finish_reason")
            yield chunk
        if finish_reason == "stop":
            remember_turn(self.threads, turn.send_prompt, "".join(turn.answer_parts), turn.conversation_id)
