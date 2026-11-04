"""One conversation's bridge session: claim pod requests, emit Hermes tool calls.

Same mechanism as opencode-dust's ``BridgeManager`` + ``PodBashWatcher``,
driven by the Hermes client instead of opencode session events:

- While a Dust turn streams, the client polls the pod (``poll()``). A
  request the agent left is claimed exactly once — the claim lives in
  memory, because the listing and the delete both go through Cloudflare's
  cache and a taken request can reappear in the next listing.
- ``next_tool_call()`` hands out one call at a time, requests in order: a
  batch that edits a file and a script testing it are two steps of one plan.
- Hermes executes each emitted call in its own loop; the results come back
  in the next client call and ``record_outcome()`` writes the ``.out`` the
  sandbox helper waits on. A request always gets an answer: every path out
  of ``record_outcome()`` ends with a ``.out`` in the pod.
- When nobody is waiting (the helper gave up and left ``.detached``, or the
  turn already ended), the finished text is queued in ``deliveries`` for the
  next turn instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hermes_dust.bridge.protocol import (
    POD_BASH_DIR,
    PodFile,
    PodRequest,
    PodToolCall,
    PodToolOutcome,
    PodToolRequest,
    detached_path,
    exchange_dir,
    format_bash_result,
    format_delivery,
    format_tool_batch_result,
    parse_tool_request,
    pending_requests,
)


@dataclass
class RequestState:
    """One claimed request and its progress through Hermes' loop."""

    request: PodRequest
    body: str
    kind: str  # "script" | "tools"
    parsed: PodToolRequest | str  # batch, or the parse error to answer with
    tool_call_ids: list[str] = field(default_factory=list)
    outcomes: list[PodToolOutcome] = field(default_factory=list)
    emitted: int = 0  # calls handed out so far
    finished: bool = False
    emitted_at_ms: int = 0


class BridgeSession:
    """Exchange state for one Dust conversation."""

    def __init__(
        self,
        conversation_id: str,
        pod: dict[str, Any],
        space_id: str,
        now_ms: Any | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self._pod = pod
        self._space_id = space_id
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._scope = f"{exchange_dir(conversation_id)}/"
        self._claimed: set[str] = set()
        self._queue: list[PodRequest] = []
        self._requests: dict[str, RequestState] = {}
        self.deliveries: list[str] = []
        self.interrupted = False
        self._calls = 0

    # -- polling ---------------------------------------------------------
    def _scoped_list(self) -> list[PodFile]:
        files = self._pod["list"]()
        scoped: list[PodFile] = []
        for entry in files:
            path = entry.path if isinstance(entry, PodFile) else entry.get("path", "")
            if path.startswith(self._scope):
                flat = f"{POD_BASH_DIR}/{path[len(self._scope):]}"
                if isinstance(entry, PodFile):
                    scoped.append(PodFile(path=flat, size_bytes=entry.size_bytes, modified_ms=entry.modified_ms))
                else:
                    scoped.append(
                        PodFile(
                            path=flat,
                            size_bytes=int(entry.get("size_bytes", 0)),
                            modified_ms=int(entry.get("modified_ms", 0)),
                        )
                    )
        return scoped

    def _inward(self, flat_path: str) -> str:
        return f"{self._scope}{flat_path[len(POD_BASH_DIR) + 1:]}"

    def poll(self) -> int:
        """Claim newly listed requests; returns outstanding count."""
        try:
            files = self._scoped_list()
        except Exception:
            return self.outstanding
        results = {f.path for f in files}
        for request in pending_requests(files, self._claimed):
            self._claimed.add(request.id)
            self._queue.append(
                PodRequest(id=request.id, kind=request.kind, path=self._inward(request.path))
            )
        return self.outstanding

    @property
    def outstanding(self) -> int:
        return sum(1 for state in self._requests.values() if not state.finished) + len(self._queue)

    # -- emission ----------------------------------------------------------
    def _read_request(self, request: PodRequest) -> str | Exception:
        try:
            return self._pod["read"](request.path)
        except Exception as exc:
            return exc

    def _forget(self, path: str) -> None:
        try:
            self._pod["remove"](path)
        except Exception:
            pass

    def next_tool_call(self, shell_tool: str | None) -> tuple[str, PodToolCall] | None:
        """Next call to emit as a Hermes tool call, or None when quiet.

        One call at a time, requests in order. Parse failures and missing
        shell tools are answered in the pod at once (the agent needs the
        refusal to correct itself) rather than emitted. Returns None while
        every emitted call still waits for its outcome.
        """
        while True:
            state = next((s for s in self._requests.values() if not s.finished), None)
            if state is None:
                if not self._queue:
                    return None
                state = self._take(self._queue.pop(0))
                if state is None:
                    continue
            if isinstance(state.parsed, str):
                self._finish_with_text(state, state.parsed)
                continue
            if state.emitted > len(state.outcomes):
                # One call at a time: the batch keeps its order only if the
                # next call waits for the previous outcome. Round pacing is
                # the client's job too, but the session enforces it as well.
                return None
            if state.kind == "script":
                calls = [PodToolCall(tool="__shell__", arguments={"command": state.body})]
            else:
                assert isinstance(state.parsed, PodToolRequest)
                calls = list(state.parsed.calls)
            if state.emitted >= len(calls):
                # Everything handed out; outcomes arrive through record_outcome.
                return None
            call = calls[state.emitted]
            name = shell_tool if call.tool == "__shell__" else call.tool
            if name is None:
                self._finish_with_text(
                    state,
                    "status: failed\nseconds: 0.0\n---\n"
                    "This host offers no shell tool, so the script was not run.",
                )
                continue
            self._calls += 1
            tool_call_id = f"hermes-{state.request.id}-{state.emitted}-{self._calls}"
            state.tool_call_ids.append(tool_call_id)
            state.emitted += 1
            if state.emitted_at_ms == 0:
                state.emitted_at_ms = self._now_ms()
            return tool_call_id, PodToolCall(tool=name, arguments=dict(call.arguments))
        return None

    def _total_calls(self, state: RequestState) -> int:
        if isinstance(state.parsed, str):
            return 0
        if state.kind == "script":
            return 1
        return len(state.parsed.calls)

    def _take(self, request: PodRequest) -> RequestState | None:
        body = self._read_request(request)
        # Delete before running: a command can run for minutes and the poll
        # comes round every second; the in-memory claim covers this process,
        # the delete covers a restarted one.
        self._forget(request.path)
        if isinstance(body, Exception):
            text = (
                f"status: failed\nseconds: 0.0\n---\n"
                f"The request at {request.path} could not be read: {body}."
            )
            state = RequestState(request=request, body="", kind=request.kind, parsed=text)
            self._requests[request.id] = state
            return state
        if request.kind == "script":
            state = RequestState(
                request=request, body=body, kind="script",
                parsed=PodToolRequest(calls=(), on_error="stop"),
            )
            self._requests[request.id] = state
            return state
        ok, parsed = parse_tool_request(body)
        state = RequestState(
            request=request,
            body=body,
            kind="tools",
            parsed=parsed if ok else str(parsed),
        )
        self._requests[request.id] = state
        return state

    # -- outcomes ----------------------------------------------------------
    def pending_tool_ids(self) -> set[str]:
        ids: set[str] = set()
        for state in self._requests.values():
            if not state.finished:
                ids.update(state.tool_call_ids)
        return ids

    def record_outcome(
        self, tool_call_id: str, output: str, is_error: bool, duration_ms: int | None = None
    ) -> bool:
        """Record Hermes' result for one emitted call.

        Returns True when the call belonged to this session. When its
        request is complete, the ``.out`` is written at once.
        """
        for state in self._requests.values():
            if state.finished or tool_call_id not in state.tool_call_ids:
                continue
            index = state.tool_call_ids.index(tool_call_id)
            original = self._original_call(state, index)
            if duration_ms is None:
                duration_ms = max(0, self._now_ms() - state.emitted_at_ms)
            state.outcomes.append(
                PodToolOutcome(
                    tool=original.tool if original.tool != "__shell__" else "shell",
                    arguments=original.arguments if original.tool != "__shell__" else {},
                    output=output,
                    is_error=is_error,
                    duration_ms=duration_ms,
                )
            )
            self._maybe_finish(state)
            return True
        return False

    def _original_call(self, state: RequestState, index: int) -> PodToolCall:
        if state.kind == "script":
            return PodToolCall(tool="__shell__", arguments={"command": state.body})
        assert isinstance(state.parsed, PodToolRequest)
        return state.parsed.calls[index]

    def _maybe_finish(self, state: RequestState) -> None:
        total = self._total_calls(state)
        if state.kind == "tools" and isinstance(state.parsed, PodToolRequest):
            stop = state.parsed.on_error == "stop" and any(o.is_error for o in state.outcomes)
        else:
            stop = False
        if len(state.outcomes) < total and not stop:
            return
        asked = total
        if state.kind == "script":
            outcome = state.outcomes[0] if state.outcomes else PodToolOutcome("shell", {}, "", True)
            text = format_bash_result(outcome.output, outcome.is_error, outcome.duration_ms)
            what = "script"
        else:
            text = format_tool_batch_result(state.outcomes, asked)
            what = "batch"
        state.finished = True
        self._publish(state, what, text)

    def _finish_with_text(self, state: RequestState, text: str) -> None:
        state.finished = True
        what = "script" if state.kind == "script" else "batch"
        self._publish(state, what, text)

    def _publish(self, state: RequestState, what: str, text: str) -> None:
        result_path = f"{exchange_dir(self.conversation_id)}/{state.request.id}.out"
        try:
            self._pod["write"](result_path, text)
        except Exception:
            pass
        if self.interrupted:
            return
        try:
            detached = self._pod["read"](detached_path(self.conversation_id, state.request.id))
            has_detached = isinstance(detached, str)
        except Exception:
            has_detached = False
        if has_detached:
            request_path = f"{exchange_dir(self.conversation_id)}/{state.request.id}.{'sh' if what == 'script' else 'json'}"
            self.deliveries.append(format_delivery(what, request_path, text))

    def interrupt(self, reason: str = "The developer interrupted the session; this call was not completed.") -> None:
        """Kill the running work and send nothing after it."""
        self.interrupted = True
        self.deliveries.clear()
        for request in self._queue:
            try:
                self._pod["write"](
                    f"{exchange_dir(self.conversation_id)}/{request.id}.out",
                    f"status: failed\nseconds: 0.0\n---\n{reason}",
                )
            except Exception:
                pass
        self._queue.clear()
        for state in self._requests.values():
            if not state.finished:
                state.finished = True
                try:
                    self._pod["write"](
                        f"{exchange_dir(self.conversation_id)}/{state.request.id}.out",
                        f"status: failed\nseconds: 0.0\n---\n{reason}",
                    )
                except Exception:
                    pass
