"""Unit tests for the client pod-bridge round-trip — fakes only, no network."""

import io
import json
import unittest

from hermes_dust.bridge.session import BridgeSession
from hermes_dust.client import DustClient, DustClientConfig
from hermes_dust.history import plan_turn
from tests.test_bridge_session import FakePod


def sse_envelope(event_id: str, event: dict) -> str:
    return json.dumps({"eventId": event_id, "data": event})


def sse_stream(*records: str) -> bytes:
    return "".join(f"data: {record}\n\n" for record in records).encode()


def agent_conversation(state: str = "created", content: str = "") -> dict:
    return {
        "sId": "conv1",
        "content": [
            [{"type": "user_message", "sId": "u1"}],
            [
                {
                    "type": "agent_message",
                    "sId": "a1",
                    "parentMessageId": "u1",
                    "status": state,
                    "content": content,
                }
            ],
        ],
    }


def conversation_created() -> dict:
    return {"conversation": {"sId": "conv1", "content": [[{"type": "user_message", "sId": "u1"}]]}}


BRIDGE_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
]


class FakeBridgeClient(DustClient):
    """DustClient with the network and pod replaced by scripted fakes."""

    def __init__(self, pod: FakePod, batches: list[bytes]) -> None:
        super().__init__(
            config=DustClientConfig(access_token="token", region="us-central1", workspace_id="w1"),
        )
        self.sleep = lambda seconds: None
        self._fake_pod = pod
        self._batches = list(batches)

    def _dust_json(self, path: str, method: str = "GET", body=None):  # type: ignore[override]
        if path == "assistant/conversations" and method == "POST":
            return conversation_created()
        if path == "assistant/conversations/conv1":
            return agent_conversation()
        if path.endswith("validate-action"):
            return {}
        if path.endswith("/cancel"):
            return {}
        raise AssertionError(f"unexpected route {method} {path}")

    def _pod_ops(self):  # type: ignore[override]
        return {**self._fake_pod.ops(), "space_id": "space1"}

    def _open_sse_response(self, path):  # type: ignore[override]
        return io.BytesIO(self._batches.pop(0))


def first_turn_batches() -> list[bytes]:
    return [
        sse_stream(
            sse_envelope("e1", {"type": "generation_tokens", "text": "working ", "classification": "text"}),
            sse_envelope("e2", {"type": "agent_message_success"}),
        )
    ]


def resume_batches() -> list[bytes]:
    return [
        sse_stream(
            sse_envelope("e3", {"type": "generation_tokens", "text": "done", "classification": "text"}),
            sse_envelope("e4", {"type": "agent_message_success"}),
        )
    ]


class TestClientBridge(unittest.TestCase):
    def test_full_round_trip(self) -> None:
        pod = FakePod(
            {
                "hs-bash/conv1/step1.json": (
                    json.dumps({"calls": [{"tool": "read_file", "arguments": {"path": "a.ts"}}]}),
                    1,
                )
            }
        )
        client = FakeBridgeClient(pod, first_turn_batches())
        result = client.complete_turn(
            messages=[{"role": "user", "content": "hi"}],
            model="dust/anthropic:claude-x",
            extra={"tools": BRIDGE_TOOLS},
        )
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")

        turn = client._turns["conv1"]
        assert turn.session is not None
        pending = turn.session.pending_tool_ids()
        self.assertEqual(len(pending), 1)
        tool_call_id = next(iter(pending))
        # The request file is taken at claim; the answer is not written yet.
        self.assertNotIn("hs-bash/conv1/step1.json", pod.files)
        self.assertNotIn("hs-bash/conv1/step1.out", pod.files)

        client._batches.extend(resume_batches())
        resumed = client.complete_turn(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "working "},
                {"role": "tool", "tool_call_id": tool_call_id, "content": "file contents"},
            ],
            model="dust/anthropic:claude-x",
            extra={"tools": BRIDGE_TOOLS},
        )
        self.assertEqual(resumed["choices"][0]["finish_reason"], "stop")
        self.assertEqual(resumed["choices"][0]["message"]["content"], "done")

        out = pod.files.get("hs-bash/conv1/step1.out")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn("status: ok", out[0])
        self.assertIn("file contents", out[0])

        # The remembered digest routes the next user message to the same conversation.
        plan = plan_turn(
            client.threads,
            [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "working done"}]},
                {"role": "user", "content": [{"type": "text", "text": "more"}]},
            ],
        )
        self.assertEqual(plan.kind, "append")
        self.assertEqual(plan.conversation_id, "conv1")

    def test_unknown_model_rejected(self) -> None:
        pod = FakePod()
        client = FakeBridgeClient(pod, [])
        with self.assertRaises(ValueError):
            client._send([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "agent-default", [])


if __name__ == "__main__":
    unittest.main()
