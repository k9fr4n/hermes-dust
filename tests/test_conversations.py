"""Unit tests for hermes_dust.conversations — no network, stdlib only."""

import unittest

from hermes_dust.conversations import (
    DustConversationGoneError,
    DustTurn,
    create_conversation,
    find_agent_message_id,
    find_agent_message_id_in,
    find_unfinished_agent_message_ids,
    message_payload,
    post_message,
    read_agent_message,
)
from hermes_dust.http import DustApiError


def _conversation() -> dict:
    return {
        "sId": "conv1",
        "content": [
            [{"type": "user_message", "sId": "u1"}],
            [
                {"type": "agent_message", "sId": "a0", "parentMessageId": "u0", "status": "succeeded"},
                {"type": "agent_message", "sId": "a1", "parentMessageId": "u1", "status": "created",
                 "content": "hello"},
            ],
        ],
    }


class TestConversations(unittest.TestCase):
    def test_payload_stamps_cli_origin(self) -> None:
        payload = message_payload(DustTurn(agent_s_id="dust", content="hi"))
        self.assertEqual(payload["context"]["origin"], "cli")
        self.assertEqual(payload["mentions"], [{"configurationId": "dust"}])
        self.assertNotIn("modelSelection", payload)

    def test_payload_model_selection(self) -> None:
        turn = DustTurn(
            agent_s_id="dust",
            content="hi",
            model_provider="anthropic",
            model_id="claude-sonnet-5",
            reasoning_effort="medium",
        )
        payload = message_payload(turn)
        self.assertEqual(
            payload["modelSelection"],
            {"providerId": "anthropic", "modelId": "claude-sonnet-5", "reasoningEffort": "medium"},
        )

    def test_find_agent_message_id(self) -> None:
        self.assertEqual(find_agent_message_id(_conversation(), "u1"), "a1")
        self.assertIsNone(find_agent_message_id(_conversation(), "missing"))

    def test_read_agent_message(self) -> None:
        message = read_agent_message(_conversation(), "a1")
        assert message is not None
        self.assertEqual(message.state, "running")
        self.assertEqual(message.content, "hello")

    def test_finished_states(self) -> None:
        for status, expected in [
            ("succeeded", "succeeded"),
            ("gracefully_stopped", "succeeded"),
            ("failed", "failed"),
            ("cancelled", "cancelled"),
            ("interrupted", "cancelled"),
        ]:
            conversation = {"content": [[{"type": "agent_message", "sId": "a", "status": status}]]}
            message = read_agent_message(conversation, "a")
            assert message is not None
            self.assertEqual(message.state, expected, status)

    def test_unfinished_ids(self) -> None:
        self.assertEqual(find_unfinished_agent_message_ids(_conversation()), ["a1"])

    def test_post_message_gone_becomes_conversation_gone(self) -> None:
        def gone_json(path: str, method: str = "GET", body: object = None) -> object:
            raise DustApiError(404, "gone", "not_found", "gone")

        with self.assertRaises(DustConversationGoneError):
            post_message(gone_json, "conv1", DustTurn(agent_s_id="dust", content="hi"))

    def test_create_conversation_returns_agent_message_id(self) -> None:
        def respond(path: str, method: str = "GET", body: object = None) -> object:
            return {
                "conversation": {
                    "sId": "conv1",
                    "content": [
                        [{"type": "user_message", "sId": "u1"}],
                        [{"type": "agent_message", "sId": "a1", "parentMessageId": "u1", "status": "created"}],
                    ],
                }
            }

        handle = create_conversation(respond, DustTurn(agent_s_id="dust", content="hi"))
        self.assertEqual(handle.conversation_id, "conv1")
        self.assertEqual(handle.user_message_id, "u1")
        self.assertEqual(handle.agent_message_id, "a1")

    def test_create_conversation_agent_message_id_none_when_absent(self) -> None:
        def respond(path: str, method: str = "GET", body: object = None) -> object:
            return {"conversation": {"sId": "conv1", "content": [[{"type": "user_message", "sId": "u1"}]]}}

        handle = create_conversation(respond, DustTurn(agent_s_id="dust", content="hi"))
        self.assertIsNone(handle.agent_message_id)

    def test_post_message_returns_agent_message_id(self) -> None:
        def respond(path: str, method: str = "GET", body: object = None) -> object:
            return {
                "message": {"sId": "u2"},
                "agentMessages": [{"type": "agent_message", "sId": "a2", "parentMessageId": "u2"}],
            }

        handle = post_message(respond, "conv1", DustTurn(agent_s_id="dust", content="hi"))
        self.assertEqual(handle.user_message_id, "u2")
        self.assertEqual(handle.agent_message_id, "a2")

    def test_find_agent_message_id_in_flat_list(self) -> None:
        flat = [
            {"type": "agent_message", "sId": "a2", "parentMessageId": "u2"},
        ]
        self.assertEqual(find_agent_message_id_in(flat, "u2"), "a2")
        self.assertIsNone(find_agent_message_id_in(flat, "u1"))


class TestHttpBytes(unittest.TestCase):
    def test_dust_bytes_reads_once(self) -> None:
        from hermes_dust.http import DustRequestContext, dust_bytes

        class Raw:
            status = 200

            def __init__(self) -> None:
                self.calls = 0

            def read(self, *args: object) -> bytes:
                self.calls += 1
                return b"plain-bytes" if self.calls == 1 else b""

        raw = Raw()
        context = DustRequestContext(
            base="https://x",
            access_token="t",
            fetch=lambda url, method="GET", headers=None, body=None, timeout_s=None: raw,
        )
        status, data = dust_bytes(context, "some/file.txt")
        self.assertEqual((status, data), (200, b"plain-bytes"))

    def test_dust_bytes_retries_401(self) -> None:
        from hermes_dust.http import DustRequestContext, dust_bytes

        seen: list[str] = []

        class Raw:
            def __init__(self, status: int, payload: bytes) -> None:
                self.status = status
                self._payload = payload

            def read(self, *args: object) -> bytes:
                return self._payload

        def fetch(url: str, method: str = "GET", headers=None, body=None, timeout_s=None):
            token = (headers or {}).get("Authorization", "")
            seen.append(token)
            if "first" in token:
                return Raw(401, b"no")
            return Raw(200, b"yes")

        context = DustRequestContext(
            base="https://x",
            access_token=lambda refresh: "first" if not refresh else "second",
            fetch=fetch,
        )
        self.assertEqual(dust_bytes(context, "f"), (200, b"yes"))
        self.assertEqual(seen, ["Bearer first", "Bearer second"])


if __name__ == "__main__":
    unittest.main()
