"""Unit tests for hermes_dust.conversations — no network, stdlib only."""

import unittest

from hermes_dust.conversations import (
    DustConversationGoneError,
    DustTurn,
    find_agent_message_id,
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


if __name__ == "__main__":
    unittest.main()
