"""Unit tests for hermes_dust.client message projection — no network."""

import unittest

from hermes_dust.client import _host_messages_to_prompt


class TestClientPrompt(unittest.TestCase):
    def test_tool_messages_dropped(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result"},
        ]
        prompt = _host_messages_to_prompt(messages)
        self.assertEqual([m["role"] for m in prompt], ["system", "user"])

    def test_unknown_model_rejected_at_send(self) -> None:
        from hermes_dust.client import DustClient, DustClientConfig
        from hermes_dust.history import DustThreadStore

        client = DustClient(config=DustClientConfig(access_token="t", region="us-central1", workspace_id="w"))
        client.threads = DustThreadStore()
        with self.assertRaises(ValueError):
            client._send([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "agent-default", [])

    def test_split_model_variant(self) -> None:
        from hermes_dust.client import split_model_variant

        self.assertEqual(split_model_variant("anthropic:claude-x+high"), ("anthropic:claude-x", "high"))
        self.assertEqual(split_model_variant("anthropic:claude-x"), ("anthropic:claude-x", None))
        # Unknown suffixes stay in the id so Dust reports the typo visibly.
        self.assertEqual(split_model_variant("anthropic:claude-x+bogus"), ("anthropic:claude-x+bogus", None))


if __name__ == "__main__":
    unittest.main()
