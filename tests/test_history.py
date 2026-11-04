"""Unit tests for hermes_dust.history — no network, stdlib only."""

import unittest

from hermes_dust.history import (
    DustThreadStore,
    digest_history,
    plan_turn,
    remember_turn,
    render_all,
)


def _text(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


class TestHistory(unittest.TestCase):
    def test_render_drops_system_and_merges_assistant(self) -> None:
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            _text("user", "hello"),
            _text("assistant", "a"),
            _text("assistant", "b"),
        ]
        rendered = render_all(prompt)
        self.assertEqual([(m.role, m.text) for m in rendered], [("user", "hello"), ("assistant", "a\nb")])

    def test_digest_whitespace_blind_and_length_prefixed(self) -> None:
        from hermes_dust.history import RenderedMessage

        a = [RenderedMessage(role="user", text="a  b")]
        b = [RenderedMessage(role="user", text="a b")]
        self.assertEqual(digest_history(a), digest_history(b))
        c = [RenderedMessage(role="user", text="ab"), RenderedMessage(role="user", text="")]
        self.assertNotEqual(digest_history(a), digest_history(c))

    def test_plan_create_then_append(self) -> None:
        threads = DustThreadStore()
        prompt = [_text("user", "hello")]
        plan = plan_turn(threads, prompt)
        self.assertEqual(plan.kind, "create")
        remember_turn(threads, prompt, "hi", "conv1")
        plan2 = plan_turn(threads, [*prompt, _text("assistant", "hi"), _text("user", "more")])
        self.assertEqual(plan2.kind, "append")
        self.assertEqual(plan2.conversation_id, "conv1")
        self.assertEqual(plan2.content, "more")

    def test_rewritten_history_misses(self) -> None:
        threads = DustThreadStore()
        prompt = [_text("user", "hello")]
        remember_turn(threads, prompt, "hi", "conv1")
        plan = plan_turn(threads, [_text("user", "changed"), _text("assistant", "hi"), _text("user", "more")])
        self.assertEqual(plan.kind, "create")

    def test_store_bounded(self) -> None:
        threads = DustThreadStore(capacity=2)
        threads.set("a", "1")
        threads.set("b", "2")
        threads.set("c", "3")
        self.assertIsNone(threads.get("a"))
        self.assertEqual(threads.get("b"), "2")


if __name__ == "__main__":
    unittest.main()
