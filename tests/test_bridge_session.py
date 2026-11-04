"""Unit tests for BridgeSession — in-memory pod, no network, stdlib only."""

import json
import unittest

from hermes_dust.bridge.protocol import PodFile
from hermes_dust.bridge.session import BridgeSession


class FakePod:
    """In-memory pod file operations."""

    def __init__(self, files: dict[str, tuple[str, int]] | None = None) -> None:
        self.files: dict[str, tuple[str, int]] = dict(files or {})
        self.clock = 0

    def list(self) -> list[PodFile]:
        return [PodFile(path=path, modified_ms=ms) for path, (_content, ms) in self.files.items()]

    def read(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path][0]

    def write(self, path: str, contents: str) -> None:
        self.clock += 1
        self.files[path] = (contents, self.clock)

    def remove(self, path: str) -> None:
        self.files.pop(path, None)

    def ops(self) -> dict:
        return {"list": self.list, "read": self.read, "write": self.write, "remove": self.remove}


def make_session(pod: FakePod, conversation_id: str = "conv1") -> BridgeSession:
    return BridgeSession(conversation_id, pod.ops(), "space1")


class TestBridgeSession(unittest.TestCase):
    def test_json_batch_emitted_one_at_a_time(self) -> None:
        pod = FakePod({"hs-bash/conv1/step1.json": (json.dumps({"calls": [{"tool": "read_file", "arguments": {"path": "a"}}, {"tool": "terminal", "arguments": {"command": "ls"}}]}), 1)})
        session = make_session(pod)
        self.assertEqual(session.poll(), 1)
        # The request is only taken (read + cleared) when its first call is
        # handed out, exactly like the opencode watcher's claim/drain split.
        self.assertIn("hs-bash/conv1/step1.json", pod.files)

        first = session.next_tool_call("terminal")
        assert first is not None
        # Taken away at claim time: no second poll can run it again.
        self.assertNotIn("hs-bash/conv1/step1.json", pod.files)
        tool_call_id, call = first
        self.assertEqual(call.tool, "read_file")
        # Second call waits for the first outcome.
        self.assertIsNone(session.next_tool_call("terminal"))

        session.record_outcome(tool_call_id, "hello", False, duration_ms=10)
        second = session.next_tool_call("terminal")
        assert second is not None
        self.assertEqual(second[1].tool, "terminal")
        self.assertIsNone(session.next_tool_call("terminal"))

        session.record_outcome(second[0], "ok", False, duration_ms=5)
        out = pod.files.get("hs-bash/conv1/step1.out")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn("status: ok", out[0])
        self.assertIn("=== call 1 read_file ok", out[0])
        self.assertEqual(session.outstanding, 0)

    def test_stop_on_first_failure(self) -> None:
        pod = FakePod({"hs-bash/conv1/step1.json": (json.dumps({"calls": [{"tool": "a", "arguments": {}}, {"tool": "b", "arguments": {}}]}), 1)})
        session = make_session(pod)
        session.poll()
        first = session.next_tool_call("terminal")
        assert first is not None
        session.record_outcome(first[0], "boom", True, duration_ms=1)
        out = pod.files.get("hs-bash/conv1/step1.out")
        assert out is not None
        self.assertIn("status: failed", out[0])
        self.assertIn("not run", out[0])
        self.assertIsNone(session.next_tool_call("terminal"))

    def test_shell_script_uses_shell_tool(self) -> None:
        pod = FakePod({"hs-bash/conv1/step1.sh": ("npm test", 1)})
        session = make_session(pod)
        session.poll()
        nxt = session.next_tool_call("terminal")
        assert nxt is not None
        tool_call_id, call = nxt
        self.assertEqual(call.tool, "terminal")
        self.assertEqual(call.arguments, {"command": "npm test"})
        session.record_outcome(tool_call_id, "pass", False, duration_ms=3)
        out = pod.files.get("hs-bash/conv1/step1.out")
        assert out is not None
        self.assertTrue(out[0].startswith("status: ok"))

    def test_parse_error_answered_not_emitted(self) -> None:
        pod = FakePod({"hs-bash/conv1/step1.json": ("{nope", 1)})
        session = make_session(pod)
        session.poll()
        self.assertIsNone(session.next_tool_call("terminal"))
        out = pod.files.get("hs-bash/conv1/step1.out")
        assert out is not None
        self.assertIn("not valid JSON", out[0])

    def test_detached_queues_delivery(self) -> None:
        pod = FakePod(
            {
                "hs-bash/conv1/step1.json": (json.dumps({"calls": [{"tool": "a", "arguments": {}}]}), 1),
                "hs-bash/conv1/step1.detached": ("", 2),
            }
        )
        session = make_session(pod)
        session.poll()
        nxt = session.next_tool_call("terminal")
        assert nxt is not None
        session.record_outcome(nxt[0], "late", False, duration_ms=1)
        self.assertEqual(len(session.deliveries), 1)
        self.assertIn("[automatic message", session.deliveries[0])

    def test_interrupt_writes_cancelled_and_clears(self) -> None:
        pod = FakePod({"hs-bash/conv1/step1.json": (json.dumps({"calls": [{"tool": "a", "arguments": {}}]}), 1)})
        session = make_session(pod)
        session.poll()
        session.interrupt()
        out = pod.files.get("hs-bash/conv1/step1.out")
        assert out is not None
        self.assertIn("interrupted", out[0])
        self.assertEqual(session.deliveries, [])

    def test_claimed_once(self) -> None:
        pod = FakePod({"hs-bash/conv1/step1.json": (json.dumps({"calls": [{"tool": "a", "arguments": {}}]}), 1)})
        session = make_session(pod)
        session.poll()
        # Re-listing the taken file changes nothing: the claim is in memory.
        pod.files["hs-bash/conv1/step1.json"] = (json.dumps({"calls": [{"tool": "a", "arguments": {}}]}), 9)
        self.assertEqual(session.poll(), 1)


if __name__ == "__main__":
    unittest.main()
