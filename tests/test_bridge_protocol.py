"""Unit tests for the pod/sandbox exchange protocol — no network, stdlib only."""

import json
import unittest

from hermes_dust.bridge import protocol as p


class TestPaths(unittest.TestCase):
    def test_pod_name(self) -> None:
        self.assertEqual(p.pod_name_for("/home/u/myproj"), "hermes-myproj")
        self.assertEqual(p.pod_name_for(""), "hermes-project")

    def test_exchange_scoping(self) -> None:
        self.assertEqual(p.exchange_dir("conv1"), "hs-bash/conv1")
        self.assertEqual(p.detached_path("conv1", "step1"), "hs-bash/conv1/step1.detached")
        self.assertTrue(p.is_pod_bash_path("hs-bash/conv1/step1.json"))
        self.assertFalse(p.is_pod_bash_path("notes.md"))

    def test_helper_script_protocol(self) -> None:
        script = p.helper_script()
        self.assertIn("hs-bash", script)
        self.assertIn("/files/pod", script)
        self.assertIn(".detached", script)
        self.assertIn("--sh", script)
        self.assertTrue(script.startswith("#!/usr/bin/env bash"))


class TestPendingRequests(unittest.TestCase):
    def _file(self, path: str, ms: int) -> p.PodFile:
        return p.PodFile(path=path, modified_ms=ms)

    def test_oldest_first_and_both_routes(self) -> None:
        files = [
            self._file("hs-bash/b.json", 200),
            self._file("hs-bash/a.sh", 100),
            self._file("hs-bash/a.out", 300),
            self._file("notes.md", 50),
        ]
        pending = p.pending_requests(files, set())
        self.assertEqual([(r.id, r.kind) for r in pending], [("a", "script"), ("b", "tools")])

    def test_claimed_and_unsafe_dropped(self) -> None:
        files = [
            self._file("hs-bash/../evil.sh", 100),
            self._file("hs-bash/ok.sh", 100),
        ]
        pending = p.pending_requests(files, {"ok"})
        self.assertEqual(pending, [])


class TestToolRequestParsing(unittest.TestCase):
    def test_batch_and_aliases(self) -> None:
        ok, parsed = p.parse_tool_request(json.dumps({"calls": [{"name": "read_file", "args": {"path": "a"}}]}))
        self.assertTrue(ok)
        assert isinstance(parsed, p.PodToolRequest)
        self.assertEqual(parsed.calls[0].tool, "read_file")
        self.assertEqual(parsed.on_error, "stop")

    def test_bare_array_and_continue(self) -> None:
        ok, parsed = p.parse_tool_request(json.dumps([{"tool": "terminal", "arguments": {}}, {"tool": "x", "arguments": {}}]))
        self.assertTrue(ok)
        assert isinstance(parsed, p.PodToolRequest)
        self.assertEqual(len(parsed.calls), 2)

    def test_failures_are_messages(self) -> None:
        for body in ["{nope", json.dumps({"calls": []}), json.dumps([{"arguments": {}}])]:
            ok, parsed = p.parse_tool_request(body)
            self.assertFalse(ok)
            self.assertIsInstance(parsed, str)
        ok, _ = p.parse_tool_request(json.dumps({"calls": [{"tool": "t"}] * (p.MAX_BATCH_CALLS + 1)}))
        self.assertFalse(ok)

    def test_signatures(self) -> None:
        sigs = p.tool_signatures(
            [{"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
        )
        self.assertEqual(sigs, ["read_file(path)"])

    def test_call_labels(self) -> None:
        self.assertEqual(p.call_label("terminal", {"command": "npm test"}), "$ npm test")
        self.assertEqual(p.call_label("read_file", {"path": "a.ts"}), "read_file a.ts")


class TestFormatting(unittest.TestCase):
    def test_batch_result_stop(self) -> None:
        outcomes = [
            p.PodToolOutcome("read_file", {"path": "a"}, "hello", False, 100),
            p.PodToolOutcome("terminal", {"command": "false"}, "boom", True, 200),
        ]
        text = p.format_tool_batch_result(outcomes, 3)
        self.assertIn("status: failed", text)
        self.assertIn("=== call 1 read_file ok", text)
        self.assertIn("=== call 3 not run", text)

    def test_trim_keeps_ends(self) -> None:
        out = p.trim_to("x" * 1000, 100)
        self.assertIn("characters cut from the middle", out)
        self.assertLess(len(out), 500)

    def test_delivery_marked_automatic(self) -> None:
        text = p.format_delivery("batch", "hs-bash/c/step1.json", "status: ok")
        self.assertIn("[automatic message", text)


if __name__ == "__main__":
    unittest.main()
