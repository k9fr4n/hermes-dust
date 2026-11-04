"""Unit tests for hermes_dust.sse — no network, stdlib only."""

import unittest

from hermes_dust.sse import iter_sse_events, parse_record


class TestParseRecord(unittest.TestCase):
    def test_joins_data_lines(self) -> None:
        event = parse_record("data: a\ndata: b")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.data, "a\nb")

    def test_skips_comments_and_reads_fields(self) -> None:
        event = parse_record(": ping\nid: 7\nevent: update\ndata: hi")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.id, "7")
        self.assertEqual(event.event, "update")
        self.assertEqual(event.data, "hi")

    def test_no_data_is_none(self) -> None:
        self.assertIsNone(parse_record("id: 1\n: keep-alive"))


class TestIterEvents(unittest.TestCase):
    def test_split_across_chunks(self) -> None:
        chunks = [b'data: {"a"', b": 1}\n\ndata: done"]
        events = list(iter_sse_events(chunks))
        self.assertEqual([e.data for e in events], ['{"a": 1}', "done"])

    def test_unterminated_trailing_yielded(self) -> None:
        events = list(iter_sse_events(['data: last']))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, "last")

    def test_crlf_normalized(self) -> None:
        events = list(iter_sse_events(["data: x\r\n\r\ndata: y\r\n\r\n"]))
        self.assertEqual([e.data for e in events], ["x", "y"])


if __name__ == "__main__":
    unittest.main()
