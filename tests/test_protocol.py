"""Unit tests for workos, tag_stripper, bridge_note — no network, stdlib only."""

import base64
import json
import unittest
from unittest.mock import patch

from hermes_dust import bridge_note, workos
from hermes_dust.bridge.pod import parse_workspaces
from hermes_dust.tag_stripper import create_tag_stripper


def _jwt(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{raw}.sig"


class TestWorkos(unittest.TestCase):
    def test_decode_region(self) -> None:
        token = _jwt({"https://dust.tt/region": "europe-west1"})
        self.assertEqual(workos.decode_region(token), "europe-west1")
        self.assertEqual(workos.decode_region("opaque"), "us-central1")

    def test_grant_expiry(self) -> None:
        token = _jwt({"exp": 1700000000})
        self.assertEqual(workos.grant_expires_at(token), 1700000000000)
        self.assertTrue(workos.is_grant_expired(1700000000000, 1700000000000 - 59_000))
        self.assertFalse(workos.is_grant_expired(1700000000000, 1700000000000 - 61_000))

    def test_device_flow(self) -> None:
        calls = {"n": 0}

        def post(url: str, fields: dict) -> tuple[int, dict]:
            if "authorize" in url:
                return 200, {
                    "device_code": "d",
                    "user_code": "u",
                    "verification_uri": "https://x",
                    "verification_uri_complete": "https://x/u",
                    "expires_in": 900,
                    "interval": 0,
                }
            calls["n"] += 1
            if calls["n"] == 1:
                return 400, {"error": "authorization_pending"}
            return 200, {"access_token": _jwt({"exp": 9999999999}), "refresh_token": "r"}

        device = workos.request_device_code(post_form=post)
        self.assertEqual(device.user_code, "u")
        grant = workos.poll_for_grant(device, post_form=post, sleep=lambda s: None)
        self.assertEqual(grant.refresh_token, "r")

    def test_refresh_rejected_requires_reauth(self) -> None:
        def post(url: str, fields: dict) -> tuple[int, dict]:
            return 400, {"error": "invalid_grant"}

        with self.assertRaises(workos.WorkosAuthError) as ctx:
            workos.refresh_grant("stale", post_form=post)
        self.assertTrue(ctx.exception.requires_reauthentication)


def _me_body(*workspaces: dict) -> dict:
    return {"user": {"workspaces": workspaces}}


class TestWorkspaceParsing(unittest.TestCase):
    def test_parse_me_body(self) -> None:
        parsed = parse_workspaces(_me_body({"sId": "w1", "name": "A", "role": "admin"}, {"name": "no-id"}))
        self.assertEqual([(w.s_id, w.name, w.role) for w in parsed], [("w1", "A", "admin")])

    def test_parse_empty(self) -> None:
        self.assertEqual(parse_workspaces({}), [])
        self.assertEqual(parse_workspaces({"workspaces": [{"sId": "w1"}]}), [])

    def test_choose_single_workspace(self) -> None:
        from hermes_dust.provider import _choose_workspace

        with patch("hermes_dust.http.dust_json", return_value=_me_body({"sId": "w1", "name": "A"})):
            self.assertEqual(_choose_workspace("token", "us-central1"), "w1")

    def test_choose_several_lists_them(self) -> None:
        from hermes_dust.provider import _choose_workspace

        body = _me_body({"sId": "w1", "name": "A"}, {"sId": "w2", "name": "B"})
        with patch("hermes_dust.http.dust_json", return_value=body):
            with self.assertRaises(RuntimeError) as ctx:
                _choose_workspace("token", "us-central1")
        self.assertIn("w1", str(ctx.exception))
        self.assertIn("--workspace", str(ctx.exception))

    def test_choose_none(self) -> None:
        from hermes_dust.provider import _choose_workspace

        with patch("hermes_dust.http.dust_json", return_value=_me_body()):
            with self.assertRaises(RuntimeError) as ctx:
                _choose_workspace("token", "us-central1")
        self.assertIn("no workspace", str(ctx.exception))


class TestTagStripper(unittest.TestCase):
    def test_tags_removed_across_chunks(self) -> None:
        stripper = create_tag_stripper()
        out = stripper.push("a<think") + stripper.push("ing>b</") + stripper.push("response>c")
        self.assertEqual(out + stripper.flush(), "abc")

    def test_plain_angle_bracket_kept(self) -> None:
        stripper = create_tag_stripper()
        self.assertEqual(stripper.push("a < b") + stripper.flush(), "a < b")


class TestBridgeNote(unittest.TestCase):
    def test_hermes_note(self) -> None:
        note = bridge_note.hermes_bridge_note()
        self.assertIn("hs-tools/hs.sh", note)
        self.assertIn("read_file", note)
        self.assertNotIn("opencode", note)


if __name__ == "__main__":
    unittest.main()
