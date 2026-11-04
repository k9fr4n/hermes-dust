"""Unit tests for provider pool handling — stubs only, no core, no network."""

import unittest
from unittest.mock import patch

from hermes_dust import provider as P


class StubEntry:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.expired = False


class StubPool:
    def __init__(self, entries: list) -> None:
        self._entries = entries
        self.selected = 0

    def entries(self):
        return list(self._entries)

    def select(self, **kwargs):
        self.selected += 1
        return self._entries[0] if self._entries else None


class FakeGrant:
    access_token = "new-access"
    refresh_token = "new-refresh"
    region = "europe-west1"
    expires_at = 9_999_999_999_000


def make_entry(**overrides):
    base = {
        "id": "abc123",
        "label": "dust:w1",
        "access_token": "old-access",
        "refresh_token": "old-refresh",
        "expires_at_ms": 9_999_999_999_000,
        "extra": {"region": "europe-west1", "workspace_id": "w1"},
    }
    base.update(overrides)
    return StubEntry(**base)


class TestPickEntry(unittest.TestCase):
    def test_returns_selected_entry_without_manual_refresh(self) -> None:
        pool = StubPool([make_entry()])
        with patch.object(P, "_pool", return_value=pool):
            self.assertEqual(P._pick_entry().access_token, "old-access")

    def test_raises_when_pool_empty(self) -> None:
        with patch.object(P, "_pool", return_value=StubPool([])):
            with self.assertRaises(RuntimeError):
                P._pick_entry()

    def test_no_expiry_field_means_no_crash(self) -> None:
        # expires_at_ms lives on the entry, never in extra: a missing field
        # must not read as "expired, rotate now".
        entry = make_entry()
        del entry.expires_at_ms
        pool = StubPool([entry])
        with patch.object(P, "_pool", return_value=pool):
            self.assertEqual(P._pick_entry().access_token, "old-access")


class TestRefreshCredential(unittest.TestCase):
    def test_flat_shape_no_nested_extra(self) -> None:
        with patch("hermes_dust.workos.refresh_grant", return_value=FakeGrant()):
            out = P.refresh_credential(make_entry())
        assert out is not None
        self.assertEqual(out["access_token"], "new-access")
        self.assertEqual(out["refresh_token"], "new-refresh")
        self.assertEqual(out["expires_at_ms"], 9_999_999_999_000)
        self.assertEqual(out["region"], "europe-west1")
        self.assertEqual(out["workspace_id"], "w1")
        self.assertNotIn("extra", out)

    def test_no_refresh_token_is_none(self) -> None:
        self.assertIsNone(P.refresh_credential(make_entry(refresh_token="")))


class TestRefreshedToken(unittest.TestCase):
    def test_plain_returns_current(self) -> None:
        self.assertEqual(P._refreshed_token(make_entry(), False), "old-access")

    def test_refresh_reselects_instead_of_posting(self) -> None:
        fresh = make_entry(access_token="live-access")
        with patch.object(P, "_pick_entry", return_value=fresh):
            self.assertEqual(P._refreshed_token(make_entry(), True), "live-access")


class TestAuthHandlerDeclines(unittest.TestCase):
    def test_logout_goes_to_generic_path(self) -> None:
        self.assertFalse(P.dust_auth_handler("logout", object()))

    def test_refresh_goes_to_generic_path(self) -> None:
        self.assertFalse(P.dust_auth_handler("refresh", object()))


if __name__ == "__main__":
    unittest.main()
