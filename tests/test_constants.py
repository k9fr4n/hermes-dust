"""Unit tests for hermes_dust.constants — no network, stdlib only."""

import unittest

from hermes_dust.constants import (
    DUST_DEFAULT_REGION,
    USER_MESSAGE_ORIGIN,
    dust_api_base,
    dust_host,
    dust_private_api_base,
)


class TestConstants(unittest.TestCase):
    def test_origin_is_cli(self) -> None:
        self.assertEqual(USER_MESSAGE_ORIGIN, "cli")

    def test_known_regions(self) -> None:
        self.assertEqual(dust_host("us-central1"), "https://dust.tt")
        self.assertEqual(dust_host("europe-west1"), "https://eu.dust.tt")

    def test_unknown_region_falls_back(self) -> None:
        self.assertEqual(dust_host("moon-south1"), dust_host(DUST_DEFAULT_REGION))

    def test_api_bases(self) -> None:
        self.assertEqual(
            dust_api_base("us-central1", "w123"),
            "https://dust.tt/api/v1/w/w123",
        )
        self.assertEqual(
            dust_private_api_base("europe-west1", "w123"),
            "https://eu.dust.tt/api/w/w123",
        )


if __name__ == "__main__":
    unittest.main()
