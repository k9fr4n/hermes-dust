"""Unit tests for hermes_dust.models — no network, stdlib only."""

import unittest

from hermes_dust.models import (
    AGENT_DEFAULT_MODEL,
    AGENT_DEFAULT_MODEL_ID,
    decode_model_id,
    encode_model_id,
    fallback_model_ids,
    fetch_models,
    hermes_model_capabilities,
    is_reasoning_effort,
    parse_catalog,
)


def _catalog_body() -> dict:
    return {
        "models": [
            {
                "providerId": "anthropic",
                "modelId": "claude-sonnet-5",
                "displayName": "Claude Sonnet 5",
                "contextSize": 250000,
                "isSelectable": True,
                "supportsVision": True,
                "supportedReasoningEfforts": {"light": True, "medium": True},
                "defaultReasoningEffort": "medium",
            },
            {
                "providerId": "openai",
                "modelId": "gpt-x",
                "isSelectable": False,
            },
        ]
    }


class TestCodec(unittest.TestCase):
    def test_round_trip(self) -> None:
        self.assertEqual(encode_model_id("anthropic", "claude-sonnet-5"), "anthropic:claude-sonnet-5")
        self.assertEqual(decode_model_id("anthropic:claude-sonnet-5"), ("anthropic", "claude-sonnet-5"))

    def test_agent_default_has_no_override(self) -> None:
        self.assertIsNone(decode_model_id(AGENT_DEFAULT_MODEL_ID))
        self.assertIsNone(decode_model_id("nocolon"))
        self.assertIsNone(decode_model_id(":empty-provider"))
        self.assertIsNone(decode_model_id("empty-model:"))

    def test_reasoning_efforts(self) -> None:
        self.assertTrue(is_reasoning_effort("high"))
        self.assertFalse(is_reasoning_effort("ultra"))


class TestCatalog(unittest.TestCase):
    def test_parse_catalog(self) -> None:
        models = parse_catalog(_catalog_body())
        self.assertEqual(models[0].id, AGENT_DEFAULT_MODEL_ID)
        self.assertEqual(len(models), 2)
        sonnet = models[1]
        self.assertEqual(sonnet.id, "anthropic:claude-sonnet-5")
        self.assertEqual(sonnet.context_window, 250000)
        self.assertEqual(sonnet.reasoning_efforts, ("light", "medium"))
        self.assertEqual(sonnet.default_reasoning_effort, "medium")
        self.assertTrue(sonnet.supports_vision)

    def test_fetch_models_falls_back(self) -> None:
        def boom() -> dict:
            raise ConnectionError("offline")

        self.assertEqual(fetch_models(boom), [AGENT_DEFAULT_MODEL])

    def test_capabilities_disable_tools(self) -> None:
        models = parse_catalog(_catalog_body())
        caps = hermes_model_capabilities(models)
        self.assertIn("anthropic:claude-sonnet-5", caps)
        self.assertFalse(caps["anthropic:claude-sonnet-5"]["supports_tools"])
        self.assertTrue(caps["anthropic:claude-sonnet-5"]["supports_vision"])
        self.assertNotIn(AGENT_DEFAULT_MODEL_ID, caps)

    def test_fallback_ids(self) -> None:
        models = parse_catalog(_catalog_body())
        self.assertEqual(fallback_model_ids(models), ["anthropic:claude-sonnet-5"])


if __name__ == "__main__":
    unittest.main()
