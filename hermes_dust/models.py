"""Workspace model catalog and the codec between Dust and Hermes model ids.

Ported from opencode-dust ``src/dust/models.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Model id meaning "whatever the agent is configured with". Sending any
# modelSelection override requires the workspace's "models_picker" feature;
# without it Dust answers HTTP 403. This id sends no override, so it is the
# one model that always works.
AGENT_DEFAULT_MODEL_ID = "agent-default"

# Reasoning efforts Dust defines, weakest first.
REASONING_EFFORTS = ("none", "light", "medium", "high")


def is_reasoning_effort(value: str) -> bool:
    """Whether Dust defines this reasoning effort."""
    return value in REASONING_EFFORTS


def encode_model_id(provider_id: str, model_id: str) -> str:
    """Encode a Dust provider/model pair as one Hermes model id."""
    return f"{provider_id}:{model_id}"


def decode_model_id(model_id: str) -> tuple[str, str] | None:
    """Decode a Hermes model id back into a Dust model override.

    An id naming no provider — including the agent default — decodes to no
    override, so the agent runs its configured model. Guessing a provider
    would produce a request Dust rejects.
    """
    separator = model_id.find(":")
    if separator <= 0 or separator == len(model_id) - 1:
        return None
    return (model_id[:separator], model_id[separator + 1 :])


@dataclass(frozen=True)
class DustModel:
    """A model offered on the Dust route."""

    id: str
    display_name: str
    context_window: int
    max_output_tokens: int | None = None
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    supports_vision: bool = True


# Always-available entry, offered whether or not the catalog can be read —
# including before anyone has logged in. Dust does not report the agent's
# configured model, so its window is a conservative figure keeping the host
# from over-filling a request it cannot measure.
AGENT_DEFAULT_MODEL = DustModel(
    id=AGENT_DEFAULT_MODEL_ID,
    display_name="Agent default",
    context_window=100_000,
    reasoning_efforts=(),
    supports_vision=True,
)


def parse_catalog(body: Any) -> list[DustModel]:
    """Project the private catalog route's answer onto Dust models."""
    entries = body.get("models", []) if isinstance(body, dict) else []
    models = [AGENT_DEFAULT_MODEL]
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("isSelectable") is not True:
            continue
        supported = entry.get("supportedReasoningEfforts") or {}
        efforts = tuple(e for e in REASONING_EFFORTS if supported.get(e) is True)
        fallback = entry.get("defaultReasoningEffort")
        provider_id = str(entry.get("providerId") or "")
        model_id = str(entry.get("modelId") or "")
        context_size = entry.get("contextSize")
        models.append(
            DustModel(
                id=encode_model_id(provider_id, model_id),
                display_name=str(entry.get("displayName") or model_id),
                context_window=int(context_size)
                if isinstance(context_size, int) and context_size > 0
                else AGENT_DEFAULT_MODEL.context_window,
                max_output_tokens=entry.get("generationTokensCount"),
                reasoning_efforts=efforts,
                default_reasoning_effort=fallback if is_reasoning_effort(str(fallback)) else None,
                supports_vision=entry.get("supportsVision") is not False,
            )
        )
    return models


def fetch_models(request_json: Any) -> list[DustModel]:
    """Read the models a workspace may select.

    ``request_json`` performs the ``models`` route and returns its decoded
    body. A catalog that cannot be read is not an error: the adapter contract
    forbids turning catalog absence into a rejected request, and the agent
    default works without it.
    """
    try:
        return parse_catalog(request_json())
    except Exception:
        return [AGENT_DEFAULT_MODEL]


def hermes_model_capabilities(models: list[DustModel]) -> dict[str, dict[str, Any]]:
    """Per-model capability declarations in the Hermes model_overrides schema.

    Dust agents run their own tool loop server-side and cannot call Hermes
    tools directly (phase 1: text only, bridge roadmap in README), so
    supports_tools is False. Images travel as content fragments.
    """
    capabilities: dict[str, dict[str, Any]] = {}
    for model in models:
        if model.id == AGENT_DEFAULT_MODEL_ID:
            continue
        capabilities[model.id] = {
            "supports_reasoning": bool(model.reasoning_efforts),
            "supports_vision": model.supports_vision,
            "supports_tools": False,
            "context_window": model.context_window,
        }
    return capabilities


def fallback_model_ids(models: list[DustModel], limit: int = 12) -> list[str]:
    """Curated picker list when the live catalog fetch fails."""
    ids = [m.id for m in models if m.id != AGENT_DEFAULT_MODEL_ID]
    return ids[:limit] if limit > 0 else ids
