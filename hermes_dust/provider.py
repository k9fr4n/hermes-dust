"""Hermes provider profile for Dust workspace agents.

Hermes keeps its own tool loop; Dust is never sent Hermes tools. The
bridge note (``hermes_bridge_note``) goes in front of every turn, and while
the turn streams the client polls the exchange pod: each request the agent
leaves there is emitted as an OpenAI-style tool call, Hermes executes it
with its own approvals, and the results go back through the pod ``.out``
files. The agent's own server-side tool approvals are answered automatically.

Auth is provider-owned (``auth_type = "dust_workos"``): the WorkOS device
grant from ``hermes_dust.workos`` lands in the credential pool, one row per
workspace. No API-key path exists on purpose — Dust bills API-key turns as
programmatic usage regardless of origin.
"""

from __future__ import annotations

import logging
from typing import Any

from hermes_dust.constants import DUST_DEFAULT_REGION, dust_private_api_base

logger = logging.getLogger(__name__)

PROVIDER_NAME = "dust"
PROVIDER_ALIASES = ("dust-ai",)
DISPLAY_NAME = "Dust"
DESCRIPTION = "Dust workspace agents (dust.tt) — WorkOS user grant, billed as interactive user usage."
SIGNUP_URL = "https://dust.tt"

# Curated picker list when the live catalog fetch fails. Entries are Dust
# modelSelection ids (providerId:modelId); the agent default needs no
# override and always works.
FALLBACK_MODELS = (
    "anthropic:claude-sonnet-5",
    "anthropic:claude-opus-5",
    "openai:gpt-5.6-sol",
    "google_ai_studio:gemini-3.1-pro-preview",
)


def _pool(provider: str = PROVIDER_NAME) -> Any:
    from agent.credential_pool import load_pool

    return load_pool(provider)


def _pick_entry() -> Any:
    """Return the live pooled credential, refreshing it when due."""
    from hermes_dust.workos import is_grant_expired
    import time

    pool = _pool()
    entries = [e for e in pool.entries() if not getattr(e, "expired", False)]
    if not entries:
        raise RuntimeError("Not logged in to Dust. Run `hermes auth add dust` first.")
    entry = entries[0]
    extra = getattr(entry, "extra", {}) or {}
    expires_at = int(extra.get("expires_at_ms") or 0)
    if expires_at and is_grant_expired(expires_at, int(time.time() * 1000)):
        refreshed = refresh_credential(entry)
        if refreshed:
            pool.update_entry(getattr(entry, "id", ""), **refreshed)
            entries = [e for e in pool.entries() if not getattr(e, "expired", False)]
            entry = entries[0] if entries else entry
    return entry


def _entry_context(entry: Any) -> tuple[str, str, str]:
    """Return (access_token, region, workspace_id) for a pool entry."""
    access_token = getattr(entry, "access_token", "")
    extra = getattr(entry, "extra", {}) or {}
    region = extra.get("region") or DUST_DEFAULT_REGION
    workspace_id = extra.get("workspace_id") or ""
    return access_token, region, workspace_id


def dust_auth_handler(action: str, args: Any) -> bool:
    """Provider-owned ``hermes auth add|status|logout|refresh dust``."""
    if action == "add":
        import uuid

        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential

        from hermes_dust.workos import poll_for_grant, request_device_code

        force = bool(getattr(args, "force", False))
        wanted_workspace = getattr(args, "workspace", None) or getattr(args, "workspace_id", None)
        pool = _pool()
        if not force and pool.entries():
            print("dust: already logged in. Use --force to replace the grant.")
            return True
        device = request_device_code()
        print("Open this page in your browser and approve the code:")
        print(f"  {device.verification_uri_complete}")
        print(f"User code: {device.user_code}")
        grant = poll_for_grant(device)
        workspace_id = wanted_workspace or _choose_workspace(grant.access_token, grant.region)
        pool.add_entry(
            PooledCredential(
                provider=PROVIDER_NAME,
                id=uuid.uuid4().hex[:6],
                label=f"dust:{workspace_id}",
                auth_type=AUTH_TYPE_OAUTH,
                priority=0,
                source="manual:dust_workos_device",
                access_token=grant.access_token,
                refresh_token=grant.refresh_token,
                extra={
                    "region": grant.region,
                    "workspace_id": workspace_id,
                    "expires_at_ms": grant.expires_at,
                },
            )
        )
        print(f"Signed in to Dust workspace {workspace_id} ({grant.region}).")
        return True
    if action == "status":
        entries = _pool().entries()
        print("dust: " + ("logged in" if entries else "logged out"))
        for entry in entries:
            extra = getattr(entry, "extra", {}) or {}
            print(f"  - {getattr(entry, 'label', '')} region={extra.get('region', '?')}")
        return True
    if action == "logout":
        for entry in list(_pool().entries()):
            _pool().remove_entry(getattr(entry, "id", ""))
        print("dust: logged out.")
        return True
    if action == "refresh":
        for entry in list(_pool().entries()):
            refreshed = refresh_credential(entry)
            if refreshed:
                _pool().update_entry(getattr(entry, "id", ""), **refreshed)
        print("dust: grant refreshed.")
        return True
    return False


def refresh_credential(entry: Any) -> dict[str, Any] | None:
    """Rotate a pooled OAuth row; the pool persists the returned fields."""
    from hermes_dust.workos import WorkosAuthError, refresh_grant

    refresh_token = getattr(entry, "refresh_token", "")
    if not refresh_token:
        return None
    try:
        grant = refresh_grant(refresh_token)
    except WorkosAuthError as exc:
        logger.warning("dust: grant refresh failed: %s", exc)
        return None
    extra = dict(getattr(entry, "extra", {}) or {})
    extra.update({"region": grant.region, "expires_at_ms": grant.expires_at})
    return {"access_token": grant.access_token, "refresh_token": grant.refresh_token, "extra": extra}


def _choose_workspace(access_token: str, region: str) -> str:
    """Resolve the workspace, asking when the user belongs to several."""
    from hermes_dust.http import DustRequestContext, dust_json

    try:
        body = dust_json(DustRequestContext(base=f"{_host(region)}/api/v1", access_token=access_token), "me")
    except Exception as exc:
        raise RuntimeError(f"Dust: cannot list workspaces: {exc}") from exc
    workspaces = (body or {}).get("workspaces", []) if isinstance(body, dict) else []
    ids = [w.get("sId") for w in workspaces if isinstance(w, dict) and w.get("sId")]
    if not ids:
        raise RuntimeError("Dust: no workspace found for this user.")
    if len(ids) == 1:
        return ids[0]
    raise RuntimeError(
        "Dust: several workspaces found; re-run with --workspace <sId>. " f"Available: {', '.join(ids)}"
    )


def _host(region: str) -> str:
    from hermes_dust.constants import dust_host

    return dust_host(region)


def fetch_dust_models(api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0) -> list[str] | None:
    """Live model ids for the picker; None lets the core use fallbacks."""
    from hermes_dust.http import DustRequestContext, dust_json
    from hermes_dust.models import fetch_models

    try:
        entry = _pick_entry()
    except RuntimeError:
        return None
    access_token, region, workspace_id = _entry_context(entry)
    if not workspace_id:
        return None
    context = DustRequestContext(
        base=dust_private_api_base(region, workspace_id),
        access_token=lambda refresh: _refreshed_token(entry, refresh),
        timeout_ms=timeout * 1000,
    )
    _ = access_token  # resolved per request through the lambda above
    try:
        models = fetch_models(lambda: dust_json(context, "models"))
    except Exception as exc:
        logger.warning("dust: catalog fetch failed: %s", exc)
        return None
    ids = [m.id for m in models if m.id != "agent-default"]
    return ids or None


def _refreshed_token(entry: Any, refresh: bool) -> str:
    if not refresh:
        return getattr(entry, "access_token", "")
    refreshed = refresh_credential(entry)
    if refreshed:
        _pool().update_entry(getattr(entry, "id", ""), **refreshed)
        return refreshed.get("access_token", "")
    return getattr(entry, "access_token", "")


def create_dust_client(**client_kwargs: Any) -> Any:
    """Supply the Dust facade client instead of the standard HTTP client."""
    from hermes_dust.client import DustClient, DustClientConfig, split_model_variant

    entry = _pick_entry()
    access_token, region, workspace_id = _entry_context(entry)
    raw_model = str(client_kwargs.get("model") or "")
    bare_model, variant_effort = split_model_variant(raw_model.split("/", 1)[-1] if "/" in raw_model else raw_model)
    _ = bare_model  # the facade re-parses the full id per call; only the effort is needed here.
    return DustClient(
        config=DustClientConfig(
            access_token=lambda refresh: _refreshed_token(entry, refresh),
            region=region,
            workspace_id=workspace_id,
            agent_s_id="dust",
            reasoning_effort=variant_effort,
        )
    )


def _live_catalog() -> Any:
    """Workspace catalog when a pooled grant exists, else None."""
    from hermes_dust.http import DustRequestContext, dust_json
    from hermes_dust.models import fetch_models

    try:
        entry = _pick_entry()
    except Exception:
        return None
    try:
        _, region, workspace_id = _entry_context(entry)
        if not workspace_id:
            return None
        context = DustRequestContext(
            base=dust_private_api_base(region, workspace_id),
            access_token=lambda refresh: _refreshed_token(entry, refresh),
        )
        return fetch_models(lambda: dust_json(context, "models"))
    except Exception as exc:
        logger.debug("dust: live catalog unavailable: %s", exc)
        return None


def build_dust_profile() -> Any:
    """Build the Hermes ProviderProfile; needs a Hermes core to import.

    Never requires a login: without a pooled grant the catalog falls back
    to the curated list, so importing the plugin always registers the
    provider and `hermes auth add dust` can find it afterwards.
    """
    from providers.base import ProviderProfile

    from hermes_dust.models import AGENT_DEFAULT_MODEL, fetch_models, hermes_model_capabilities

    catalog = _live_catalog() or [AGENT_DEFAULT_MODEL]

    profile = ProviderProfile(
        name=PROVIDER_NAME,
        aliases=PROVIDER_ALIASES,
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        signup_url=SIGNUP_URL,
        base_url="dust://conversation",
        auth_type="dust_workos",
        fallback_models=FALLBACK_MODELS,
        model_capabilities=hermes_model_capabilities(catalog),
        default_headers={"User-Agent": "Dust CLI"},
    )
    profile.auth_handler = dust_auth_handler  # type: ignore[attr-defined]
    profile.refresh_credential = refresh_credential  # type: ignore[attr-defined]
    profile.fetch_models = fetch_dust_models  # type: ignore[method-assign]
    profile.create_client = create_dust_client  # type: ignore[method-assign]
    # No HTTP endpoint to probe: inference goes through the DustClient over
    # the conversation API, so the generic /models health check is skipped.
    profile.supports_health_check = False  # type: ignore[attr-defined]
    return profile


try:
    dust_profile = build_dust_profile()
except Exception as exc:  # Hermes core absent (unit tests) — provider.py still imports.
    logger.debug("dust: profile not built here: %s", exc)
    dust_profile = None
