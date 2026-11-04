"""WorkOS device-authorization grant — the only way this plugin authenticates.

Ported from opencode-dust ``src/dust/workos.ts``. Dust classifies any turn
made with a workspace API key as programmatic usage regardless of origin, so
an API-key path would defeat the point of the plugin. There is deliberately
none. HTTP is injected so tests never touch the network.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from hermes_dust.constants import (
    DUST_DEFAULT_REGION,
    WORKOS_CLIENT_ID,
    WORKOS_DEVICE_AUTHORIZE_URL,
    WORKOS_REGION_CLAIM,
    WORKOS_SCOPE,
    WORKOS_TOKEN_URL,
)

# Refresh this far ahead of the true expiry. A turn can stream for minutes
# on a token WorkOS issues with a short lifetime, so a token that merely has
# not expired yet is not good enough to start one with.
EXPIRY_SKEW_MS = 60_000

# Extra delay WorkOS's "slow_down" response asks for, per RFC 8628.
SLOW_DOWN_STEP_MS = 5_000


class WorkosAuthError(Exception):
    """A WorkOS grant attempt that failed."""

    def __init__(self, code: str | None, requires_reauth: bool, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.requires_reauthentication = requires_reauth


@dataclass(frozen=True)
class DeviceCode:
    """A pending device authorization, as WorkOS issued it."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval_seconds: int


@dataclass(frozen=True)
class TokenGrant:
    """A usable WorkOS grant, with the facts Dust needs derived from it."""

    access_token: str
    refresh_token: str
    expires_at: int
    region: str


def _default_post_form(url: str, fields: dict[str, str]) -> tuple[int, Any]:
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.getcode(), json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:
            body = {}
        return exc.code, body


def decode_claims(token: str) -> dict[str, Any] | None:
    """Decode a JWT payload without verifying it.

    Read only for claims this client uses to route its own requests. Dust
    verifies the signature on every call, so a forged token buys nothing but
    a rejected request.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception:
        # A token this client cannot read is still usable as opaque bearer.
        return None


def decode_region(access_token: str) -> str:
    """Read the Dust region an access token belongs to."""
    claims = decode_claims(access_token) or {}
    claim = claims.get(WORKOS_REGION_CLAIM)
    return claim if isinstance(claim, str) else DUST_DEFAULT_REGION


def grant_expires_at(access_token: str) -> int:
    """Read when an access token stops being valid (epoch ms, 0 if unknown)."""
    claims = decode_claims(access_token) or {}
    exp = claims.get("exp")
    return int(exp * 1000) if isinstance(exp, (int, float)) else 0


def is_grant_expired(expires_at: int, now_ms: int) -> bool:
    """Whether a grant should be refreshed before being used."""
    return expires_at - EXPIRY_SKEW_MS <= now_ms


def _to_grant(body: Any) -> TokenGrant:
    if not isinstance(body, dict):
        body = {}
    access = body.get("access_token")
    refresh = body.get("refresh_token")
    if not isinstance(access, str) or not isinstance(refresh, str):
        raise WorkosAuthError(None, True, "WorkOS returned no token pair; the grant cannot be used.")
    return TokenGrant(
        access_token=access,
        refresh_token=refresh,
        expires_at=grant_expires_at(access),
        region=decode_region(access),
    )


def request_device_code(
    post_form: Callable[[str, dict[str, str]], tuple[int, Any]] | None = None,
) -> DeviceCode:
    """Begin a device authorization."""
    post = post_form or _default_post_form
    status, body = post(WORKOS_DEVICE_AUTHORIZE_URL, {"client_id": WORKOS_CLIENT_ID, "scope": WORKOS_SCOPE})
    if status < 200 or status >= 300:
        error = body if isinstance(body, dict) else {}
        raise WorkosAuthError(
            error.get("error"),
            True,
            f"WorkOS refused the device authorization request: "
            f"{error.get('error_description') or error.get('error') or status}",
        )
    device = body if isinstance(body, dict) else {}
    return DeviceCode(
        device_code=str(device.get("device_code")),
        user_code=str(device.get("user_code")),
        verification_uri=str(device.get("verification_uri")),
        verification_uri_complete=str(device.get("verification_uri_complete")),
        expires_in=int(device.get("expires_in") or 0),
        interval_seconds=int(device.get("interval") or 5),
    )


def poll_for_grant(
    device: DeviceCode,
    post_form: Callable[[str, dict[str, str]], tuple[int, Any]] | None = None,
    sleep: Callable[[float], None] | None = None,
    deadline_s: float = 900,
) -> TokenGrant:
    """Poll WorkOS until the human approves the device code.

    "authorization_pending" and "slow_down" are normal progress, not
    failures; every other error ends the attempt.
    """
    post = post_form or _default_post_form
    pause = sleep or time.sleep
    interval_ms = device.interval_seconds * 1000
    started = time.monotonic()
    while True:
        status, body = post(
            WORKOS_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device.device_code,
                "client_id": WORKOS_CLIENT_ID,
            },
        )
        if 200 <= status < 300:
            return _to_grant(body)
        code = body.get("error") if isinstance(body, dict) else None
        if code == "slow_down":
            interval_ms += SLOW_DOWN_STEP_MS
        elif code != "authorization_pending":
            raise WorkosAuthError(
                code,
                True,
                "The login code expired before it was approved."
                if code == "expired_token"
                else f"WorkOS ended the login attempt: {code or status}",
            )
        if time.monotonic() - started > deadline_s:
            raise WorkosAuthError("expired_token", True, "The login code expired before it was approved.")
        pause(interval_ms / 1000)


def refresh_grant(
    refresh_token: str,
    post_form: Callable[[str, dict[str, str]], tuple[int, Any]] | None = None,
) -> TokenGrant:
    """Exchange a refresh token for a fresh grant."""
    post = post_form or _default_post_form
    status, body = post(
        WORKOS_TOKEN_URL,
        {"grant_type": "refresh_token", "client_id": WORKOS_CLIENT_ID, "refresh_token": refresh_token},
    )
    if 200 <= status < 300:
        return _to_grant(body)
    error = body if isinstance(body, dict) else {}
    rejected = status in (400, 401)
    raise WorkosAuthError(
        error.get("error"),
        rejected,
        "The stored Dust login is no longer valid; log in again."
        if rejected
        else f"WorkOS could not refresh the login: {error.get('error') or status}",
    )
