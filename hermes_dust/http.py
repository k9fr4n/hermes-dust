"""HTTP transport for Dust's workspace API.

Ported from opencode-dust ``src/dust/http.ts``. Uses only the standard
library so the plugin has no runtime dependencies. The ``fetch`` callable is
injected (default: a thin urllib wrapper) so tests never touch the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import json
import urllib.request

# User agent Dust requires before it accepts a "cli" message origin.
# isUserMessageContextValid gates that origin on authMethod == "oauth" and
# userAgent == "Dust CLI". A different value does not downgrade the origin —
# Dust rejects the whole message with HTTP 400 — so this exact value is part
# of the origin contract, not decoration.
_USER_AGENT = "Dust CLI"

# What this client actually is, kept alongside the required user agent.
_CLIENT_ID = "hermes-dust"

# This plugin's version, reported alongside the client id.
CLIENT_VERSION = "0.1.0"

DustFailureKind = Literal[
    "unauthorized", "forbidden", "gone", "rate-limited", "client", "server"
]


class DustApiError(Exception):
    """A Dust response that was not a success, classified and attributed."""

    def __init__(self, status: int, kind: DustFailureKind, code: str | None, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.code = code


def classify_status(status: int) -> DustFailureKind:
    """Classify an HTTP status.

    403 is deliberately not "gone": Dust answers 403 when a workspace lacks
    a feature flag the request needs, a permanent configuration fact about a
    resource that still exists.
    """
    if status == 401:
        return "unauthorized"
    if status == 403:
        return "forbidden"
    if status in (404, 410):
        return "gone"
    if status == 429:
        return "rate-limited"
    return "server" if status >= 500 else "client"


def dust_auth_headers(access_token: str) -> dict[str, str]:
    """Headers every Dust request carries."""
    return {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": _USER_AGENT,
        # The Dust CLI sends its version under this name; keeping it means
        # the required user agent above does not hide which client is calling.
        "X-Dust-CLI-Version": CLIENT_VERSION,
        "X-Dust-Client": _CLIENT_ID,
    }


@dataclass
class DustRequestContext:
    """Authenticated identity of one Dust API caller."""

    base: str
    access_token: str | Callable[[bool], str]
    timeout_ms: float | None = 30_000
    fetch: Callable[..., Any] | None = None


def resolve_token(source: str | Callable[[bool], str], refresh: bool = False) -> str:
    """Resolve the bearer token for one request."""
    return source(refresh) if callable(source) else source


def _default_fetch(
    url: str, method: str, headers: dict[str, str], body: bytes | None, timeout_s: float | None
) -> Any:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    return urllib.request.urlopen(request, timeout=timeout_s)


class _ResponseView:
    """Minimal adapter over urllib responses and test doubles."""

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def status(self) -> int:
        status = getattr(self._response, "status", None)
        if status is None:
            status = self._response.getcode()
        return int(status)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        raw = self._response.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw) if raw else {}


@dataclass
class DustResponse:
    status: int
    body: Any = field(default=None)
    raw: Any = field(default=None)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def dust_fetch(
    context: DustRequestContext,
    path: str,
    method: str = "GET",
    body: Any = None,
    raw_body: bool = False,
    headers: dict[str, str] | None = None,
) -> DustResponse:
    """Perform one Dust API call without inspecting the response.

    A 401 is retried once against a freshly minted token when the context
    can mint one. Dust's tokens expire on a fifteen-minute clock no caller
    can align a turn to, so the alternative is a turn dying at an arbitrary
    point in its own generation.
    """
    fetch = context.fetch or _default_fetch
    timeout_s = context.timeout_ms / 1000 if context.timeout_ms else None

    def send(token: str) -> DustResponse:
        request_headers = {**dust_auth_headers(token), **(headers or {})}
        payload: bytes | None = None
        if body is not None:
            if raw_body:
                payload = body if isinstance(body, bytes) else str(body).encode("utf-8")
            else:
                request_headers.setdefault("Content-Type", "application/json")
                payload = json.dumps(body).encode("utf-8")
        try:
            response = fetch(
                f"{context.base}/{path}",
                method=method,
                headers=request_headers,
                body=payload,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status", None)
            if status is None:
                raise
            return DustResponse(status=int(status), raw=exc)
        if isinstance(response, DustResponse):
            return response
        view = _ResponseView(response)
        try:
            decoded: Any = view.json() if view.ok else None
            if not view.ok:
                try:
                    decoded = view.json()
                except Exception:
                    decoded = None
        except Exception:
            decoded = None
        return DustResponse(status=view.status, body=decoded, raw=response)

    response = send(resolve_token(context.access_token))
    if response.status != 401 or not callable(context.access_token):
        return response
    return send(resolve_token(context.access_token, True))


def dust_json(
    context: DustRequestContext,
    path: str,
    method: str = "GET",
    body: Any = None,
    raw_body: bool = False,
    headers: dict[str, str] | None = None,
) -> Any:
    """Perform one Dust API call and decode its JSON body."""
    response = dust_fetch(context, path, method=method, body=body, raw_body=raw_body, headers=headers)
    if not response.ok:
        detail = ""
        if isinstance(response.body, dict):
            error = response.body.get("error") or {}
            detail = error.get("message") or error.get("type") or ""
        raise DustApiError(
            response.status,
            classify_status(response.status),
            (response.body.get("error") or {}).get("type")
            if isinstance(response.body, dict)
            else None,
            f"Dust {path} failed with {response.status}: {detail or 'request failed'}",
        )
    return response.body
