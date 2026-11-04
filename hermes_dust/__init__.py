"""Dust protocol ported from opencode-dust (TypeScript) to Python.

Only the host-independent protocol lives here: endpoint constants, HTTP
transport, SSE framing, WorkOS device grant, conversation helpers, history
digest and the Hermes bridge note. Anything Hermes-specific (provider
profile, credential pool, tool dispatch) stays in ``provider.py`` and
``client.py`` so the pure logic is testable without a Hermes core.
"""

from hermes_dust.constants import (
    DUST_DEFAULT_REGION,
    USER_MESSAGE_ORIGIN,
    WORKOS_CLIENT_ID,
    WORKOS_DEVICE_AUTHORIZE_URL,
    WORKOS_REGION_CLAIM,
    WORKOS_SCOPE,
    WORKOS_TOKEN_URL,
    dust_api_base,
    dust_host,
    dust_private_api_base,
)

__all__ = [
    "DUST_DEFAULT_REGION",
    "USER_MESSAGE_ORIGIN",
    "WORKOS_CLIENT_ID",
    "WORKOS_DEVICE_AUTHORIZE_URL",
    "WORKOS_REGION_CLAIM",
    "WORKOS_SCOPE",
    "WORKOS_TOKEN_URL",
    "dust_api_base",
    "dust_host",
    "dust_private_api_base",
]
