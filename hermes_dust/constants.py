"""Fixed Dust and WorkOS endpoints, and the constant deciding billing.

Ported from opencode-dust ``src/dust/constants.ts``. These are external
protocol constants, not deployment tunables: changing any of them changes
which service is talked to, not how this plugin is configured.
"""

from __future__ import annotations

# Origin stamped on every user message. Dust classifies a turn as
# programmatic usage when the message origin maps to "programmatic" in
# front/lib/api/programmatic_usage/common.ts. "cli" maps to "user" while
# "api" and "cli_programmatic" map to "programmatic". This plugin
# authenticates as a WorkOS user grant and stamps "cli", so its turns are
# billed as interactive user usage — the same as the Dust web app.
# There is deliberately no way to override this per request.
USER_MESSAGE_ORIGIN = "cli"

# WorkOS tenant issuing Dust user grants.
WORKOS_DOMAIN = "api.workos.com"

# Production WorkOS client the Dust CLI is registered as. The device grant
# is bound to this client, so a different id yields tokens Dust rejects.
WORKOS_CLIENT_ID = "client_01JGCT55T7FVDG9XF74925R1KT"

# Device-authorization endpoint issuing a user code and device code.
WORKOS_DEVICE_AUTHORIZE_URL = f"https://{WORKOS_DOMAIN}/user_management/authorize/device"

# Token endpoint for both the device-code exchange and refresh.
WORKOS_TOKEN_URL = f"https://{WORKOS_DOMAIN}/user_management/authenticate"

# OAuth scopes the device grant requests.
WORKOS_SCOPE = "openid profile email"

# Claim on the WorkOS access token naming the Dust region holding user data.
WORKOS_REGION_CLAIM = "https://dust.tt/region"

# Dust deployment host per region claim value.
DUST_REGION_HOSTS = {
    "us-central1": "https://dust.tt",
    "europe-west1": "https://eu.dust.tt",
}

# Region assumed when the access token carries no region claim.
DUST_DEFAULT_REGION = "us-central1"


def dust_host(region: str) -> str:
    """Resolve the Dust host for a region claim.

    An unrecognized region is not an error: WorkOS may report a region
    this build predates, and the default deployment can still answer it.
    """
    return DUST_REGION_HOSTS.get(region, DUST_REGION_HOSTS[DUST_DEFAULT_REGION])


def dust_api_base(region: str, workspace_id: str) -> str:
    """Base URL of Dust's public, documented workspace API."""
    return f"{dust_host(region)}/api/v1/w/{workspace_id}"


def dust_private_api_base(region: str, workspace_id: str) -> str:
    """Base URL of Dust's private workspace API.

    Reachable by a WorkOS user grant, not by a workspace API key. The model
    catalog lives here and has no public equivalent.
    """
    return f"{dust_host(region)}/api/w/{workspace_id}"
