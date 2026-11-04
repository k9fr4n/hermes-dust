"""Pod/sandbox bridge: the Dust agent works on the developer's machine.

Same mechanism as opencode-dust: the agent leaves requests in the pod from
its Computer sandbox and waits in the same free ``bash`` call; this side
polls, runs the work through Hermes' own tool loop, and writes the ``.out``.
"""

from hermes_dust.bridge import protocol
from hermes_dust.bridge.protocol import (
    HELPER_PATH,
    HELPER_WAIT_SECONDS,
    POD_BASH_DIR,
    SANDBOX_POD_ROOT,
)

__all__ = ["HELPER_PATH", "HELPER_WAIT_SECONDS", "POD_BASH_DIR", "SANDBOX_POD_ROOT", "protocol"]
