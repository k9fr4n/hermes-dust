"""What every bridged turn tells the Dust agent about Hermes.

Adapted from opencode-dust ``src/bridge/note.ts``. The agent runs Dust's
global ``dust`` agent loop and has none of Hermes' tools: Hermes strips them
from the request. Its own Computer sandbox is free, and the pod mounted in
it carries only the exchange — the project itself stays on the developer's
machine. So the note has one job: make the helper the obvious way to touch
the project, batched a whole step at a time.
"""

from __future__ import annotations

from hermes_dust.bridge.protocol import HELPER_PATH, HELPER_WAIT_SECONDS, SANDBOX_POD_ROOT

# Sandbox `bash` timeout the helper's wait needs, under the sandbox ceiling.
BASH_TIMEOUT_MS = 120_000


def hermes_bridge_note(tools: list[str] | None = None) -> str:
    """Build the turn note.

    :param tools: the developer's direct Hermes tools as compact signatures
        (see ``tool_signatures``); a static fallback when the call carried none.
    """
    available = ", ".join(tools) if tools else "read_file, write_file, patch, terminal, search_files, todo_list"
    helper = f"bash {SANDBOX_POD_ROOT}/{HELPER_PATH} <name>"
    return "".join(
        [
            "You are working on a project that lives on the developer's machine, not in ",
            f"`{SANDBOX_POD_ROOT}` and not in your sandbox.",
            " To read, search, edit or run anything there, use your sandbox `bash` tool,",
            f" with `timeoutMs: {BASH_TIMEOUT_MS}`, to run `{helper} <<'EOF'` followed by the request and `EOF`.",
            " It runs on the developer's machine through their own tools and permissions, and prints the result in the same call.",
            "",
            "",
            "A request is a batch of tool calls, run in order and stopping at the first failure unless you add `\"onError\":\"continue\"`:",
            ' `{"calls":[{"tool":"read_file","arguments":{...}},{"tool":"terminal","arguments":{"command":"npm test"}}]}`.',
            f" Available tools: {available}.",
            " Separate calls do not share a shell; when a step needs one (a persistent `cd`, a heredoc), run the helper with `--sh` after the name and send plain shell commands: they run as one script.",
            " The request is a request for the developer's tools, not a prompt: anything else is refused unread.",
            "",
            "",
            "Each request costs a round-trip, whatever is in it, so put a whole step in one request: read the files you need together, make every edit and run the check in one batch.",
            " Prefer `patch` and `write_file` over `sed`: they are exact where a shell rewrite is a guess.",
            " `<name>` must be one you have never used in this conversation (`step1`, `step2`, …).",
            f" If the helper says the work is still running (after {HELPER_WAIT_SECONDS}s), end your turn at once and say you are waiting: the result is delivered to you as a message.",
            "",
            "",
            "Your sandbox stays yours for scratch work — Python, quick computations, drafts in `/tmp` — which costs nothing; just remember it never holds the project's files.",
        ]
    )
