"""What every bridged turn tells the Dust agent about Hermes.

Adapted from opencode-dust ``src/bridge/note.ts``. The Dust agent runs its
own loop server-side and is never sent Hermes tools directly. In phase 1
(text-only transport) the note keeps the agent answering in the
conversation. In phase 2 (tool bridge roadmap in README) the same note will
point the agent at the Hermes tool-exchange instead of the opencode pod
helper — the mechanism changes, the job does not: make the host's tools the
obvious way to touch the project.
"""

from __future__ import annotations

_HERMES_TOOLS_LINE = (
    "Available Hermes tools (executed by the host on your behalf, "
    "one round-trip per batch): read_file, write_file, patch, terminal, "
    "search_files, web_search, web_extract, delegate_task, todo_list."
)


def hermes_bridge_note(tools: list[str] | None = None, code_mode_catalog: str | None = None) -> str:
    """Build the turn note for Hermes.

    :param tools: host tool signatures to advertise; defaults to the core set.
    :param code_mode_catalog: extra tool catalog when the host offers code-mode tools.
    """
    available = ", ".join(tools) if tools else _HERMES_TOOLS_LINE
    parts = [
        "You are working on a project that lives on the developer's machine, not in your sandbox. ",
        "To read, search, edit or run anything there, ask the host: emit one request per step as ",
        "`{\"calls\":[{\"tool\":\"read_file\",\"arguments\":{...}},"
        "{\"tool\":\"terminal\",\"arguments\":{\"command\":\"npm test\"}}]}`. ",
        "It runs on the developer's machine through their own tools and permissions, "
        "and the result comes back as a message.",
        "",
        "",
        "A request is a batch of tool calls, run in order and stopping at the first failure. ",
        f"{available} ",
        "Put a whole step in one request: read the files you need together, make every edit "
        "and run the check in one batch. ",
        "If the host says the work is still running, end your turn at once and say you are waiting: "
        "the result is delivered to you as a message.",
        "",
        "",
        "Your sandbox stays yours for scratch work — Python, quick computations, drafts in `/tmp` — "
        "which costs nothing; just remember it never holds the project's files.",
    ]
    if code_mode_catalog and code_mode_catalog.strip():
        parts.append(
            "\n\nOther tools — MCP servers, the browser, and the like — live behind the host's "
            "code mode and are only reachable through the `execute` tool with the path from "
            "the catalog below.\n\n" + code_mode_catalog.strip()
        )
    return "".join(parts)
