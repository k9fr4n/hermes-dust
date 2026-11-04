"""Pod/sandbox exchange protocol — the same mechanism as opencode-dust.

The Dust agent runs in its own Computer sandbox, where the pod is mounted
at ``/files/pod``. It leaves a request (a JSON batch of tool calls, or a
shell script) and waits for the answer inside the same free ``bash`` call —
see ``helper_script()``. This side polls the pod, runs the work through
Hermes' own tool loop, and writes the ``.out`` the helper is waiting on.

A bridged tool call costs Dust nothing extra: writing and reading pod files
is ``basic`` with ``freeUsage``, where an external MCP tool call is always
``advanced``. What it costs instead is a round-trip, so a request is a batch
carrying a whole step, not a single call.

Ported from opencode-dust ``src/bridge/{helper,pod-bash,pod-tool-request}.ts``.
Hermes naming (``hs-bash/``, ``hs-tools/hs.sh``) replaces the opencode names;
the wire shapes — suffixes, ``.out``/``.detached`` markers, result formats —
are unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Where a pod keeps the shell exchange, apart from the project.
# Not dot-prefixed: Dust's listing omits dot-directories entirely, and the
# listing is the only way this side learns a request exists at all.
POD_BASH_DIR = "hs-bash"

# Where the helper lives in the pod: outside the exchange the watcher polls.
HELPER_PATH = "hs-tools/hs.sh"

# Where the sandbox mounts the conversation's pod.
SANDBOX_POD_ROOT = "/files/pod"

# Seconds the helper waits for an answer. Under the sandbox's 120-second
# ceiling with room to spare: a call killed by that ceiling leaves no
# marker, and the agent learns nothing but a timeout.
HELPER_WAIT_SECONDS = 100

# What the agent writes: the script to run.
REQUEST_SUFFIX = ".sh"

# What the agent writes to run Hermes tools instead of a shell.
TOOL_REQUEST_SUFFIX = ".json"

# What this side writes back: what the work printed.
RESULT_SUFFIX = ".out"

# Marker a timed-out wait leaves beside its request.
DETACHED_SUFFIX = ".detached"

# Line between the header and the program's own bytes in a script result.
BASH_RESULT_SEPARATOR = "---"

# How much command output is worth sending back. The agent pays for every
# character of this file in its own context; generous rather than tight.
MAX_RESULT_CHARS = 60_000

# Most calls one request may carry. A bound on the damage a confused batch
# can do before anybody sees it, and on the output one `.out` has to hold.
MAX_BATCH_CALLS = 32

# Least output any one call in a batch is guaranteed. Below this a result
# is not worth returning at all; a batch crossing it overruns the total.
MIN_CALL_CHARS = 4_000

# A request name this side turns into a path. The id is chosen by the model
# and becomes part of a path this process writes to; anything outside this
# set — a slash above all — is not a name.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Tools Hermes reserves for itself; the agent has Dust's own equivalents.
NOT_BRIDGED = frozenset({"web_search", "web_extract"})


def pod_name_for(directory: str) -> str:
    """Pod name a project's exchange lives under."""
    base = directory.rstrip("/").split("/")[-1] if directory else ""
    return f"hermes-{base or 'project'}"


def exchange_dir(conversation_id: str) -> str:
    """Where one conversation's requests live (pod-relative).

    One pod serves every session of a project, while a request name is only
    unique within its conversation: two sessions would otherwise both leave
    a ``step1`` and read each other's answer.
    """
    return f"{POD_BASH_DIR}/{conversation_id}"


def detached_path(conversation_id: str, request_id: str) -> str:
    """Marker a timed-out wait leaves beside its request (pod-relative)."""
    return f"{exchange_dir(conversation_id)}/{request_id}{DETACHED_SUFFIX}"


def bash_request_path(request_id: str) -> str:
    return f"{POD_BASH_DIR}/{request_id}{REQUEST_SUFFIX}"


def tool_request_path(request_id: str) -> str:
    return f"{POD_BASH_DIR}/{request_id}{TOOL_REQUEST_SUFFIX}"


def bash_result_path(request_id: str) -> str:
    return f"{POD_BASH_DIR}/{request_id}{RESULT_SUFFIX}"


def is_pod_bash_path(path: str) -> bool:
    return path == POD_BASH_DIR or path.startswith(f"{POD_BASH_DIR}/")


@dataclass(frozen=True)
class PodRequest:
    """One request the agent left in the pod, and how to read it."""

    id: str
    kind: str  # "script" | "tools"
    path: str  # pod-relative path to read it from


@dataclass(frozen=True)
class PodFile:
    path: str
    size_bytes: int = 0
    modified_ms: int = 0


def pending_requests(files: list[PodFile], claimed: set[str]) -> list[PodRequest]:
    """Requests this side has not answered yet, oldest first.

    Order is by Dust's modification stamp, not name: the agent writes a
    script per step and expects step two to see what step one did. Both
    routes come back from one call, interleaved by that stamp, because they
    are one queue. ``claimed`` is held in memory rather than derived from
    the listing: the delete and the listing both go through Cloudflare's
    cache, so a request already taken can reappear in the next listing.
    """
    found: list[tuple[PodFile, str, str, str]] = []
    prefix = f"{POD_BASH_DIR}/"
    for entry in files:
        if not entry.path.startswith(prefix):
            continue
        name = entry.path[len(prefix):]
        for suffix, kind in ((REQUEST_SUFFIX, "script"), (TOOL_REQUEST_SUFFIX, "tools")):
            if not name.endswith(suffix):
                continue
            found.append((entry, name[: -len(suffix)], kind, entry.path))
            break
    return [
        PodRequest(id=request_id, kind=kind, path=path)
        for _, request_id, kind, path in sorted(
            found, key=lambda item: (item[0].modified_ms, item[1])
        )
        if _SAFE_ID.match(request_id) and request_id not in claimed
    ]


def trim_to(output: str, cap: int) -> str:
    """Cut an over-long output, keeping both ends."""
    if len(output) <= cap:
        return output
    half = cap // 2
    cut = len(output) - cap
    return f"{output[:half]}\n\n[{cut} characters cut from the middle]\n\n{output[-half:]}"


@dataclass(frozen=True)
class PodToolCall:
    tool: str
    arguments: dict


@dataclass(frozen=True)
class PodToolRequest:
    calls: tuple
    on_error: str  # "stop" | "continue"


def _pick(source: dict, names: tuple[str, ...]) -> object:
    for name in names:
        if name in source:
            return source[name]
    return None


_SHAPE_HINT = (
    'Expected {"calls":[{"tool":"read_file","arguments":{...}},...]} — a JSON object with'
    " a `calls` array, or a bare array of the same objects."
)


def _parse_call(value: object, index: int) -> PodToolCall | str:
    at = f"call {index + 1}"
    if not isinstance(value, dict):
        kind = "an array" if isinstance(value, list) else type(value).__name__
        return f"{at} is {kind}, not an object."
    tool = _pick(value, ("tool", "name"))
    if not isinstance(tool, str) or not tool.strip():
        return f"{at} has no tool name. Give it a `tool`, as in {{\"tool\":\"read_file\",\"arguments\":{{...}}}}."
    args = _pick(value, ("arguments", "args", "input", "parameters"))
    if args is not None and not isinstance(args, dict):
        kind = "an array" if isinstance(args, list) else type(args).__name__
        return f"{at} ({tool}) has `arguments` as {kind}; it must be an object of named arguments."
    return PodToolCall(tool=tool.strip(), arguments=dict(args or {}))


def parse_tool_request(text: str) -> tuple[bool, object]:
    """Read a request file as a batch of tool calls.

    Never throws: every failure is a message the agent can read and correct.
    Returns ``(True, PodToolRequest)`` or ``(False, error_message)``.
    """
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"This file is not valid JSON: {exc}. {_SHAPE_HINT}"
    if isinstance(body, list):
        raw = body
    elif isinstance(body, dict):
        raw = _pick(body, ("calls", "tools", "requests")) or [body]
    else:
        raw = None
    if not isinstance(raw, list):
        if raw is None:
            return False, f"This file is JSON but not a batch of calls. {_SHAPE_HINT}"
        return False, f"`calls` is {type(raw).__name__}, not an array. {_SHAPE_HINT}"
    if not raw:
        return False, f"This batch has no calls in it. {_SHAPE_HINT}"
    if len(raw) > MAX_BATCH_CALLS:
        return False, (
            f"This batch asks for {len(raw)} calls; the most one request may carry"
            f" is {MAX_BATCH_CALLS}. Split it across two requests."
        )
    calls: list[PodToolCall] = []
    for index, candidate in enumerate(raw):
        call = _parse_call(candidate, index)
        if isinstance(call, str):
            return False, call
        calls.append(call)
    on_error = "continue" if isinstance(body, dict) and _pick(body, ("onError", "on_error")) == "continue" else "stop"
    return True, PodToolRequest(calls=tuple(calls), on_error=on_error)


def call_budget(count: int) -> int:
    """Per-call character cap for a batch of ``count`` calls."""
    return max(MIN_CALL_CHARS, MAX_RESULT_CHARS // max(1, count))


def _arg_names(schema: dict) -> list[str]:
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = {n for n in schema.get("required", []) if isinstance(n, str)}
    return [f"{name}{'' if name in required else '?'}" for name in properties]


def _arg_shape(schema: object) -> str:
    if not isinstance(schema, dict):
        return ""
    if schema.get("type") == "array":
        items = schema.get("items")
        if isinstance(items, dict) and isinstance(items.get("properties"), dict):
            return f"[{','.join(_arg_names(items))}]"
        return "[]"
    if isinstance(schema.get("properties"), dict):
        inner = schema["properties"]
        required = {n for n in schema.get("required", []) if isinstance(n, str)}
        args = [f"{name}{'' if name in required else '?'}" for name in inner]
        return f"{{{','.join(args)}}}"
    return ""


def tool_signatures(tools: list[dict]) -> list[str]:
    """Compact ``name(arg, nested[{a, b}], optional?)`` signatures for the note.

    Names and shapes only — derived from the live tool definitions so a tool
    changing its arguments changes the note with it.
    """
    lines = []
    for tool in tools:
        schema = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        properties = schema.get("properties") if isinstance(schema, dict) and isinstance(schema.get("properties"), dict) else {}
        required = {n for n in (schema.get("required", []) if isinstance(schema, dict) else []) if isinstance(n, str)}
        args = [f"{name}{'' if name in required else '?'}{_arg_shape(properties[name])}" for name in properties]
        lines.append(f"{tool.get('name')}({','.join(args)})")
    return lines


_SEARCH_TOOLS = frozenset({"terminal", "search_files", "read_file", "write_file"})


def call_label(tool: str, args: dict) -> str:
    """How one call reads in a plan: ``$ npm test``, ``read_file src/a.ts``."""
    if tool in ("terminal", "shell", "bash"):
        command = args.get("command")
        return tool if not isinstance(command, str) or not command else f"$ {command}"
    pattern = args.get("pattern") or args.get("query")
    path = args.get("path") or args.get("filePath")
    detail = pattern if tool in _SEARCH_TOOLS and pattern else (path or pattern)
    return tool if not isinstance(detail, str) or not detail else f"{tool} {detail}"


def describe_tool_request(body: str) -> str:
    """A request's plan, one call per line, for the human to read."""
    ok, parsed = parse_tool_request(body)
    if not ok:
        return body.strip()
    assert isinstance(parsed, PodToolRequest)
    lines = [f"{i + 1}. {call_label(call.tool, call.arguments)}" for i, call in enumerate(parsed.calls)]
    if parsed.on_error == "continue":
        lines.append("(continues past a failure)")
    return "\n".join(lines)


@dataclass(frozen=True)
class PodToolOutcome:
    tool: str
    arguments: dict
    output: str
    is_error: bool
    duration_ms: int = 0


def _call_header(index: int, outcome: PodToolOutcome) -> str:
    return f"=== call {index + 1} {outcome.tool} {'failed' if outcome.is_error else 'ok'} {(outcome.duration_ms / 1000):.1f}s"


def format_tool_batch_result(outcomes: list[PodToolOutcome], asked: int) -> str:
    """The file the agent reads to find out what its batch did."""
    failed = sum(1 for o in outcomes if o.is_error)
    cap = call_budget(asked)
    sections = [
        f"{_call_header(i, o)}\nargs: {json.dumps(o.arguments)}\n{trim_to(o.output, cap)}"
        for i, o in enumerate(outcomes)
    ]
    for i in range(len(outcomes), asked):
        sections.append(
            f"=== call {i + 1} not run"
            " (the batch stopped at the first failure; send it again from here)"
        )
    total_s = sum(o.duration_ms for o in outcomes) / 1000
    return (
        f"status: {'failed' if failed else 'ok'}\n"
        f"seconds: {total_s:.1f}\n"
        f"calls: {asked} asked, {len(outcomes)} ran, {failed} failed\n"
        "---\n" + "\n".join(sections)
    )


def format_bash_result(output: str, is_error: bool, duration_ms: int) -> str:
    return (
        f"status: {'failed' if is_error else 'ok'}\n"
        f"seconds: {(duration_ms / 1000):.1f}\n"
        f"{BASH_RESULT_SEPARATOR}\n"
        f"{trim_to(output, MAX_RESULT_CHARS)}"
    )


def format_delivery(what: str, path: str, body: str) -> str:
    """Message one finished request is delivered in.

    Arrives as an automatic message because that is the only channel into a
    turn the agent already ended — and marked, so a wall of output never
    reads as something the human typed.
    """
    return (
        f"[automatic message, not the developer] The {what} you left at"
        f" `{path}` has finished. Carry on from here.\n\n{body}"
    )


def format_bash_delivery(request_id: str, body: str) -> str:
    return format_delivery("script", bash_request_path(request_id), body)


def helper_script() -> str:
    """The helper's source.

    The conversation is read from the sandbox's own mount,
    ``/files/conversation`` → ``/files/conversation-<id>``, so the agent
    never has to know its id. ``HS_POD_ROOT``, ``HS_WAIT`` and
    ``HS_CONVERSATION`` exist for tests; in the sandbox the defaults apply.
    The request is staged in ``/tmp`` and copied in whole, so the watcher
    never lists a request that is still being written.
    """
    return f"""#!/usr/bin/env bash
# hermes bridge: run work on the developer's machine and wait for the answer.
# Usage: bash {SANDBOX_POD_ROOT}/{HELPER_PATH} <name> <<'EOF'
#        {{"calls":[...]}}                       (a JSON batch of tool calls)
#        EOF
#    or: bash {SANDBOX_POD_ROOT}/{HELPER_PATH} <name> --sh <<'EOF'
#        shell commands                        (one script, run as one script)
#        EOF
set -u
root="${{HS_POD_ROOT:-{SANDBOX_POD_ROOT}}}"
wait="${{HS_WAIT:-{HELPER_WAIT_SECONDS}}}"
conversation="${{HS_CONVERSATION:-$(basename "$(readlink /files/conversation)" | sed 's/^conversation-//')}}"
dir="$root/{POD_BASH_DIR}/$conversation"
name="${{1:-}}"
mode="${{2:-}}"

if ! [[ "$conversation" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "error: this sandbox has no conversation mount at /files/conversation." >&2
  exit 2
fi

if ! [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{{0,63}}$ ]]; then
  echo "error: '$name' is not a usable request name; use letters, digits, '.', '_' or '-', like step1." >&2
  exit 2
fi
if compgen -G "$dir/$name.*" > /dev/null; then
  echo "error: the name $name was already used in this conversation; pick one you have never used." >&2
  exit 2
fi

body="$(mktemp)"
cat > "$body"
if ! grep -q '[^[:space:]]' "$body"; then
  echo "error: the request is empty; pass the batch or the script on stdin." >&2
  rm -f "$body"
  exit 2
fi
if [ "$mode" = "--sh" ]; then
  kind=sh
else
  # A batch is JSON or nothing: text that is not would otherwise run as a
  # shell script, and prose run as a script is an error at best.
  case "$(tr -d '[:space:]' < "$body" | head -c 1)" in
    "{{"|"[") kind=json ;;
    *)
      echo "error: the request is not a JSON batch, so nothing was sent. Send {{"calls":[...]}}, or shell commands with --sh. It is a request for the developer's tools, not a prompt." >&2
      rm -f "$body"
      exit 2
      ;;
  esac
  if command -v jq > /dev/null; then valid() {{ jq -e . "$1" > /dev/null 2>&1; }}
  elif command -v python3 > /dev/null; then valid() {{ python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" 2> /dev/null; }}
  elif command -v node > /dev/null; then valid() {{ node -e 'JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"))' "$1" 2> /dev/null; }}
  else valid() {{ true; }}
  fi
  if ! valid "$body"; then
    echo "error: the batch is not valid JSON, so nothing was sent; fix it and send it under a new name." >&2
    rm -f "$body"
    exit 2
  fi
fi

mkdir -p "$dir"
cp "$body" "$dir/$name.$kind"
rm -f "$body"

for _ in $(seq 1 "$wait"); do
  if [ -f "$dir/$name.out" ]; then
    cat "$dir/$name.out"
    exit 0
  fi
  sleep 1
done

touch "$dir/$name.detached"
echo "$name is still running on the developer's machine after ${{wait}}s."
echo "End your turn now and say you are waiting: its output will be delivered to you as a message when it finishes."
"""
