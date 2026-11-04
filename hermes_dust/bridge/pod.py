"""Dust pod filesystem and pod management.

Ported from opencode-dust ``src/dust/{pods,pod-files,directory}.ts``.
Everything goes through ``files/path/pod-<spaceId>/<rel>``, the same
canonical route Dust's own ``files`` tools are built on — the only way both
sides see one pod.

Every read carries a cache-buster: Dust serves these files through
Cloudflare with ``cache-control: public, max-age=14400``, so a plain GET can
answer with a four-hour-old body. A unique query parameter makes it a MISS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from hermes_dust.bridge.protocol import PodFile
from hermes_dust.constants import dust_host
from hermes_dust.http import DustApiError, DustRequestContext, dust_fetch, dust_json

_reads = 0


@dataclass(frozen=True)
class DustPod:
    s_id: str
    name: str


@dataclass(frozen=True)
class DustWorkspace:
    s_id: str
    name: str
    role: str = ""


def pod_relative_path(listed: str) -> str:
    """Strip the ``pod-<id>/`` prefix the listing adds."""
    if "/" not in listed:
        return listed
    first, rest = listed.split("/", 1)
    return rest if first.startswith("pod-") else listed


def scoped(space_id: str, path: str) -> str:
    """Address one pod file the way Dust's own file tools address it."""
    return "files/path/pod-" + space_id + "/" + "/".join(quote(part, safe="") for part in path.split("/"))


def _uncached(path: str) -> str:
    global _reads
    _reads += 1
    return f"{path}?_={int(time.time() * 1000):x}{_reads:x}"


def list_pods(dust_json_fn: Any) -> list[DustPod]:
    """List the workspace's pods (project spaces only)."""
    body = dust_json_fn("spaces")
    spaces = body.get("spaces", []) if isinstance(body, dict) else []
    return [
        DustPod(s_id=str(s.get("sId") or ""), name=str(s.get("name") or ""))
        for s in spaces
        if isinstance(s, dict) and s.get("kind") == "project"
    ]


def ensure_pod(dust_json_fn: Any, name: str) -> tuple[DustPod, bool]:
    """Find a pod by name (case-insensitive), creating it when missing."""
    wanted = name.strip().lower()
    for pod in list_pods(dust_json_fn):
        if pod.name.strip().lower() == wanted:
            return pod, False
    body = dust_json_fn(
        "spaces",
        method="POST",
        body={"name": name, "spaceKind": "project", "isRestricted": True},
    )
    space = body.get("space", {}) if isinstance(body, dict) else {}
    s_id = space.get("sId") if isinstance(space, dict) else None
    if not isinstance(s_id, str) or not s_id:
        raise ValueError(f"Dust could not create a pod named {name}.")
    return DustPod(s_id=s_id, name=str(space.get("name") or name)), True


def fetch_workspaces(dust_json_fn: Any) -> list[DustWorkspace]:
    """Workspaces the logged-in user belongs to (non-workspace-scoped route)."""
    body = dust_json_fn("me")
    user = body.get("user", {}) if isinstance(body, dict) else {}
    workspaces = user.get("workspaces", []) if isinstance(user, dict) else []
    return [
        DustWorkspace(s_id=str(w.get("sId") or ""), name=str(w.get("name") or ""), role=str(w.get("role") or ""))
        for w in workspaces
        if isinstance(w, dict)
    ]


def list_pod_files(dust_json_fn: Any, space_id: str) -> list[PodFile]:
    """List every file in a pod, directories excluded."""
    body = dust_json_fn(f"spaces/{space_id}/files")
    files = body.get("files", []) if isinstance(body, dict) else []
    result = []
    for entry in files:
        if not isinstance(entry, dict) or entry.get("isDirectory") is True:
            continue
        if not isinstance(entry.get("path"), str):
            continue
        result.append(
            PodFile(
                path=pod_relative_path(entry["path"]),
                size_bytes=int(entry.get("sizeBytes") or 0),
                modified_ms=int(entry.get("lastModifiedMs") or 0),
            )
        )
    return result


def download_pod_file(fetch_fn: Any, space_id: str, path: str) -> str:
    """Read one pod file as text (cache-busted)."""
    response = fetch_fn(_uncached(scoped(space_id, path)))
    status = response.get("status", 0)
    if status < 200 or status >= 300:
        raise DustApiError(status, "server", None, f"Could not read {path} from the pod ({status}).")
    body = response.get("body")
    if isinstance(body, bytes):
        return body.decode("utf-8")
    return str(body or "")


def upload_pod_file(fetch_fn: Any, space_id: str, path: str, contents: str) -> None:
    """Write one file into a pod (PUT creates intervening folders)."""
    response = fetch_fn(scoped(space_id, path), method="PUT", body=contents, raw_body=True)
    status = response.get("status", 0)
    if status < 200 or status >= 300:
        raise DustApiError(status, "server", None, f"Dust refused {path} with {status}.")


def delete_pod_file(fetch_fn: Any, space_id: str, path: str) -> None:
    """Remove one file from a pod (204 answers no body)."""
    response = fetch_fn(scoped(space_id, path), method="DELETE")
    status = response.get("status", 0)
    if status < 200 or status >= 300:
        raise DustApiError(status, "server", None, f"Could not delete {path} from the pod ({status}).")


def make_pod_ops(context_fn: Any, space_id: str) -> dict[str, Any]:
    """Bind pod file operations for one pod; ``context_fn`` builds request contexts."""

    def fetch_fn(path: str, method: str = "GET", body: Any = None, raw_body: bool = False) -> dict[str, Any]:
        from hermes_dust.http import DustResponse

        response: DustResponse = dust_fetch(context_fn(), path, method=method, body=body, raw_body=raw_body)
        raw_body_bytes: bytes | None = None
        raw = getattr(response, "raw", None)
        read = getattr(raw, "read", None)
        if callable(read):
            try:
                data = read()
                raw_body_bytes = data if isinstance(data, bytes) else str(data).encode("utf-8")
            except Exception:
                raw_body_bytes = None
        return {"status": response.status, "body": raw_body_bytes if raw_body_bytes is not None else response.body}

    def json_fn(path: str, method: str = "GET", body: Any = None) -> Any:
        return dust_json(context_fn(), path, method=method, body=body)

    return {
        "list": lambda: list_pod_files(json_fn, space_id),
        "read": lambda path: download_pod_file(fetch_fn, space_id, path),
        "write": lambda path, contents: upload_pod_file(fetch_fn, space_id, path, contents),
        "remove": lambda path: delete_pod_file(fetch_fn, space_id, path),
    }
