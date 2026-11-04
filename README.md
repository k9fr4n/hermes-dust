# hermes-dust

Dust workspace agents as a **Hermes Agent** model provider, ported from
[opencode-dust](https://github.com/jphetphoumy/opencode-dust) (opencode v2
plugin, TypeScript) to a Hermes `model-provider` plugin (Python, stdlib
only). The Dust agent works on your machine through Hermes' own tool loop;
every turn is billed as interactive user usage.

Cloned from opencode-dust — the original TypeScript sources are preserved
under [`opencode-reference/`](opencode-reference/) — then ported to the
Hermes provider API documented in
[Hermes Model Provider Plugins](https://hermes-agent.nousresearch.com/docs/developer-guide/model-provider-plugin).

## Status: phase 2 (pod/sandbox tool bridge)

| # | Scope | State |
| --- | --- | --- |
| 1 | Protocol port: endpoints, billing origin, HTTP transport, SSE framing, WorkOS device grant, conversation helpers, history digest, tag stripper | Done, tested |
| 2 | Hermes provider profile (`dust`), provider-owned WorkOS auth, live model catalog, OpenAI-facade client | Done, needs live Hermes validation |
| 3 | Tool bridge, same pod/sandbox mechanism as opencode-dust | Done, needs live validation |

`hermes_dust/bridge/` ports the whole exchange: request parsing
(`protocol.py`), pod filesystem (`pod.py`), per-conversation sessions
(`session.py`). While a Dust turn streams, the client polls the pod
(`hs-bash/<conversation>/`), emits each request as one OpenAI `tool_calls`
round — a JSON batch as Hermes tool calls, a shell script as one `terminal`
call, one call at a time, requests in order — then writes the `.out` the
sandbox helper waits on once Hermes reports the results. A `.detached`
request (helper gave up after 100 s) is delivered as an automatic message
on the next turn; interrupts cancel Dust's generation and write a cancelled
`.out` instead of stranding the agent.

## Layout

```
plugin.yaml            # kind: model-provider, installed via hermes plugins install
__init__.py            # provider discovery: registers the dust profile
hermes_dust/
  constants.py         # Dust/WorkOS endpoints, cli origin (billing)
  http.py              # transport, error classification, 401 retry
  sse.py               # text/event-stream framing
  models.py            # catalog codec + Hermes capabilities
  workos.py            # device-authorization grant (only auth path)
  conversations.py     # open/continue/read conversations
  history.py           # history digest → conversation routing
  tag_stripper.py      # remove <thinking>/<response> delimiters
  bridge_note.py       # Hermes turn note: helper usage, batching rules
  bridge/
    protocol.py        # exchange shapes: requests, .out/.detached, helper script
    pod.py             # pod filesystem + pod management (cache-busted reads)
    session.py         # per-conversation claim/emit/outcome/delivery state
  client.py            # DustClient: OpenAI facade + SSE/pod interleaving
  provider.py          # ProviderProfile, auth handler, catalog, client
tests/                 # stdlib unittest suite, no network (60 tests)
opencode-reference/    # original opencode-dust sources (history)
```

## Requirements

- **Hermes Agent** with the documented external model-provider API.
- A **Dust account** in a workspace with the `models_picker` feature
  (every picked model is sent as a `modelSelection` override; without it
  Dust answers HTTP 403) and the **Computer** feature on.
- Python 3.11+. No runtime dependencies.

## Install

```sh
hermes plugins install k9fr4n/hermes-dust --enable
hermes auth add dust        # device code, approve in browser, pick workspace
hermes model                # choose the dust provider and a dust/… model
```

Validate without Hermes core (protocol only):

```sh
python3 -m unittest discover -s tests
```

## Billing: the `cli` origin

Every message is stamped `origin: "cli"` with `User-Agent: Dust CLI` under a
WorkOS **user** grant. Dust bills the turn as interactive user usage only
with that combination. Neither can be overridden, and there is no API-key
path — on purpose.

## How a call works

Dust has no chat-completions endpoint, so each Hermes LLM call sends one
user message to a Dust conversation and streams the agent's answer back:
chain of thought as reasoning, answer as text. Hermes tools are never sent
to Dust; the bridge note goes in front of every turn instead.

An LLM call carries no session id, so a conversation is found again by a
digest of the history Dust already holds. When the prompt extends that
history, only the new user message is sent. Anything else (compaction,
revert, edit) opens a new conversation with the transcript replayed and the
system prompt in front.

## Tool bridge (same mechanism as opencode-dust)

The Dust agent runs in its Computer sandbox, where the pod is mounted at
`/files/pod`, and reaches the developer's machine through the helper the
plugin leaves in the pod (`hs-tools/hs.sh`, rewritten on every connect):

```sh
bash /files/pod/hs-tools/hs.sh step1 <<'EOF'
{"calls":[{"tool":"read_file","arguments":{"path":"src/a.ts"}}]}
EOF
```

One pod per workspace (`hermes-<folder>`, `hermes-project` when the host
does not report a project directory); each conversation leaves its
requests under its own `hs-bash/<conversation>/` directory. The client
polls once a second while the turn streams, claims each request exactly
once (in-memory claim set — the listing goes through Cloudflare's cache),
and emits one Hermes tool call per round so batches keep their order and
stop-on-error survives. Hermes executes with its own approvals; the client
formats the `.out` and resumes the same Dust event stream from its cursor.
Server-side tool approvals are auto-approved; `web_search`/`web_extract`
stay with Dust's own tools and are never bridged.
