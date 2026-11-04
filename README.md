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

## Status: phase 1 (text transport)

| # | Scope | State |
| --- | --- | --- |
| 1 | Protocol port: endpoints, billing origin, HTTP transport, SSE framing, WorkOS device grant, conversation helpers, history digest, tag stripper | Done, 38 stdlib tests |
| 2 | Hermes provider profile (`dust`), provider-owned WorkOS auth, live model catalog, OpenAI-facade client | Scaffold done, needs live Hermes validation |
| 3 | Tool bridge (Hermes tools executed for Dust turns, approvals, pod/MCP exchange) | Roadmap — text-only for now, `supports_tools: False` |

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
  bridge_note.py       # Hermes-adapted turn note (phase 2 hook)
  client.py            # DustClient: OpenAI-facade over conversations
  provider.py          # ProviderProfile, auth handler, catalog, client
tests/                 # stdlib unittest suite, no network
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

## Tool bridge roadmap (phase 2)

opencode-dust bridges tools through a private pod the agent's Computer
sandbox mounts (`/files/pod`) and a bash helper that waits for opencode to
run each call. The Hermes equivalent needs a host-side exchange Hermes can
drive: either the same pod mechanism with Hermes executing calls via its
plugin context, or an MCP server Dust dials back (the approach
[dust-code-proxy](https://github.com/k9fr4n/dust-code-proxy) already uses
for Claude Code). `hermes_dust/bridge_note.py` is the seam: phase 1 tells
the agent to answer in-conversation, phase 2 will point it at the exchange.
