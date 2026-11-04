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
tests/                 # stdlib unittest suite, no network (76 tests)
opencode-reference/    # original opencode-dust sources (history)
```

## Requirements

- **Hermes Agent** with the documented external model-provider API.
- A **Dust account** in a workspace with the `models_picker` feature
  (every picked model is sent as a `modelSelection` override; without it
  Dust answers HTTP 403) and the **Computer** feature on.
- Python 3.11+. No runtime dependencies.

## Requirements

- **Hermes Agent** with the documented external model-provider API
  (`register_provider`, provider-owned auth, `create_client`).
- A **Dust account** in a workspace with:
  - the `models_picker` feature on — every picked model is sent as a
    `modelSelection` override, and without it Dust answers HTTP 403;
  - the **Computer** feature on (default) — the bridge's sandbox runs on it.
- Python 3.11+. No runtime dependencies (stdlib only).

## Install

### 1. Install the plugin

```sh
hermes plugins install k9fr4n/hermes-dust --enable
```

This clones the repo into Hermes' plugin directory
(`~/.hermes/plugins/hermes-dust` by default) and enables it. A running
Hermes process picks provider plugins up on its next provider lookup —
no restart needed for the lookup itself, though the Desktop app may need
one to refresh its pickers.

Manual alternative, useful to hack on it — note the `model-providers/`
directory: older Hermes cores only discover provider plugins there, while
newer ones also scan the flat install directory:

```sh
git clone https://github.com/k9fr4n/hermes-dust.git ~/.hermes/plugins/model-providers/dust
hermes plugins enable dust
```

Validate the install without signing in (same check catalog CI runs):

```sh
hermes plugins validate ~/.hermes/plugins/hermes-dust
python3 -m unittest discover -s ~/.hermes/plugins/hermes-dust/tests
```

The second command runs the plugin's own 76-test stdlib suite; it makes
no network calls.

> **Profiles.** Provider plugins resolve per Hermes profile. If you work
> with several profiles, install and sign in in each one that should use
> Dust: `hermes -p <profile> plugins install k9fr4n/hermes-dust --enable`.

### 2. Sign in to Dust

```sh
hermes auth add dust
```

This starts the plugin-owned WorkOS device flow (the only auth path —
there is deliberately no API key, which Dust would bill as programmatic
usage):

1. The command prints a URL and a user code.
2. Open the URL, approve, and type the code.
3. The plugin stores the grant (access + refresh token, region, workspace)
   in that profile's credential pool and prints the workspace it scoped to.

If you belong to several workspaces, the command lists them and stops —
re-run naming one:

```sh
hermes auth add dust --workspace <workspace-sId> --force
```

`--force` replaces the stored grant without asking. One workspace is
stored per Hermes profile; use a second profile (`hermes -p <other> …`)
for a second workspace.

Useful companions:

```sh
hermes auth status dust    # show the stored workspace and region
hermes auth refresh dust   # rotate the grant now (otherwise automatic)
hermes auth logout dust    # forget the grant
```

Tokens live 15 minutes and refresh themselves: every request mints from
the pool, a 401 retries once against a freshly minted token, and long
streams stay alive across rotations.

### 3. Pick a model

```sh
hermes model               # choose the dust provider, then a dust/… model
```

Model ids mirror the workspace catalog as `dust/<providerId>:<modelId>`:

```sh
hermes chat --provider dust --model dust/anthropic:claude-sonnet-5
```

- The list is your workspace's live catalog; when the fetch fails, the
  picker falls back to a curated list (`anthropic:claude-sonnet-5`,
  `anthropic:claude-opus-5`, `openai:gpt-5.6-sol`,
  `google_ai_studio:gemini-3.1-pro-preview`).
- Models with vision accept pasted images (uploaded once, attached as
  content fragments, deduplicated by bytes across replays).
- Reasoning models accept an effort suffix, weakest first —
  `dust/anthropic:claude-sonnet-5+high` sends `reasoningEffort: high`
  with the override. Without a suffix Dust applies the model's own
  default. Supported: `+none`, `+light`, `+medium`, `+high`.
- There is no `agent-default` entry: it would run whatever model the Dust
  agent is configured with, which Dust never reports, so it is not a model
  a human can pick knowingly.

### 4. First run

On the first Dust turn in a profile, the plugin creates a private pod
named `hermes-project` in the workspace and leaves the bridge helper in
it (`hs-tools/hs.sh`). The pod holds only the exchange, never your code.

During a turn the Dust agent works through its Computer sandbox (free
`bash`, `basic` + `freeUsage`): it leaves a JSON batch of tool calls or a
shell script in `hs-bash/<conversation>/`, Hermes executes it locally —
with Hermes' own cards, diffs and approval prompts — and the result lands
in the `.out` the sandbox call is waiting on. A request is therefore
billed as interactive usage plus the model's own credits; the file
round-trips themselves are free.

## Everyday use

- **Approvals** are Hermes' own: the agent cannot touch your machine
  without going through the same policy and prompts as any Hermes model.
  If an approval takes longer than the sandbox's 100 s wait, the agent
  ends its turn; the finished result is delivered as an automatic message
  on your next turn.
- **Interrupting** a session cancels Dust's generation and writes a
  cancelled result into the pod, so the sandbox never waits on a file
  that will never come.
- **History** follows the conversation: as long as Hermes' prompt extends
  the history Dust holds, only the new message is sent. Compaction,
  revert or edit opens a fresh conversation with the transcript replayed
  and the system prompt in front.
- **Images** go with the turn that carries them and are skipped with a
  warning when Dust refuses one — the words are still answered.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Unknown provider 'dust'` right after install | The plugin directory is not where your core scans for providers. Newer cores scan both `plugins/<name>/` and `plugins/model-providers/<name>/`; older ones only the latter. Move it and retry: `mv ~/.hermes/plugins/dust ~/.hermes/plugins/model-providers/dust` (replace `~/.hermes` with your `HERMES_HOME` — the install log shows the location). No reinstall needed; the next lookup picks it up. |
| `403` on every turn | The workspace lacks `models_picker`. Ask an admin, or pick no model until it is on — every offered model sends an override. |
| `auth add` lists workspaces and stops | You belong to several; re-run with `--workspace <sId>`. |
| `Not logged in to Dust` | `hermes auth add dust`; per profile — the CLI profile and the Desktop profile store separate grants. |
| Agent says it cannot reach the project | The workspace's **Computer** feature is off, or the sandbox mount is missing. Re-enable Computer; the helper path is `bash /files/pod/hs-tools/hs.sh <name>`. |
| A turn ends while work "is still running" | An approval outlasted the 100 s sandbox wait. Approve, then keep chatting: the result arrives as `[automatic message, …]` and the agent carries on. |
| `Dust stopped answering` after a stall | The event stream broke past its reopen budget. Retry the turn; if it persists, check Dust status — the plugin never guesses an ending. |
| `Unknown tool` for a bridged call | The agent invented a name outside the note's signatures. It reads the refusal and corrects itself next round; nothing ran. |
| Desktop picker lacks `dust` | The profile has no live grant (`hermes auth status dust`), or the app needs a restart to refresh pickers. |

## Uninstall

```sh
hermes plugins disable dust
```

then delete `~/.hermes/plugins/hermes-dust` (and `hermes auth logout dust`
to forget the grant). The exchange pod stays in the workspace — Dust has
no delete path the plugin can rely on — remove `hermes-project` by hand
if you want it gone.

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
