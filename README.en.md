# agent-base

[![CI](https://github.com/paopaonb666/agent-base/actions/workflows/ci.yml/badge.svg)](https://github.com/paopaonb666/agent-base/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

[简体中文](README.md) | **English**

**A production governance foundation on top of LangGraph.** Orchestration is
delegated to LangGraph (not reinvented here); the foundation fills the three
facilities missing between "demo" and "production" — **long-term memory,
cost governance, and event observability**. Business logic plugs in as
modules: five steps to onboard, zero changes to the foundation.

## Why this isn't yet another LangGraph wrapper

LangGraph solves "how to orchestrate agents"; this repo solves "how to dare
take that orchestration to production". Every row below maps to code that is
already implemented and tested in this repository — none of it is a roadmap:

| Production concern | Common status quo (left to the framework) | Built into agent-base | Details |
| --- | --- | --- | --- |
| Long-term memory | Memory choice locks you in (Mem0/Zep/Letta architectures are mutually incompatible) | Hybrid retrieval + formation pipeline + versioned audit; the `MemoryPort` protocol inverts the dependency — modules are not bound to any implementation | [Memory system (M6)](#memory-system-m6) |
| Cost governance | usage scattered everywhere, overspending discovered via the month-end bill | Metering on every LLM call (streaming included), daily/monthly budget circuit breaker (429), exposed via `/metrics` and `/health` | [Cost governance (T4)](#cost-governance-t4) |
| Event observability | black-box runs, failures impossible to trace | SSE event contract (step / delta / plan / tool), `request_id` end to end | [Event contract (SSE)](#event-contract-sse) |
| Module onboarding | business logic and framework bleed into each other | Explicit `AGENT_MODULES` manifest + fail-fast validation, five-step onboarding guide | See [Project structure](#project-structure) below |
| Tool governance | bare functions wired straight to the model | Shared tool pool: timeout wrapping + full audit + duplicate-name fail-fast; MCP servers plug in as external tool sources | See [Configuration](#configuration-env) below |

Engineering discipline underpins these claims: four quality gates
(lint / mypy strict / pytest coverage ≥85% / pip-audit), identical locally
and in CI, plus dependency floor pinning (langgraph ≥1.2.6, covering three
CVE fixes).

> **Scope boundary**: business auth, business storage, and the like do
> **not** belong to the foundation — they plug in as "modules" (rationale in
> [ADR-001](docs/adr/001-langgraph-as-runtime.md) and
> [ADR-002](docs/adr/002-module-contract.md)). The foundation only provides
> the runtime, the registration contract, configuration management, runnable
> entrypoints, and extension points.

## Current status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Engineering foundation (scaffold / toolchain / four CI gates / ADRs) | ✅ Done |
| 1 | Core assembly layer (config / contracts / registry / llm / bootstrap) + chat sample module + CLI entrypoint | ✅ Done |
| 2 | Observability (end-to-end request_id / structured logging / LangSmith & Langfuse toggles) | ✅ Done |
| 3 | Tool extension point (shared tool pool + timeout + exception normalization) + conversation state (checkpointer) | ✅ Done |
| 4 | FastAPI + SSE server entrypoint + supervisor multi-agent template | ✅ Done |
| 5 | Wrap-up + documentation loop (five-step onboarding in practice / coverage ≥85% / CircuitBreaker archive) | ✅ Done |

## Architecture

```
                         ┌────────────────────────────────────────────────┐
                         │              agent-base (foundation)           │
                         │                                                │
  CLI / SSE server ────▶ │  entrypoints   core                 extensions │
  (python -m / uvicorn)  │  ┌──────────┐  ┌────────────────┐  ┌────────┐  │
                         │  │ cli.py   │─▶│ bootstrap      │◀─│ memory │  │ ◀─ conversation state
                         │  │ server/  │  │ (assembly hub) │  │ collab │  │ ◀─ supervisor
                         │  └──────────┘  └───┬──┬──┬──┬───┘  │ events │  │ ◀─ SSE contract
                         │                    │  │  │  │      │ metrics│  │ ◀─ metrics
                         │                    ▼  ▼  ▼  ▼      │observ. │  │ ◀─ request_id
                         │               config contracts     └────────┘  │
                         │               registry llm tools               │
                         │                                                │
                         │  modules/ (business modules onboard in five    │
                         │  steps, no foundation code changes)            │
                         │  ┌──────┐ ┌──────┐ ┌──────┐                    │
                         │  │ chat │ │writer│ │hello │  … future modules  │
                         │  └──────┘ └──────┘ └──────┘                    │
                         └────────────────────────────────────────────────┘
                                       ▲ AgentModule contract (ADR-002)
                                       │
                                 LangGraph runtime (delegated, >=1.2.6)
```

Data flow: entrypoint (CLI / HTTP SSE) → `create_runtime` assembles per the
`AGENT_MODULES` contract → module graphs are mounted on the runtime and
executed; extension points (state / collaboration / events / metrics /
observability) are supplied uniformly by the foundation.

## Requirements

- Python ≥ 3.10 (3.12 recommended for development)
- Optional: Node.js + pnpm (only for the companion frontend agent-base-ui)

## Dependencies

Runtime dependencies (`pyproject.toml` [project.dependencies]):

| Dependency | Purpose | Version floor and rationale |
| --- | --- | --- |
| `langgraph` | Runtime (delegated, not reinvented) | `>=1.2.6`: fixes CVE-2025-68664 (deserialization injection), CVE-2025-67644 (SQLite checkpointer SQL injection), CVE-2026-34070 (path traversal); see ADR-001 |
| `langchain-core` / `langchain-openai` | Message primitives + OpenAI-compatible model client | `>=1.6`: lower bound of langchain-openai 1.x |
| `pydantic-settings` | Configuration as validation (fail-fast) | `>=2.2`: NoDecode support for comma-separated list fields (`AGENT_MODULES`) |
| `langgraph-checkpoint-sqlite` + `aiosqlite` | SQLite conversation-state backend | `>=2.0`: AsyncSqliteSaver serving astream |
| `langgraph-supervisor` | Supervisor multi-agent template | `>=0.0.31` |
| `fastapi` / `uvicorn` | HTTP + SSE server entrypoint | `>=0.115` / `>=0.30` |
| `pypdf` | Tool library parsing: PDF text extraction (prerequisite of M4) | `>=5.0`: pure Python and lightweight; PDF is the primary document-reading format, so it ships as a main dependency out of the box |
| `httpx` / `tzdata` | Tool library basics: HTTP client for the Tavily search engine / timezone database on Windows | `>=0.27` (promoted from dev dependency) / `>=2024.1` (bundled on POSIX) |
| `redis` | Shared governance-state layer (ADR-012): budget counters / nonce replay guard / distributed locks / injection snapshots / SSE event mirror | `>=5.0`: stable asyncio client with full type hints; `REDIS_ENABLED=false` degrades to in-process state with zero Redis usage |
| `python-multipart` | multipart/form parsing for the file-upload endpoint | `>=0.0.9`: officially required by FastAPI |

Optional extras:

- `[search]`: `ddgs` — the DuckDuckGo search engine. Enabling `web_search`
  without it and without a Tavily key fails fast at startup.
- `[doc]`: `python-docx` — DOCX parsing. Without it the docx format in
  tools/parsing is not registered (install via `pip install -e ".[doc]"`);
  other formats are unaffected.
- `[memory]`: `agent-base-memory` — the memory subsystem (standalone
  distribution since 0.3.0). For development, install straight from the
  working tree: `pip install -e packages/agent-base-memory`. The memory
  package is not yet on PyPI, so installing `agent-base[memory]` from PyPI
  currently yields only the foundation; the image and compose already bundle
  the package.

Development dependencies (`[dev]`): pytest / pytest-asyncio / pytest-cov /
ruff / mypy / pip-audit / httpx.
The security audit runs permanently in CI: `pip-audit --skip-editable`.
Database migrations (the `mysql` backend): Alembic + SQLAlchemy + PyMySQL,
installed via the `[migrate]` extra (the dev extra includes the same set;
the runtime image is built exactly with `.[migrate]`).
Conversation-state tables are versioned under [`alembic/`](alembic/); the
initial migration reuses langgraph's built-in migrations.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev,doc]"    # [doc] provides DOCX parsing; degrades gracefully without it
pip install -e packages/agent-base-memory   # memory subsystem (standalone package, since 0.3.0)
cp .env.example .env           # fill in LLM_API_KEY (consumed by config.py since phase 1)
pytest                         # core suite (integration marker excluded by default; no MySQL needed, see below)
```

One-command container startup (app + MySQL 8.0 + Redis 7.4, migrate before start):

```bash
cp .env.example .env           # fill in LLM_API_KEY / MEMORY_AUTH_SECRET
docker compose up --build      # http://localhost:8000/health
docker compose config --quiet  # validate compose syntax and variable interpolation only
```

> Running bare-metal (outside containers) without a local Redis? Set
> `REDIS_ENABLED=false` in `.env` — governance state falls back to in-process
> implementations (single-process semantics, identical to 0.3).
>
> **Tests and the coverage gate**: pytest runs the **core suite** by default
> (addopts in `pyproject.toml` includes `-m "not integration"`); the coverage
> gate (branch ≥85%) is computed on the core suite and spans both
> distributions — the foundation and the memory package
> (`--cov=agent_base --cov=agent_base_memory`) — so everything is green
> locally without MySQL or Redis. Integration tests carry the
> `@pytest.mark.integration` marker and run in exactly two places: CI's test
> job (with MySQL 8.0 and Redis 7.4 service containers, running the full set
> via `-m "integration or not integration"`) and locally when you pass the same
> marker explicitly and the connection is available — connection parameters
> come from the `MYSQL_*` and `REDIS_URL` environment variables
> (auto-skip when probing fails).
> CI also has a cross-platform test-core job backing
> "works on Windows / Python ≥3.10".

### Configuration (.env)

`.env.example` is the configuration contract (consumed by `core/config.py`);
key entries:

| Variable | Description |
| --- | --- |
| `LLM_API_KEY` | Required. API key for an OpenAI-compatible model (DeepSeek / Zhipu / custom all go through the same client). Refuses to start in production when missing |
| `LLM_PROVIDER` | `deepseek` \| `zhipu` \| `openai-compatible` (spelling check only; does not change the client) |
| `LLM_BASE_URL` / `LLM_MODEL` | Model service address and model name; switching providers only means changing these two |
| `AGENT_MODULES` | Comma-separated module manifest; order is assembly order (default `chat,writer`) |
| `AGENT_MODULE_PACKAGES` | Import prefixes of the packages containing modules, comma-separated, searched in order (default `agent_base.modules`). Add independently distributed module packages and they become discoverable; for same-named modules the earlier package wins |
| `CORS_ORIGINS` | Origins allowed to call the SSE endpoints cross-origin (default `http://localhost:3000`, i.e. agent-base-ui) |
| `CHECKPOINTER_BACKEND` | `sqlite` (default; conversations survive restarts) \| `memory` (zero dependencies, not persisted) \| `mysql` (MySQL persistence, with `CHECKPOINTER_MYSQL_*`) |
| `REDIS_ENABLED` / `REDIS_URL` | Shared governance-state layer (ADR-012, on by default): budget-gate counters / nonce replay guard / capture distributed locks / injection snapshots / SSE event mirroring shared across workers; **enabled means committed** (refuses to start when unreachable). Single-host deployments without Redis set `REDIS_ENABLED=false` to fall back to in-process state. `REDIS_KEY_PREFIX` (default `agent_base:`) isolates multiple apps on a shared instance |
| `TOOL_TIMEOUT_SECONDS` | Per-execution tool timeout (default 30s) |
| `TOOL_CALL_LOG_ENABLED` | Tool-call audit (M5): every call's args/result/duration persisted; the UI renders a call panel |
| `DOC_PARSE_MAX_INPUT_BYTES` / `DOC_PARSE_MAX_OUTPUT_CHARS` | Upload document parsing limits (default 10MB / 50k chars) |
| `TOOLKIT_ENABLED` | Built-in tools assembled into the shared pool (default `current_time,calculator,json_query`); `web_search` needs `SEARCH_*`; `python_repl` is exec-level and off by default |
| `MCP_ENABLED` / `MCP_SERVERS_JSON` | MCP tool source (needs `pip install -e ".[mcp]"`): enabling is a commitment — invalid config or unreachable servers fail fast **at startup**; tool names get a `{server}_` prefix and join the shared pool under the same timeout / audit / duplicate-name rules as built-in tools |
| `LOG_JSON` | `true` for structured JSON logs (recommended in production) |

> **Attachment formats**: documents (pdf/docx/txt/md) are text-extracted and
> injected into context; images (png/jpg/webp/gif) are verified by magic
> bytes, stored byte-exact, and reach vision models as multimodal content
> blocks (pair with a vision model).

Run a conversation (available since phase 1):

```bash
python -m agent_base --message "hello"    # single turn (requires LLM_API_KEY in .env)
python -m agent_base --module chat         # interactive multi-turn chat
python -m agent_base --module chat --thread-id <id>   # resume a previous session (phase 3)
python -m agent_base --module supervisor   # multi-agent session (phase 4; orchestrates every module registered in AGENT_MODULES)
python -m agent_base --version
```

Run the SSE server (available since phase 4):

```bash
uvicorn agent_base.entrypoints.server:app --reload

curl -N -X POST http://localhost:8000/v1/agents/chat/invoke \
  -H "Content-Type: application/json" -d '{"message": "hello"}'
# SSE events: ping → step(running/completed) → delta... → done(thread_id); full contract in "Event contract (SSE)" below
# For complex tasks add mode: "plan" (forced) / "auto" (heuristic); see the "Task modes" section
# Other endpoints: GET /health (per-component health; degraded does not crash); GET /metrics (Prometheus text, route-template label)
```

### Quick tour: one minute each for the three governance pillars

Each of the first three rows in the table above has a reproducible minimal
demo (source in [`examples/`](examples/)):

| What to see | How to run | Claim demonstrated |
| --- | --- | --- |
| Swap the memory backend wholesale | `python examples/custom_memory_port.py` (no API key needed) | [Long-term memory](#memory-system-m6) — modules only know the `MemoryPort` protocol; implementations are swapped at the composition root |
| Budget circuit breaker | `python examples/budget_gate_demo.py` (no API key needed) | [Cost governance](#cost-governance-t4) — 80% warning → over-limit 429 → health `blocked` |
| Print the event stream frame by frame | Start the server, then `python examples/sse_stream_demo.py` | [Event contract (SSE)](#event-contract-sse) — the contract is the frontend/backend decoupling surface |

## Event contract (SSE)

SSE is the only realtime channel between frontend and backend, and the
decoupling surface: the frontend (agent-base-ui) depends only on the event
format, not on the backend implementation. The contract lives in
`extensions/events.py` — a closed set of Pydantic models; the wire format is
validated and versionable, not ad-hoc strings:

```
event: delta
data: {"type":"delta","content":"he"}
```

Typical sequence of one invoke: `ping` (keep-alive heartbeat) → `step`
(node running/completed) → `delta`... (incremental tokens, interleaved with
`tool_call` and `plan`) → `done(thread_id)`; failures terminate with
`error`. All event types:

| type | When emitted | Payload essentials |
| --- | --- | --- |
| `ping` | Idle heartbeat, keeps intermediaries from dropping the stream | none |
| `step` | Graph node starts / ends | `name`, `status` (running/completed/error), `detail` |
| `delta` | Each chunk of streamed reply tokens | `content` |
| `tool_call` | One before and one after each tool execution, paired by `call_id` into a complete call panel | `name`, `phase` (start/end), `args`, `result`, `status` (ok/timeout/error), `duration_ms` |
| `plan` | Planner splits / advances / replans / completes | `status` (created/progress/replanned/done), `plan` — the **full** task list (the frontend replaces the whole table; no incremental merge needed) |
| `sources` | Reserved in the contract; the foundation never emits it | `sources[]` (title/url) — a future RAG module can publish citations without changing the encoder |
| `done` | Terminal success | `thread_id` (the handle for resuming the session) |
| `error` | Terminal failure | `message` (a human-readable, actionable summary) |

Failure troubleshooting path: the `error` event carries only a summary; the
full chain is already in server-side structured logs tagged with
`request_id` (spanning entrypoint / module / node) — use the summary to
locate the request_id and reconstruct the whole run from the logs; tool-call
details are additionally persisted in full via `TOOL_CALL_LOG_ENABLED`.

## Memory system (M6)

`packages/agent-base-memory/` (import name `agent_base_memory`) is a
self-built hybrid memory system (the landed form after surveying
mem0 / Letta / Memobase / Zep / LangMem). Since 0.3.0 it is a **standalone
distribution**: the foundation keeps only the `MemoryPort` /
`MemoryServicePort` contracts and the entry-point assembly slot (the
`agent_base.memory` group); the foundation source no longer imports the
memory package. `MEMORY_ENABLED=true` without the package installed prints
an install hint at startup and the `/v1/memory` endpoints are not registered
(404). The memory implementation can be swapped wholesale — business modules
depend only on `MemoryPort`
([swap example](examples/custom_memory_port.py)).

Storage follows `CHECKPOINTER_BACKEND` (memory/sqlite/mysql). The session
index `thread_index` and the plan snapshots `plan_snapshots` belong to the
foundation (`core/storage/`: still needed by the HTTP layer even when
`MEMORY_ENABLED=false`); the remaining six memory tables (`memories` /
`memory_blocks` / `session_summaries` / `doc_chunks` / `memory_ops` /
`memory_versions`) belong to the memory package: on MySQL they are managed
by the package's own alembic chain
(`version_table=alembic_version_memory`; migration `0001` creates tables
idempotently, so existing databases can upgrade directly), while sqlite is
still bootstrapped by the runtime (single DDL source in `store/ddl.py`):

| Capability | Description |
| --- | --- |
| Cross-session long-term memory | mem0-style two-phase formation: a background LLM extracts candidate facts → compares against existing memories and consolidates with ADD/UPDATE/DELETE/NOOP (dedup / correction / invalidation), fully audited into `memory_ops` |
| Resident memory blocks | Letta-style persona/human/custom blocks; the agent edits them itself via the `memory_update_block` tool, and they are injected into context every turn |
| User profile | Memobase-style structured JSON profile that merges and evolves with the conversation, injected as a whole (not part of similarity recall) |
| Rolling session summary | Once the message count crosses a threshold, a background job merges and updates it, used by short-term compression to replace over-budget old history |
| Document knowledge base | Uploaded PDF/DOCX/TXT are auto-chunked + embedded into `doc_chunks` (user-level knowledge assets that outlive sessions), retrieved by the `knowledge_search` tool |
| Hybrid retrieval | Vector cosine (default SiliconFlow BAAI/bge-m3, 1024-dim) + BM25 (Chinese bigrams) + time half-life + salience; falls back to a keyword path automatically when embeddings are unavailable |
| Context engineering | Every turn injects a "profile + memory blocks + summary + relevant memories" block; a foundation-level context engine (`core/context.py` + `core/graphs.py`) deduplicates injections and trims every module's model input to the token budget (tool pairs are never split apart), while the checkpointer keeps full history (recall semantics) |
| Agent tools | `memory_search` / `memory_save` / `memory_update_block` / `knowledge_search` join the shared tool pool (timeout + audit apply automatically) |

Management endpoints (registered by the memory package's `routes` entry
point; all scoped by the `X-User-Id` header, defaulting to `default`.
With only the foundation installed these paths return 404; with the package
installed but `MEMORY_ENABLED=false` they return 503):

```
POST   /v1/memory                 # manual memory write (best-effort embedding before persisting)
GET    /v1/memory?q=&module=&kind=  # with q → hybrid retrieval; otherwise browse by update time
PATCH  /v1/memory/{memory_id}     # partial update (changing content re-embeds)
DELETE /v1/memory/{memory_id}
GET    /v1/memory/blocks          # list resident memory blocks (?module= to scope)
PUT    /v1/memory/blocks/{label}  # manual block write (overwrites; version increments)
DELETE /v1/memory/blocks/{label}
GET    /v1/memory/profile         # the current user's structured profile
GET    /v1/memory/audit           # memory operation audit (extraction/consolidation/profile/summary/ingest…)
GET    /v1/memory/{id}/versions   # content version history (create/update/delete/restore snapshots + diffs)
POST   /v1/memory/{id}/versions/{vid}/restore   # restore to a given version (creates a new restore version)
```

Every content change (write/update/delete/restore, profile evolution
included) writes a version snapshot; deletes leave tombstones (last content
kept for audit). The BM25 index used by retrieval is cached in-process via
an LRU keyed on candidate-content fingerprints — the key changes as soon as
content changes, so no invalidation hooks are needed.

**Document preview and chunk visualization (M7)**: three read-only endpoints
support the frontend's "open an attachment to see its preview and how it was
chunked" — `GET …/files/{id}/preview` (metadata + extracted body, the same
source as what gets injected to the model), `GET …/files/{id}/chunks`
(chunk list: index / body / source offsets / whether embedded;
paragraph-packed chunks span multiple ranges, older data is null and the
frontend degrades), and `GET …/files/{id}/raw` (raw bytes: images served by
their stored mime, documents downloaded as attachments). Chunk boundaries
come from the ChunkSpan returned by `chunk_text` (each segment's character
range in the source), persisted in `doc_chunks.offsets_json`.

**Standalone knowledge-base page (M8)**: `GET /v1/knowledge/files` — a
user-level file list (chunk count per file, `X-User-Id` scoped), backing the
frontend `/knowledge` route's three tabs "document knowledge base /
long-term memory / user profile" (delivered in the frontend repo).

**Document knowledge base lifecycle**: uploads are chunked with paragraph
awareness (aligned to blank lines; tables/lists are never split mid-structure;
over-long single paragraphs fall back to a fixed window), and repeated
ingestion is idempotent; to undo a wrongly uploaded file use
`DELETE /v1/agents/{module}/files/{file_id}` — the file's raw rows and all
knowledge chunks are cascade-deleted, with ownership checked against the
user_id recorded at upload; after switching embedding models run
`python scripts/memory_backfill_embeddings.py` to backfill vectors for both
memories and knowledge chunks.

**Identity and security (S1/P0-5)**: identity resolution has exactly one
path — a pluggable `AuthBackend` (default `HmacHeaderAuth`: the `X-User-Id`
header + `X-User-Sig` HMAC verification); `create_app(auth_backend=...)`
swaps in a gateway's real identity system. In production (`ENV=production`)
with memory enabled, `MEMORY_AUTH_SECRET` **must** be configured, and every
user-scoped request must carry
`X-User-Sig = HMAC-SHA256(user_id\ntimestamp\nnonce, secret)`
(timestamp in unix seconds; nonce optional but non-repeating within the
window), otherwise 401. Signature generation:
`python scripts/memory_user_sig.py --user-id alice --secret <secret>`.
Replay protection: the timestamp must fall within
`MEMORY_AUTH_REPLAY_WINDOW_SECONDS` (default 300s); requests carrying a
nonce that are replayed within the window get 401; setting the window to 0
reverts to the old scheme covering only user_id (not recommended). Browsers
never hold the secret — for public multi-user deployments, use a server-side
proxy or a real identity system. Known limitation: the nonce replay cache is
single-process (multi-worker deployments need gateway-side replay
protection).

**Session thread scoping (S1)**: at invoke time the thread owner is written
to the `thread_index` table (storage follows `CHECKPOINTER_BACKEND`,
alembic 0008; the index is present even when `MEMORY_ENABLED=false`). Thread
lists return only the **current user's** threads; history / delete /
tool-audit endpoints verify ownership (masked as 404). Historical threads
from before the upgrade (no index row) are considered inaccessible;
attachments referencing another user's file_id are rejected at invoke (400).
Memory deletion has two levels: `PATCH /v1/memory/{id}` setting
`status=archived` is a soft delete (data kept, excluded from recall), while
`DELETE` is a hard delete; after switching embedding models/dimensions run
`python scripts/memory_backfill_embeddings.py` to backfill vectors for
historical memories.

**Capacity contract (H2)**: the current retrieval implementation is "fetch
all candidates + in-memory hybrid scoring", with a per-user cap of 2000
memory candidates (same order for knowledge chunks) — hitting the cap logs a
WARNING and older memories simply don't participate in recall rather than
being silently truncated. Larger scale needs sharding or a candidate-pushdown
index (sqlite-vec / FTS5 / a server-side vector store); see
`docs/design-review.md` H2.

**CLI vs server capability difference**: the CLI (`python -m agent_base`)
targets single-machine personal use — no memory context injection, no
formation pipeline, no attachments, no multi-user scoping; those
capabilities are provided only through the HTTP entrypoint. Metrics are
single-process; multi-worker deployments aggregate at the collection side.

Key configuration (full list in `.env.example`'s "memory system" section):
`MEMORY_ENABLED` master switch;
`MEMORY_EMBEDDING_API_KEY` (OpenAI-compatible `/embeddings`; empty falls
back to keyword retrieval);
`MEMORY_CAPTURE_ENABLED` / `MEMORY_CAPTURE_EVERY_TURNS` (formation
frequency);
`MEMORY_CONTEXT_MAX_CHARS` / `MEMORY_CONTEXT_MAX_TOKENS` (injection and
model-input budgets).
Metrics: `/metrics`'s `memory_operations_total{op,outcome}`; health:
`/health`'s `memory` component (absent when disabled — not counted as
degraded).

## Task modes (M9)

With `AGENT_MODULES=chat,planner` enabling the planner module, the invoke
request body may carry a `mode` field for difficulty triage:

| mode | Behavior |
| --- | --- |
| `chat` (default) | Requests the module's own graph; behavior identical to before |
| `plan` | Forces the planner graph (plan → execute → check →(replan)* → synthesize); **503 without degradation** if planner is not enabled |
| `auto` | Heuristic: message >200 chars or with attachments → plan; silently degrades to chat when planner is absent |

**Graph selection and thread namespaces are decoupled**: `mode` only decides
which graph to call; the thread is always
`{request module}:{thread_id}` — chat turns and plan turns within one
session share continuous history, and thread ownership, memory `agent_id`,
and attachment bindings are unaffected.

Planner execution semantics: the LLM splits the goal into
≤`PLANNER_MAX_SUBTASKS` subtasks, executed one by one in an **inline ReAct
loop** (per-subtask tool-round cap `PLANNER_SUBTASK_TOOL_ROUNDS`); subtask
failure (rounds exhausted / empty-output gate) triggers replanning (budget
`PLANNER_MAX_REPLANS`, completed results kept); once the budget is exhausted
a best-effort synthesis runs, honestly marking unfinished items.

The SSE event set gains `plan` (the full task list; the frontend replaces
the whole table):

```json
{"type": "plan", "status": "created",
 "plan": [{"id": 1, "goal": "check the weather", "status": "pending"}],
 "detail": "split into 1 subtask"}
```

`status` is one of `created / progress / replanned / done`. Companion
read-only endpoint
`GET /v1/agents/{module}/threads/{thread_id}/plan` returns
`{tasks, cursor, replans}` (503 when planner is not enabled; threads that
never ran a plan return empty defaults).

Known limitations:

- **Cross-turn plan continuation (M10)**: plan snapshots are persisted in a
  dedicated table (`plan_snapshots`, alembic 0009 / sqlite runtime
  bootstrap); the planner upserts at every change point
  (split/advance/replan/complete) and the `GET …/plan` endpoint reads the
  table — chat turns interleaved no longer reset the plan. A new plan turn
  still re-splits for the current goal (continuing an old plan is future
  incremental work);
- Frontend PlanEvent rendering belongs to the agent-base-ui repo; this repo
  provides only the contract and the endpoints;
- The worst-case graph step count for planner-heavy use is ~34, above the
  default `AGENT_RECURSION_LIMIT=25` — raise it in .env as needed.

## Cost governance (T4)

The token-cost foundation for a personal full-scale document-QA setup:
query triage + model tiering + nightly batch + budget circuit breaker. In
the target shape (Tier 0 takes ~60%, the fast tier ~30%, the primary model
only handles adjudication and polish), the monthly bill stays in
single-digit CNY (2026 domestic model pricing).

**Query triage (three tiers)**:

| Tier | Path | LLM | When to use |
| --- | --- | --- | --- |
| Tier 0 | `GET /v1/knowledge/search?q=` | Zero | Lookup questions ("what's the waiting period in that policy") — returns hit chunks directly, fields matching the M7 chunk view |
| Tier 1 | invoke + `"profile": "fast"` | Fast | Light synthesis grounded in chunks |
| Tier 2 | invoke (default) | Primary | Deep dives / synthesis / adjudication |

`profile` and `mode` are orthogonal (mode picks the graph, profile picks the
graph's models); passing `profile: "fast"` while `LLM_FAST_*` is unconfigured
returns an explicit 400 — never a silent downgrade. The fast view shares the
checkpointer with the primary — thread history is continuous across tiers.

**Model tiering**: `LLM_FAST_*` (free/cheap models, e.g. the Zhipu glm flash
series — free long-term but very low concurrency) takes the "simple, high
volume, repetitive" calls; `MEMORY_PIPELINE_PROFILE=split` (default) routes
the formation pipeline's extraction/profile/summary through the fast tier
(`ResilientLLM` semaphore-throttled with retries, falling back to the
primary once exhausted), while consolidation adjudication always uses the
primary (destructive decision steps never downgrade); `all_fast` is an
explicit override.

**Nightly batch**: `MEMORY_CAPTURE_ENABLED=false` +
`MEMORY_SUMMARY_ENABLED=false` turn off the per-turn pipeline's LLM
consumption; run once a day via cron / Task Scheduler:

```bash
python scripts/memory_nightly_capture.py            # force bypasses gating; batch extract+consolidate
python scripts/kb_ingest.py --dir path/to/docs --module chat --user-id alice --watch
```

`kb_ingest.py` bulk-ingests a local directory (recursive;
pdf/docx/txt/md) into the knowledge base: it reuses the existing upload
endpoints (parsing/chunking/embedding/ownership all shared), is idempotent
via a sha256 state file (re-runs skip), and `--watch` polls for increments.
Note that server-side ingestion is a background task after upload;
retrieval visibility lags by a second or two.

**Metering and budget**: with `COST_ENABLED=true`, `build_llm` attaches a
usage-collection callback to every client (streaming `stream_usage`
included) and prices calls against the `COST_PRICES_JSON` price table
(CNY per million tokens, manually maintained); totals land in the
`cost_usage` table (following the checkpointer backend; sqlite keeps a
standalone sqlite ledger), and the daily/monthly accumulators are restored
from the ledger after a restart. When `COST_DAILY_CNY` / `COST_MONTHLY_CNY`
is exceeded: `COST_ACTION=block` makes invoke return 429 (only new LLM
consumption is blocked; retrieval / history / memory management endpoints
are unaffected), `warn` only alerts and passes; a one-time alert fires at
the 80% threshold. Observability: `/metrics`'s
`llm_tokens_total{model,profile,kind}` and `llm_cost_cny_total`;
`/health`'s `cost` component (`ok|warn|blocked`; absent when disabled — not
counted as degraded).

**Known limitations**: metering and budgeting are single-process
(multi-worker deployments aggregate at the collection side, same as
/metrics); prices are maintained by hand; the memory backend's cost ledger
is in-memory (reset on restart; fail-safe direction: it only under-counts
and never falsely trips — the sqlite/mysql backends both persist and
restore); the CLI doesn't benefit (the CLI never injects memory or
multi-user anyway).

## Database migrations (mysql backend)

Schema is versioned by **Alembic** in two independent chains (same database,
same connection parameters):

| Chain | Version table | Covers |
| --- | --- | --- |
| Foundation | `alembic_version` | checkpointer (initial migration reuses langgraph's built-in migrations), tool audit, uploaded files, thread index, plan snapshots, cost ledger |
| Memory package | `alembic_version_memory` | the six memory tables (shipped with `agent-base-memory`) |

```bash
# Foundation chain (repo root)
python -m alembic upgrade head

# Database whose tables were already created by langgraph setup(): align versions only, no DDL re-run
python -m alembic stamp head

# Memory chain (install the memory package first; the first migration is idempotent, 0.2.0 databases need no stamp)
python -m alembic -c packages/agent-base-memory/alembic.ini upgrade head

# Connection parameter precedence: AGENT_BASE_DB_* env vars > CHECKPOINTER_MYSQL_* in .env
# Example: AGENT_BASE_DB_NAME=mydb AGENT_BASE_DB_PASSWORD=*** python -m alembic upgrade head
```

Migration directories: [`alembic/`](alembic/) (foundation chain:
`alembic.ini`, `env.py`, `versions/`) and
[`packages/agent-base-memory/alembic/`](packages/agent-base-memory/alembic/)
(memory chain). Note: the foundation migrations track progress by reading
the current max version in `checkpoint_migrations`, consistent with
`AIOMySQLSaver.setup()`'s idempotent logic — the two coexist without
conflict; the memory chain only looks at its own
`alembic_version_memory`. Inside containers, both chains run sequentially
in the compose start command.

## Pairing with the agent-base-ui frontend

The companion web UI lives in a separate repo, `agent-base-ui` (clone it as
a sibling of this repo):

```bash
# 1. Start this backend (default port 8000)
uvicorn agent_base.entrypoints.server:app --reload

# 2. In another terminal, start the frontend (see agent-base-ui/README.md for details)
cd ../agent-base-ui
pnpm install
pnpm dev        # http://localhost:3000
```

In the frontend settings page, enter `http://localhost:8000` and the target
module (chat / writer / supervisor) to start chatting. The two sides are
decoupled through the SSE contract; the frontend does not depend on backend
implementation details. (For the full experience, configure `LLM_API_KEY`
in `.env` first.)

## Quality gates (four, identical locally and in CI)

```bash
ruff check src tests examples packages && ruff format --check src tests examples packages   # 1. lint + format
mypy                                                     # 2. type check (strict)
pytest                                                   # 3. tests (built-in branch coverage ≥85% gate)
pip-audit --skip-editable                               # 4. dependency security audit
```

## Project structure

```
agent-base/
├── pyproject.toml            # dependencies + packaging metadata + toolchain config (ruff / mypy / pytest + coverage ≥85%)
├── .env.example              # configuration contract (consumed by core/config.py)
├── LICENSE / NOTICE          # Apache-2.0 full text and copyright notice
├── CONTRIBUTING.md           # contribution process and the four quality gates
├── SECURITY.md               # vulnerability reporting path and security boundaries
├── CHANGELOG.md              # Keep a Changelog style change log
├── Dockerfile                # python:3.12-slim multi-stage image (non-root 10001, HEALTHCHECK /health)
├── docker-compose.yml        # app + MySQL 8.0 single-machine deployment (healthchecks / named volumes / pre-start migrations)
├── .dockerignore
├── src/agent_base/
│   ├── core/                # config (nested sections) / contracts (incl. MemoryPort) / registry / llm
│   │                        #           · bootstrap / tools (tool pool) / context (context engine) / graphs
│   │                        #           · storage (thread_index + plan_snapshots, foundation-owned)
│   │                        #           · memory_plugin (entry-point discovery & assembly) / memory_scope
│   ├── extensions/          # extension points: observability (request_id + logging) · memory (checkpointer)
│   │                        #           · collab (supervisor) · events (SSE contract) · metrics
│   │                        #           · toollog (tool audit) · filestore (attachment storage) · mysql57
│   ├── modules/chat/        # sample module (graph + module + tools; doubles as the onboarding template)
│   ├── modules/writer/      # second sample module (for supervisor orchestration demos)
│   ├── modules/hello/       # the five-step guide's real landed instance (verified by tests/test_hello_module.py)
│   └── entrypoints/
│       ├── cli.py           # CLI entrypoint (--module / --message / --thread-id / --version)
│       └── server/          # FastAPI + SSE server package (split in H1): app assembly + auth/deps/
│                            #           serializers/sse/background + routes/{agents,files,system}
├── packages/agent-base-memory/  # memory subsystem standalone package (import name agent_base_memory)
│   ├── src/agent_base_memory/   # store/ (models/protocol/ddl/three backends) · embeddings · retrieval
│   │                            # · pipeline · textkit · context · tools · service · api · plugin
│   └── alembic/                 # memory table migration chain (version_table=alembic_version_memory)
├── alembic/                 # foundation migration chain (0004/0005/0007 memory-table migrations left as no-op stubs for old databases)
├── tests/                   # config / registry / llm / chat / cli / smoke / observability
│                            # / memory / tools / events / server / collab / hello
│                            # / bootstrap / metrics
├── docs/                    # public docs (published with the repo)
│   ├── adr/                 # ADR-001 runtime / ADR-002 module contract / ADR-003 memory package split
│   ├── module-guide.md      # five-step onboarding guide + checklist (incl. standalone module packages)
│   ├── deployment.md        # deployment shapes / env vars / single-process scope / dual migration chains / security / containers
│   ├── memory-package-split.md  # split plan, migration compatibility and rollback
│   └── design-review.md     # known boundaries H2–H6 and quality gates
└── .github/workflows/ci.yml # CI: lint → typecheck → test (MySQL 8.0) → audit
                            #     → memory-package (wheel + entry points) → docker
```

## Design decisions

Architecture decision and boundary docs ship with the repo — read them
right after cloning:

- [ADR-001: LangGraph as the foundation runtime](docs/adr/001-langgraph-as-runtime.md) — alternatives compared, CVE version floors, consequences and constraints
- [ADR-002: explicit manifest + fail-fast module onboarding contract](docs/adr/002-module-contract.md) — why no auto-discovery, validation rules and costs
- [Module onboarding guide](docs/module-guide.md) — five steps and a checklist
- [Deployment guide](docs/deployment.md) — three shapes, env vars, single-process scope, migrations and security
- [Design boundaries and quality gates](docs/design-review.md) — known boundaries H2–H6

Positioning and boundaries are grounded in ADR-001/002; the external
narrative (the differentiation table at the top of this README) stays
consistent with them — the foundation doesn't do orchestration; it does
governance on top of orchestration.

## License

This project is licensed under the [Apache License 2.0](LICENSE); copyright
and attribution notices are in [NOTICE](NOTICE). Submitting a contribution
means you agree to license your contribution under the same license.

## Contributing

Issues and PRs are welcome. Before starting, read
[CONTRIBUTING.md](CONTRIBUTING.md): all four quality gates must be green;
contract-first updates, security floors, and doc sync are hard requirements.
Security issues: report privately per [SECURITY.md](SECURITY.md), don't
open a public issue.
