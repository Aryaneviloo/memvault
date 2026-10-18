
<div align="center">

<h1>MemVault</h1>

<p><strong>Persistent, semantic memory for AI agents.</strong><br>
SQLite or Postgres · MCP server included · 28 ms retrieval at 5,000 memories (CPU only)</p>

<p>
  <a href="https://github.com/Aryaneviloo/memvault/actions/workflows/ci.yml">
    <img src="https://github.com/Aryaneviloo/memvault/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/eviloomemvault/">
    <img src="https://img.shields.io/pypi/v/eviloomemvault?color=blue" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/eviloomemvault/">
    <img src="https://img.shields.io/pypi/pyversions/eviloomemvault" alt="Python">
  </a>
  <a href="https://github.com/Aryaneviloo/memvault/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  </a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> ·
  <a href="#why-memvault">Why MemVault</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="https://github.com/Aryaneviloo/memvault/blob/main/docs/architecture.md">Architecture</a> ·
  <a href="https://github.com/Aryaneviloo/memvault/blob/main/docs/contributing.md">Contributing</a>
</p>

</div>

---

Most AI applications are stateless. They forget users between sessions, lose
context mid-conversation, and treat every interaction as if it never happened.

**MemVault is the memory layer that fixes this.** Give any agent or
MCP-compatible client persistent, semantically searchable memory. Results are
ranked by similarity, recency, importance, and access frequency, and duplicates
are consolidated automatically.

```python
from memvault import MemVault

mc = MemVault()
mc.remember("User prefers Python over JavaScript", user_id="alice")

results = mc.recall("programming language preferences", user_id="alice")
# → [0.82] User prefers Python over JavaScript
```

## Why MemVault

- **Meaning, not keywords.** Local BGE embeddings find the right memory even
  when the wording differs.
- **Memory that behaves like memory.** Unused memories decay, used ones
  strengthen, near-duplicates merge.
- **Runs anywhere.** Zero-setup SQLite for development, PostgreSQL for
  production, in-memory for tests. All three share one interface and one test suite.
- **Works with your stack.** Python SDK, REST API, CLI, and an MCP server for
  Claude Desktop and Cursor.
- **Fast and flat.** Retrieval latency barely moves from 100 to 5,000 memories.

## Quick Start

### Install

```bash
pip install "eviloomemvault[local]"
```

### Use as a library

```python
from memvault import MemVault, MemoryType

mc = MemVault()

# Store a memory
mc.remember(
    "User prefers dark mode and concise answers",
    user_id="alice",
    memory_type=MemoryType.SEMANTIC,
    importance=0.8,
)

# Retrieve by meaning
results = mc.recall("display preferences", user_id="alice")
for r in results:
    print(f"[{r.final_score:.3f}] {r.item.content}")

# Auto-extract memorable facts from a conversation
mc.ingest(
    messages=[
        {"role": "user", "content": "I've been using Rust for systems work."},
        {"role": "user", "content": "Python is my go-to for AI projects."},
    ],
    user_id="alice",
)
```

### Use with Claude Desktop or Cursor (MCP)

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memvault": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "memvault.mcp_server.server"],
      "env": {
        "MEMVAULT_DB": "/path/to/memvault.db",
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": "/path/to/memvault/src"
      }
    }
  }
}
```

Restart Claude Desktop. It now has 6 memory tools and remembers things across
conversations using your local database.

### Use as a service (REST API)

```bash
# Docker (recommended)
docker compose -f docker/docker-compose.yml up

# Or directly
uvicorn memvault.api.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "a1", "user_id": "alice", "content": "Prefers Python", "type": "semantic"}'

curl -X POST http://localhost:8000/memories/search \
  -H "Content-Type: application/json" \
  -d '{"text": "programming preferences", "user_id": "alice"}'
```

Interactive docs at `http://localhost:8000/docs`.

### Use from the command line

```bash
memvault remember "User prefers dark mode" --user alice
memvault recall "display preferences" --user alice
memvault consolidate --user alice
memvault doctor
```

## Installation options

```bash
pip install eviloomemvault                 # core: REST API + CLI
pip install "eviloomemvault[local]"        # + local BGE embeddings (recommended)
pip install "eviloomemvault[postgres]"     # + PostgreSQL backend
```

**Requires:** Python 3.10+

## Benchmarks

Measured on CPU only (no GPU), BGE-small embeddings, Python 3.10.
Full results: [docs/benchmark_results.md](https://github.com/Aryaneviloo/memvault/blob/main/docs/benchmark_results.md)

| Operation | Median | P95 |
|-----------|--------|-----|
| Embed 1 sentence (BGE-small, CPU) | 29 ms | 31 ms |
| Insert with embedding | 25 ms | 30 ms |
| Insert without embedding | <1 ms | <1 ms |
| Retrieval, 100 memories | 24 ms | 28 ms |
| Retrieval, 1,000 memories | 27 ms | 34 ms |
| Retrieval, 5,000 memories | 28 ms | 33 ms |
| Rule-based ingestion (10 messages) | <1 ms | <1 ms |

**Key result:** latency barely changes from 100 to 5,000 memories, because the
candidate fetch is `O(limit)`, not `O(n)`.

Reproduce: `python benchmarks/benchmark_suits.py`

## Memory types

| Type | When to use | Example |
|------|-------------|---------|
| `semantic` | Stable facts and preferences | "User prefers Python over Java" |
| `episodic` | Events at a point in time | "User completed onboarding on Jan 5" |
| `procedural` | Repeatable workflows | "Always run tests before committing" |
| `working` | Short-term session context | "Currently debugging the auth module" |
| `consolidated` | Auto-generated summaries | Created by `mc.consolidate()` |

## Storage backends

| Backend | Use case | Setup |
|---------|----------|-------|
| `SQLiteStorage` | Local dev, single-process production | Zero setup |
| `PostgresStorage` | Production, multi-process | Postgres instance |
| `InMemoryStorage` | Tests, experimentation | Zero setup |

Swap with one line:

```python
from memvault import MemVault
from memvault.storage.postgres import PostgresStorage

mc = MemVault(storage=PostgresStorage("postgresql://user:pass@host/db"))
```

Every backend passes the same contract test suite, so behaviour is identical
across all three.

## Ingestion

Extract memorable facts from conversations without deciding manually what to remember:

```python
# Rule-based: zero dependencies, works offline, <1 ms
mc.ingest(messages, user_id="alice")

# LLM-powered: higher quality, requires an API key
from memvault.ingestion.anthropic_extractor import AnthropicExtractor
mc.ingest(messages, user_id="alice", extractor=AnthropicExtractor())
```

<details>
<summary><strong>How retrieval scoring works</strong></summary>

```
final_score = (similarity_weight × cosine_similarity)
            + (relevance_weight  × relevance_score)

relevance_score = 0.4 × recency
                + 0.4 × importance
                + 0.2 × frequency
```

Recency decays with a configurable half-life (default 7 days), and frequency
saturates so heavily accessed memories don't dominate. All weights are
configurable via `RetrievalConfig` and `ScoringWeights`.

</details>

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                     │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     MemVault Facade     │
              │  remember · recall      │
              │  ingest · consolidate   │
              └────────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼────────┐
│  Intelligence │  │    Storage    │  │  Embeddings   │
│  retrieval    │  │  SQLite       │  │  BGE-small    │
│  scoring      │  │  PostgreSQL   │  │  (pluggable)  │
│  decay        │  │  In-Memory    │  └───────────────┘
│  ingestion    │  │  (pluggable)  │
└───────────────┘  └───────────────┘

Service interfaces: REST API · CLI · MCP Server · Python SDK
```

More detail in [docs/architecture.md](https://github.com/Aryaneviloo/memvault/blob/main/docs/architecture.md).

## Project structure

```
src/memvault/
├── core/               # Models, scoring, retrieval, consolidation
├── storage/            # SQLite, PostgreSQL, in-memory backends
├── embeddings/         # BGE-small and pluggable embedding providers
├── ingestion/          # Rule-based and LLM-powered fact extraction
├── integrations/       # Framework adapters
├── api/                # FastAPI REST service
├── cli/                # Typer CLI
├── mcp_server/         # MCP server for Claude Desktop / Cursor
└── observability/      # Structured logging and metrics

benchmarks/             # Performance benchmark suite
docs/                   # Architecture, getting started, benchmarks
```

## Contributing

Contributions are welcome: bug fixes, new backends, embedding providers, and
performance work especially. Good first issues are labelled on the
[issue tracker](https://github.com/Aryaneviloo/memvault/issues).

```bash
git clone https://github.com/Aryaneviloo/memvault.git
cd memvault
pip install -e ".[local,dev]"
pytest
```

See [CONTRIBUTING](https://github.com/Aryaneviloo/memvault/blob/main/docs/contributing.md) for details.

## License

Apache 2.0. See [LICENSE](https://github.com/Aryaneviloo/memvault/blob/main/LICENSE).