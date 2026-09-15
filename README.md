<div align="center">

<h1>MemVault</h1>

<p><strong>Production-grade memory infrastructure for AI agents.</strong></p>

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
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  </a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/contributing.md">Contributing</a>
</p>

</div>

---

Most AI applications are stateless. They forget users between sessions, lose
context mid-conversation, and treat every interaction as if it never happened before.

**MemVault is the memory layer that fixes this.** Plug it into any AI agent or
MCP-compatible client and give it persistent, semantically-searchable memory —
backed by SQLite or PostgreSQL, ranked by recency, importance, and access frequency,
and smart enough to consolidate duplicates automatically.

```python
from memvault import MemVault

mc = MemVault()
mc.remember("User prefers Python over JavaScript", user_id="alice")

results = mc.recall("programming language preferences", user_id="alice")
# → [0.82] User prefers Python over JavaScript
```

## Features

- **Semantic retrieval** — finds memories by meaning, not keyword matching
- **Hybrid ranking** — embedding similarity + recency + importance + access frequency
- **Auto-ingestion** — extract memorable facts from raw conversations automatically
- **MCP server** — plug into Claude Desktop, Cursor, or any MCP-compatible client
- **Multiple backends** — SQLite (zero setup), PostgreSQL (production), in-memory (tests)
- **Consolidation** — detects near-duplicate memories and merges them
- **Decay & reinforcement** — unaccessed memories fade; accessed ones strengthen
- **REST API + CLI** — use as a standalone service or a Python library
- **Provider-agnostic** — bring your own embedder, extractor, or storage backend

## Quick Start

### Library

```bash
pip install "eviloomemvault[local]"
```

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

# Retrieve by meaning — not keywords
results = mc.recall("display preferences", user_id="alice")
for r in results:
    print(f"[{r.final_score:.3f}] {r.item.content}")

# Auto-ingest from a conversation
mc.ingest(
    messages=[
        {"role": "user", "content": "I've been using Rust for systems work."},
        {"role": "user", "content": "Python is my go-to for AI projects."},
    ],
    user_id="alice",
)
```

### MCP Server (Claude Desktop / Cursor)

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

Restart Claude Desktop. It now has 6 memory tools and will remember things
across conversations using your local database.

### REST API

```bash
# Docker (recommended)
docker compose -f docker/docker-compose.yml up

# Direct
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

### CLI

```bash
# Install with local embeddings
pip install "eviloomemvault[local]"

memvault remember "User prefers dark mode" --user alice
memvault recall "display preferences" --user alice
memvault consolidate --user alice
memvault doctor
```

## Installation

```bash
# Core (REST API + CLI)
pip install eviloomemvault

# With local BGE embeddings (recommended)
pip install "eviloomemvault[local]"

# With PostgreSQL support
pip install "eviloomemvault[postgres]"
```

**Requirements:** Python 3.10+

## Benchmarks

Measured on CPU-only hardware (no GPU), BGE-small embeddings, Python 3.10.
Full results: [docs/benchmark_results.md](docs/benchmark_results.md)

| Operation | Median | P95 |
|-----------|--------|-----|
| Embed 1 sentence (BGE-small, CPU) | 29ms | 31ms |
| Insert with embedding | 25ms | 30ms |
| Insert without embedding | <1ms | <1ms |
| Retrieval — 100 memories | 24ms | 28ms |
| Retrieval — 1,000 memories | 27ms | 34ms |
| Retrieval — 5,000 memories | 28ms | 33ms |
| Rule-based ingestion (10 messages) | <1ms | <1ms |

**Key result:** retrieval latency barely changes from 100 to 5,000 memories —
candidate fetch is `O(limit)` not `O(n)`, so the store can grow without
degrading response time.

To reproduce: `python benchmarks/benchmark_suite.py`

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

## Memory Types

| Type | When to use | Example |
|------|-------------|---------|
| `semantic` | Stable facts and preferences | "User prefers Python over Java" |
| `episodic` | Events at a point in time | "User completed onboarding on Jan 5" |
| `procedural` | Repeatable workflows | "Always run tests before committing" |
| `working` | Short-term session context | "Currently debugging the auth module" |
| `consolidated` | Auto-generated summaries | Created by `mc.consolidate()` |

## Storage Backends

| Backend | Use case | Setup |
|---------|----------|-------|
| `SQLiteStorage` | Local dev, single-process production | Zero setup |
| `PostgresStorage` | Production, multi-process | Postgres instance |
| `InMemoryStorage` | Tests, experimentation | Zero setup |

All backends implement the same interface — swap with one line:

```python
from memvault import MemVault
from memvault.storage.postgres import PostgresStorage

mc = MemVault(storage=PostgresStorage("postgresql://user:pass@host/db"))
```

## Ingestion

Extract memorable facts from conversations without manually deciding
what to remember:

```python
# Rule-based (zero dependencies, works offline, <1ms)
mc.ingest(messages, user_id="alice")

# LLM-powered (higher quality, requires API key)
from memvault.ingestion.anthropic_extractor import AnthropicExtractor
mc.ingest(messages, user_id="alice", extractor=AnthropicExtractor())
```

## Retrieval Scoring

```
final_score = (similarity_weight × cosine_similarity)
            + (relevance_weight  × relevance_score)

relevance_score = 0.4 × recency
                + 0.4 × importance
                + 0.2 × frequency
```

All weights are configurable via `RetrievalConfig` and `ScoringWeights`.

## Project Structure

```
src/memvault/
├── core/               # Models, scoring, retrieval, consolidation
├── storage/            # SQLite, PostgreSQL, in-memory backends
├── embeddings/         # BGE-small and pluggable embedding providers
├── ingestion/          # Rule-based and LLM-powered fact extraction
├── api/                # FastAPI REST service
├── cli/                # Typer CLI
├── mcp_server/         # MCP server for Claude Desktop / Cursor
└── observability/      # Structured logging and metrics

benchmarks/             # Performance benchmark suite
docs/                   # Architecture, getting started, benchmarks
```


## Contributing

Contributions are welcome — bug fixes, new backends, embedding providers,
and performance work especially.

```bash
git clone https://github.com/Aryaneviloo/memvault.git
cd memvault
pip install -e ".[local,dev]"
pytest
