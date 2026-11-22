# Changelog

All notable changes to MemVault are documented here.

## [Unreleased]

## [0.4.1] — 2026-11-22

### Fixed
- `embeddings/local.py`: `sentence_transformers` is now imported lazily inside `LocalEmbedder.__init__`. It was imported at module level, so `memvault --help` crashed on any install without the `[local]` extra. A missing extra now raises a clear `ImportError` with the exact install command (`pip install 'eviloomemvault[local]'`)
- `cli/main.py`: a missing `[local]` extra prints one clean error line and exits with code 1 instead of a traceback, and no `memories.db` is created in that case
- `cli/main.py`: `forget` and `consolidate` no longer load the embedding model, so they work without the `[local]` extra
- `cli/main.py`: `recall` loads the embedding model once instead of twice
- `cli/main.py`: `doctor` checks storage and the embedder independently, and only reports "All systems operational" when every check passes (it previously printed it unconditionally and blamed storage for embedder failures). It exits with code 1 on failure
- `core/consolidation.py`: consolidated memories now store an embedding (the normalized average of the source vectors). They were previously saved without one, so `recall` could never find them
- `core/models.py`: `MemoryItem.content` now rejects empty strings
- `mcp_server/server.py`: the storage backend and embedder are built on first use instead of at import time. Importing the module is now safe, and `memvault-mcp` prints a clean error when the `[local]` extra is missing
- Docs: install hints now use the correct package name, `eviloomemvault[local]`

### Changed
- Version bumped to `0.4.1`

## [0.4.0] — 2026

### Added
- Batched processing for `MemVault.decay()` to reduce memory usage (PR #11)
- PGVECTOR support planning included in roadmap

### Changed
- `__init__.py`: version updated to `0.4.0`

### Known issues
- The CLI and MCP server crash on startup without the `[local]` extra (fixed in 0.4.1)

## [0.3.0] — 2026

### Fixed
- `api/app.py`: `app = create_app()` was missing `()`, causing `uvicorn` to receive a function object instead of a FastAPI instance (REST API would not start)
- `ingestion/anthropic_extractor.py`: `extract()` was nested inside `__init__` (indentation bug), so calling `.extract()` raised `AttributeError`
- `ingestion/openai_extractor.py`: was an empty file; implemented `OpenAIExtractor(BaseExtractor)` and exported it from the top-level `memvault` package
- `memvault.py`: added missing `MemVault.recent()` method used by LangChain history adapter and examples
- `mcp_server/server.py`: `search_memories` no longer reinforces and updates every returned memory (read operation was mutating state)
- `storage/sqlite.py`: added `PRAGMA journal_mode=WAL` and `busy_timeout=30000` to fix "database is locked" on concurrent writes
- `storage/base.py`: `EmbeddingStorageWrapper.update()` skips re-embedding when content is unchanged (avoids wasteful model reloads on `reinforce()`/`PATCH`)
- `core/retrieval.py`: `recency_bias` field on `MemoryQuery`/`SearchRequest` now actually influences ranking (was previously ignored)

### Changed
- `__init__.py`: version aligned to `0.3.0` to match `pyproject.toml`

## [0.2.0] — 2026

### Added
- MCP server with 6 tools: `store_memory`, `search_memories`,
  `list_recent_memories`, `consolidate_memories`, `delete_memory`,
  `ingest_conversation`
- Tested with Claude Desktop — memories persist across sessions
- Auto-ingestion pipeline: `MemVault.ingest()` extracts memorable facts
  from raw conversations automatically
- `RuleBasedExtractor` — zero-dependency pattern-matching extraction
- `AnthropicExtractor` — LLM-powered extraction via Claude Haiku
- `BaseExtractor` ABC — bring your own extraction provider

## [0.1.1] — 2026

### Added
- `typer` and `rich` added to core dependencies (CLI now works
  without `[local]` extras)

### Fixed
- CLI entry point missing `typer` dependency

## [0.1.0] — 2026

### Added
- Core memory engine: `MemoryItem`, `MemoryQuery`, `MemoryType`
- Scoring: recency, importance, frequency, `relevance_score()`
- Decay and reinforcement: `apply_decay()`, `reinforce()`
- Storage backends: `SQLiteStorage`, `PostgresStorage`, `InMemoryStorage`
- Contract test suite — 16 tests run against all backends automatically
- BGE-small local embeddings via `sentence-transformers`
- Hybrid semantic retrieval: cosine similarity + relevance scoring
- Consolidation: near-duplicate detection and merging
- `MemVault` facade class — single-import public API
- FastAPI REST API with OpenAPI docs
- Typer CLI: `remember`, `recall`, `forget`, `consolidate`, `doctor`
- Docker + docker-compose support
- Structured logging via `structlog`
- In-process metrics + `/metrics` endpoint
- GitHub Actions CI/CD
- Published on PyPI as `eviloomemvault`