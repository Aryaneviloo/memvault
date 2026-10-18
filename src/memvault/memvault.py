"""
MemVault — the main entry point for using this library.

This facade class wires together storage, embeddings, retrieval,
scoring, and consolidation into a single clean interface.
Advanced users can still import and use each layer directly.
"""

from __future__ import annotations

from memvault.core.consolidation import ConsolidationConfig, ConsolidationResult, consolidate
from memvault.core.models import MemoryItem, MemoryQuery, MemoryType
from memvault.core.retrieval import RetrievalConfig, RetrievalResult, retrieve
from memvault.core.scoring import apply_decay
from memvault.core.scoring import reinforce as _reinforce
from memvault.embeddings.base import BaseEmbedder
from memvault.ingestion.base import BaseExtractor
from memvault.ingestion.rule_based import RuleBasedExtractor
from memvault.storage.base import EmbeddingStorageWrapper, StorageBackend


class MemVault:
    """
    The main interface for MemVault.

    Wires together storage, embeddings, retrieval, scoring, and
    consolidation into a single clean API.

    Basic usage:
        mc = MemVault()
        mc.remember("user likes Python", user_id="alice")
        results = mc.recall("programming preferences", user_id="alice")

    Custom backends:
        from memvault.storage.postgres import PostgresStorage
        from memvault.embeddings.local import LocalEmbedder

        mc = MemVault(
            storage=PostgresStorage("postgresql://..."),
            embedder=LocalEmbedder(),
        )
    """

    def __init__(
        self,
        storage: StorageBackend | None = None,
        embedder: BaseEmbedder | None = None,
        db_path: str = "memories.db",
    ) -> None:
        """
        Initialize MemVault.

        Args:
            storage: Any StorageBackend. Defaults to SQLiteStorage(db_path).
            embedder: Any BaseEmbedder. Defaults to LocalEmbedder().
            db_path: Path for SQLite database (used only if storage not provided).
        """
        if embedder is None:
            from memvault.embeddings.local import LocalEmbedder

            embedder = LocalEmbedder()

        if storage is None:
            from memvault.storage.sqlite import SQLiteStorage

            storage = SQLiteStorage(db_path)

        self._embedder = embedder
        self._backend = storage
        self._store = EmbeddingStorageWrapper(
            backend=self._backend,
            embedder=self._embedder,
        )

    def remember(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str = "default-agent",
        namespace: str = "default",
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        source: str | None = None,
    ) -> MemoryItem:
        """
        Store a new memory.

        Args:
            content: The memory text to store.
            user_id: Who this memory belongs to.
            agent_id: Which agent is storing it.
            namespace: Memory pool (default: "default").
            memory_type: episodic, semantic, procedural, working, or consolidated.
            importance: How significant this memory is (0.0–1.0).
            tags: Optional list of tags for filtering.
            metadata: Optional dict of extra info.
            source: Where this memory came from (e.g. "user_message").

        Returns:
            The stored MemoryItem with generated ID and embedding.
        """
        item = MemoryItem(
            agent_id=agent_id,
            user_id=user_id,
            namespace=namespace,
            type=memory_type,
            content=content,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            source=source,
        )
        return self._store.insert(item)

    def recall(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        namespace: str = "default",
        top_k: int = 5,
        memory_types: list[MemoryType] | None = None,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """
        Search memories semantically.

        Returns results ranked by combined similarity + relevance score.

        Args:
            query: Natural language query.
            user_id: Whose memories to search.
            agent_id: Optional agent scope.
            namespace: Memory pool to search.
            top_k: Maximum number of results to return.
            memory_types: Filter to specific memory types.
            config: Custom retrieval weights and thresholds.

        Returns:
            List of RetrievalResult (item + scores), ranked best-first.
        """
        memory_query = MemoryQuery(
            text=query,
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            top_k=top_k,
            types=memory_types,
        )
        return retrieve(
            query=memory_query,
            backend=self._backend,
            embedder=self._embedder,
            config=config,
        )

    def forget(self, memory_id: str, *, hard: bool = False) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: The memory's unique ID.
            hard: If True, permanently delete. If False (default), soft-delete.

        Returns:
            True if something was deleted, False if not found.
        """
        return self._store.delete(memory_id, hard=hard)

    def get(self, memory_id: str) -> MemoryItem | None:
        """Fetch a single memory by ID. Returns None if not found."""
        return self._store.get(memory_id)

    def recent(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Return the most recently created memories for a user."""
        return self._backend.list_recent(
            user_id=user_id,
            agent_id=agent_id,
            namespace=namespace,
            limit=limit,
        )


    def list_namespaces(
            self,
            user_id: str,
            *,
            agent_id: str | None = None,
    ) -> list[str]:
        """Return the distinct namespaces that held live memories for a user"""
        return self._backend.list_namespaces(
            user_id=user_id,
            agent_id=agent_id,
        )

    def consolidate(
        self,
        user_id: str,
        *,
        agent_id: str | None = None,
        namespace: str = "default",
        similarity_threshold: float = 0.85,
    ) -> ConsolidationResult:
        """
        Merge near-duplicate memories for a user.

        Detects clusters of similar memories, summarizes each cluster
        into a CONSOLIDATED memory, and soft-deletes the originals.

        Returns:
            ConsolidationResult with counts and consolidated item IDs.
        """
        config = ConsolidationConfig(similarity_threshold=similarity_threshold)
        return consolidate(
            user_id=user_id,
            backend=self._backend,
            agent_id=agent_id,
            namespace=namespace,
            config=config,
        )

    def reinforce(self, memory_id: str) -> MemoryItem | None:
        """
        Record that a memory was accessed and found useful.

        Increments access_count, updates last_accessed_at,
        and gives a small importance boost.

        Returns:
            Updated MemoryItem, or None if not found.
        """
        item = self._store.get(memory_id)
        if item is None:
            return None
        _reinforce(item)
        return self._store.update(item)

    def decay(
        self,
        user_id: str,
        *,
        namespace: str = "default",
        decay_rate_per_day: float = 0.01,
    ) -> int:
        """
        Apply time-based decay to all memories for a user.

        Reduces importance of unaccessed memories over time.
        Call this periodically (e.g. once per day).

        Returns:
            Number of memories updated.
        """
        memories = self._backend.list_recent(
            user_id=user_id,
            namespace=namespace,
            limit=10000,
        )
        count = 0
        for item in memories:
            apply_decay(item, decay_rate_per_day=decay_rate_per_day)
            self._backend.update(item)
            count += 1
        return count

    def ingest(
        self,
        messages: list[dict],
        *,
        user_id: str,
        agent_id: str = "default-agent",
        namespace: str = "default",
        extractor: BaseExtractor | None = None,
    ) -> list[MemoryItem]:
        """
        Extract and store memorable facts from a conversation.
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            user_id: Who this conversation belongs to.
            agent_id: Which agent is ingesting.
            namespace: Memory namespace.
            extractor: Extraction strategy. Defaults to RuleBasedExtractor
                      (no API key needed). Use AnthropicExtractor or
                      OpenAIExtractor for higher quality.

        Returns:
            List of MemoryItems that were stored.

        Example:
            mc.ingest(messages, user_id="alice")

            # With LLM extraction:
            from memvault.ingestion.anthropic_extractor import AnthropicExtractor
            mc.ingest(messages, user_id="alice", extractor=AnthropicExtractor())
        """
        if extractor is None:
            extractor = RuleBasedExtractor()

        facts = extractor.extract(messages)
        stored = []

        for fact in facts:
            from memvault.core.models import MemoryType
            try:
                mem_type = MemoryType(fact.memory_type)
            except ValueError:
                mem_type = MemoryType.EPISODIC

            item = self.remember(
                fact.content,
                user_id=user_id,
                agent_id=agent_id,
                namespace=namespace,
                memory_type=mem_type,
                importance=fact.importance,
                tags=fact.tags or [],
                source="auto-ingested",
            )
            stored.append(item)

        return stored