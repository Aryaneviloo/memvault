import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memvault.core.models import MemoryItem, MemoryType
from memvault.core.retrieval import cosine_similarity
from memvault.storage.base import StorageBackend


@dataclass
class ConsolidationConfig:
    """Tunable parameters for the consolidation pipeline."""

    similarity_threshold: float = 0.85
    min_cluster_size: int = 2
    max_cluster_size: int = 10


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    clusters_found: int
    memories_consolidated: int = 0
    consolidated_items: list[MemoryItem] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)


def _find_clusters(
    items: list[MemoryItem],
    threshold: float,
    max_size: int,
) -> list[list[MemoryItem]]:
    """
    Group memories into clusters of similar items.

    Uses a greedy approach: iterate through items, assign each
    to an existing cluster if similar enough to its first member,
    otherwise start a new cluster.

    Returns only clusters with 2+ members (singletons aren't consolidated).
    """
    if not items:
        return []

    clusters: list[list[MemoryItem]] = []
    assigned = set()

    for i, item in enumerate(items):
        if i in assigned or item.embedding is None:
            continue

        cluster = [item]
        assigned.add(i)

        for j, other in enumerate(items):
            if j in assigned or other.embedding is None:
                continue
            if len(cluster) >= max_size:
                break

            sim = cosine_similarity(item.embedding, other.embedding)
            if sim >= threshold:
                cluster.append(other)
                assigned.add(j)

        clusters.append(cluster)

    return [c for c in clusters if len(c) >= 2]


def _make_summary(items: list[MemoryItem]) -> str:
    """
    Generate a simple summary from a cluster of similar memories.

    In v1 this is a deterministic text merge — no LLM needed.
    The LLM-based summarizer is a planned v2 improvement.
    """

    contents = [item.content for item in items]
    unique = list(dict.fromkeys(contents))

    if len(unique) == 1:
        return unique[0]

    joined = " | ".join(unique)
    return f"[Consolidated] {joined}"


def _centroid(items: list[MemoryItem]) -> list[float]:
    """
    Average the cluster's vectors, then rescale lenght to 1"""
    vectors = [i.embedding for i in items]
    avg = [sum(col)/len(vectors) for col in zip(*vectors, strict = True)]
    norm = math.sqrt(sum(x * x for x in avg))
    if norm == 0:
        return avg
    return[x/norm for x in avg]

def consolidate(
    user_id: str,
    backend: StorageBackend,
    *,
    agent_id: str | None = None,
    namespace: str = "default",
    config: ConsolidationConfig | None = None,
    now: datetime | None = None,
) -> ConsolidationResult:
    """
    Find clusters of similar memories, merge each cluster into
    a single CONSOLIDATED memory, and soft-delete the originals.

    Args:
        user_id: Scope consolidation to this user
        backend: Storage backend to read from and write to
        agent_id: Optional agent scope
        namespace: Memory namespace to consolidate
        config: Tunable thresholds
        now: Reference time (uses UTC now if None)
    """

    if config is None:
        config = ConsolidationConfig()
    if now is None:
        now = datetime.now(timezone.utc)

    candidates = backend.list_recent(
        user_id=user_id,
        agent_id=agent_id,
        namespace=namespace,
        limit=500,
    )

    embeddable = [c for c in candidates if c.embedding is not None]

    clusters = _find_clusters(
        embeddable,
        threshold=config.similarity_threshold,
        max_size=config.max_cluster_size,
    )

    result = ConsolidationResult(clusters_found=len(clusters))

    for cluster in clusters:
        summary = _make_summary(cluster)

        max_importance = max(item.importance for item in cluster)

        consolidated = MemoryItem(
            agent_id=cluster[0].agent_id,
            user_id=user_id,
            namespace=namespace,
            type=MemoryType.CONSOLIDATED,
            content=summary,
            summary=summary,
            embedding = _centroid(cluster),
            importance=min(max_importance * 1.1, 1.0),
            confidence=min(sum(i.confidence for i in cluster) / len(cluster), 1.0),
            tags=list({tag for item in cluster for tag in item.tags}),
            metadata={
                "consolidated_from": [item.id for item in cluster],
                "consolidated_at": now.isoformat(),
                "cluster_size": len(cluster),
            },
            created_at=now,
            updated_at=now,
        )

        backend.insert(consolidated)

        for item in cluster:
            backend.delete(item.id)

        result.consolidated_items.append(consolidated)
        result.source_ids.extend(item.id for item in cluster)
        result.memories_consolidated += len(cluster)

    return result
