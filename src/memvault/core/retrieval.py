from dataclasses import dataclass, field
from datetime import datetime

from memvault.core.models import MemoryItem, MemoryQuery
from memvault.core.scoring import ScoringWeights, relevance_score
from memvault.embeddings.base import BaseEmbedder
from memvault.storage.base import StorageBackend


@dataclass
class RetrievalResult:
    """
    A memory item paired with ranking scores returned bu the retrieval pipeline so callers can see
    not just *what* but *why*
    """

    item: MemoryItem
    similarity: float
    relevance: float
    final_score: float


@dataclass
class RetrievalConfig:
    """Tunable parameter for hybrid retrieval pipeline"""

    similarity_weight: float = 0.5
    relevance_weight: float = 0.5
    min_similarity: float = 0.3
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Dot Product of two normalized vectors = cosine similarity
    Both vectors must already be L2 normalized (guaranteed by LocalEmbedder)
    """

    return sum(x * y for x, y in zip(a, b, strict=False))


def retrieve(
    query: MemoryQuery,
    backend: StorageBackend,
    embedder: BaseEmbedder,
    config: RetrievalConfig | None = None,
    now: datetime | None = None,
) -> list[RetrievalResult]:
    """
    Hybrid retrieval pipeline

    1> Fetch candidate memmories from storage
    2> Embed the query
    3> Score each candidate
    4> Filter out low similarity
    5> Rank by final score


    Args:
        query: The search request (text, user_id, filters, top_k etc.)
        backend: Any StorageBackend to fetch candidates from
        embedder: Any BaseEmbedder to embed the query
        config: Tunable weights and thresholds (uses defaults if None)
        now: Reference time for recency scoring (uses UTC now if None)
    """
    if config is None:
        config = RetrievalConfig()

    # fetch broad candidates from storage
    # we use list_recent with a high limit to get a candidate pool,
    # then re-rank here. In v2 this could use vector DB ANN search.

    candidates = backend.list_recent(
        user_id=query.user_id,
        agent_id=query.agent_id,
        namespace=query.namespace,
        limit=200,
    )

    if query.types is not None:
        candidates = [c for c in candidates if c.type in query.types]

    if not candidates:
        return []

    # ---------STEP 2 embed the query once-------------

    query_vector = embedder.embed_query(query.text)

    results = []
    for item in candidates:
        if item.embedding is None:
            continue
        # --------------- #STEP: 3 Part 1:---------
        similarity = cosine_similarity(query_vector, item.embedding)

        # ----------Step 3 part 2: filter beofre min similarity

        if similarity < config.min_similarity:
            continue

        # --------Step 3 part 3: relevance score

        rel = relevance_score(
            item,
            weights=config.scoring_weights,
            now=now,
        )

        # ---------Step 3 part 4--------------

        final = config.similarity_weight * similarity + config.relevance_weight * rel

        results.append(
            RetrievalResult(
                item=item,
                similarity=round(similarity, 4),
                relevance=round(rel, 4),
                final_score=round(final, 4),
            )
        )

        # --------------STEP $ SORT -----------

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[: query.top_k]
