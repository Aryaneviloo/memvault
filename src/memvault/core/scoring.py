from dataclasses import dataclass
from datetime import datetime, timezone

from memvault.core.models import MemoryItem


def recency_score(
    item: MemoryItem, *, half_life_days: float = 7.0, now: datetime | None = None
) -> float:
    """
    Score how recent a memory is, using exponential decay.

    Returns a value in (0, 1]. 1.0 means "just happened",
    approaching 0 as the memory gets older relative to half_life_days.

    Uses last_accessed_at if available, otherwise falls back to created_at.

    """
    if now is None:
        now = datetime.now(timezone.utc)

    reference_time = item.last_accessed_at or item.created_at
    age_seconds = (now - reference_time).total_seconds()
    age_days = max(age_seconds, 0) / 86400

    return 0.5 ** (age_days / half_life_days)


def importance_score(item: MemoryItem) -> float:
    """
    Return the memory's importance, weighted by confidence.

    A highly important but low-confidence memory should rank lower
    than a highly important, high-confidence one.
    """
    return item.importance * item.confidence


def frequency_score(item: MemoryItem, *, saturation: float = 10.0) -> float:
    """
    count how often the memory was revisited
    Returns a value in (0, 1], approaches 1 as frequency count increases
    """
    return item.access_count / (item.access_count + saturation)


@dataclass
class ScoringWeights:
    """Tunable weights for combining individual scores into a final relevance score."""

    recency: float = 0.4
    importance: float = 0.4
    frequency: float = 0.2


def relevance_score(
    item: MemoryItem,
    *,
    weights: ScoringWeights | None = None,
    half_life_days: float = 7.0,
    now: datetime | None = None,
) -> float:
    """
    Combine recency, importance, and frequency into a single relevance score.

    Returns a value roughly in [0, 1], though not strictly bounded
    since weights aren't required to sum to 1.
    """
    if weights is None:
        weights = ScoringWeights()

    r = recency_score(item, half_life_days=half_life_days, now=now)
    i = importance_score(item)
    f = frequency_score(item)

    return (weights.recency * r) + (weights.importance * i) + (weights.frequency * f)


def apply_decay(
    item: MemoryItem,
    *,
    decay_rate_per_day: float = 0.01,
    now: datetime | None = None,
) -> MemoryItem:
    """
    Reduce a memory's importance based on elapsed time since last access.

    Memories with higher importance decay more slowly (proportionally),
    matching the principle that important things should persist longer.

    Mutates and returns the item. Does not touch memories that have
    been accessed very recently (decay only applies to elapsed time
    since last_accessed_at, or created_at if never accessed).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    reference_time = item.last_accessed_at or item.created_at
    age_days = max((now - reference_time).total_seconds(), 0) / 86400

    # Higher importance => slower decay (effective rate scales down)
    effective_rate = decay_rate_per_day * (1.0 - 0.5 * item.importance)

    decay_factor = (1.0 - effective_rate) ** age_days
    item.importance = max(0.0, item.importance * decay_factor)

    return item


def reinforce(
    item: MemoryItem,
    *,
    importance_boost: float = 0.02,
    now: datetime | None = None,
) -> MemoryItem:
    """
    Record that a memory was accessed/retrieved.

    Updates last_accessed_at, increments access_count, and gives
    a small boost to importance (repeated relevance increases
    perceived importance, with diminishing effect as importance
    approaches 1.0).

    Mutates and returns the item.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    item.last_accessed_at = now
    item.access_count += 1

    headroom = 1.0 - item.importance
    item.importance = min(1.0, item.importance + importance_boost * headroom)

    return item
