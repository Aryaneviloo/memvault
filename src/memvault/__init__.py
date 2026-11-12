"""MemVault - open source infrastructure for AI agents

Quick starts:
    from memvault import MemVault

    mc = MemVault()
    mc.remember("User likes C++", user_id="aryan")
    results = mc.recall("programming preferences", user_id="aryan")

"""

from memvault.core.models import MemoryItem, MemoryQuery, MemoryType
from memvault.core.retrieval import RetrievalConfig, RetrievalResult
from memvault.core.scoring import ScoringWeights
from memvault.ingestion.base import BaseExtractor, ExtractedFact
from memvault.ingestion.rule_based import RuleBasedExtractor
from memvault.memvault import MemVault

__version__ = "0.3.0"
__all__ = [
    "MemVault",
    "MemoryItem",
    "MemoryQuery",
    "MemoryType",
    "RetrievalConfig",
    "RetrievalResult",
    "ScoringWeights",
    "BaseExtractor",
    "ExtractedFact",
    "RuleBasedExtractor",
]
