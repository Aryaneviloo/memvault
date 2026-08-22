import pytest
from memvault.core.models import MemoryItem, MemoryQuery, MemoryType
from memvault.core.retrieval import retrieve


# --- Phase 1: Basic Store & Retrieve ---

def test_benchmark_store_in_memory(benchmark, in_memory_setup, sample_item):
    wrapper, _, _ = in_memory_setup
    benchmark(wrapper.insert, sample_item)


def test_benchmark_retrieve_in_memory(benchmark, in_memory_setup, sample_item):
    wrapper, backend, embedder = in_memory_setup
    wrapper.insert(sample_item)
    query = MemoryQuery(text="programming language preferences", user_id="bench-user")

    benchmark(retrieve, query=query, backend=backend, embedder=embedder)


# --- Phase 2: Scale Testing (SQLite vs In-Memory) ---

@pytest.mark.parametrize("backend_type", ["in_memory", "sqlite"])
@pytest.mark.parametrize("entry_count", [100, 1000])
def test_benchmark_scaled_retrieval(benchmark, request, backend_type, entry_count):
    wrapper, backend, embedder = request.getfixturevalue(f"{backend_type}_setup")

    # Seed dataset size
    for i in range(entry_count):
        wrapper.insert(
            MemoryItem(
                agent_id="bench-agent",
                user_id="bench-user",
                type=MemoryType.SEMANTIC,
                content=f"Memory record #{i} storing user preferences and agent logs.",
                importance=0.5,
            )
        )

    query = MemoryQuery(text="user preferences", user_id="bench-user")

    # Measure search latency across scaling bounds
    benchmark(retrieve, query=query, backend=backend, embedder=embedder)