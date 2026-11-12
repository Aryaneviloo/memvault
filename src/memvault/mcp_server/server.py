"""
Memvault MCP server

Exposes memvault's memory operaitons as MCP toos so any MCP
compatible client can use memvault as a memory layer
Run:
    python -m memvault.mcp_server.server
"""

import os

from fastmcp import FastMCP

from memvault.core.consolidation import ConsolidationConfig, consolidate
from memvault.core.models import MemoryItem, MemoryQuery, MemoryType
from memvault.core.retrieval import retrieve
from memvault.embeddings.local import LocalEmbedder
from memvault.storage.base import EmbeddingStorageWrapper
from memvault.storage.sqlite import SQLiteStorage

#-----SETUP---------

DB_PATH = os.environ.get("MEMVAULT_DB", "memvault.db")
_backend = SQLiteStorage(DB_PATH)
_embedder = LocalEmbedder()
_store = EmbeddingStorageWrapper(backend=_backend, embedder=_embedder)

mcp = FastMCP("MemVault")

#--------TOOLS--------------

@mcp.tool()
def store_memory(
    content: str,
    user_id: str,
    agent_id: str = "claude",
    memory_type: str = "episodic",
    importance: float = 0.5,
) -> str:
    """
    Store a new memory in Memvault
    
    Use this when you learn something new about the user, thei
    preferences, etc
    
    Args:
         content: The memory text to store
         user_id: Who this memory belongs to (name or ID)
         agent_id: Which agent is using it (default = claude)
         memory_type: one of episodic/semantic/procedural/working
         importance: how significant the memory is
         
    Returns:
         Confirmation with the shared memory ID
    """

    try:
        mem_type = MemoryType(memory_type)
    except ValueError:
        mem_type = MemoryType.EPISODIC

    item = MemoryItem(
        agent_id=agent_id,
        user_id=user_id,
        type=mem_type,
        content=content,
        importance=importance,
    )
    stored = _store.insert(item)
    return f"Memory stored — ID: {stored.id} | Type: {mem_type.value} | Importance: {importance}"



@mcp.tool()
def search_memories(
    query: str,
    user_id: str,
    top_k: int = 3,
    agent_id: str | None = None,  
) -> str:
    """
    Search memories semantically using hybrid retreival
    
    Use this BEFORE responding to retrieve relevant context about
    the user. The search understands meaning
    
      Args:
        query: Natural language search query.
        user_id: Whose memories to search.
        top_k: Maximum number of results to return (default 5).
        agent_id: Optional — restrict to a specific agent's memories.

    Returns:
        Ranked list of relevant memories with scores.
    """

    memory_query = MemoryQuery(
        text = query,
        user_id=user_id,
        agent_id=agent_id,
        top_k=top_k,
    )

    results = retrieve(
        query=memory_query,
        backend=_backend,
        embedder=_embedder,
    )

    if not results: 
        return f"No memories found for user '{user_id}' matching '{query}'."


    lines = [f"Found {len(results)} memories for '{query}' :\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. [{r.item.type.value}] {r.item.content}\n"
            f"   Score: {r.final_score:.3f} "
            f"(similarity: {r.similarity:.3f}, relevance: {r.relevance:.3f})\n"
        )




    return "\n".join(lines)


@mcp.tool()
def list_recent_memories(
    user_id: str,
    limit: int = 10,
    agent_id: str | None = None

) -> str:
    """
    List the most recently stores memories for a user
    
    Use this to get a braod overview of what you know about
    a user at the start of conversation
    
    
    Args: 
        user_id: Whose mmeories to list
        limit: Maxmimum number of memories to return(default = 10)
        agent_id: WHo is belongs to, Optional
        
    Returns:
        Chronological list of recent memories
    """

    memories = _backend.list_recent(
        user_id=user_id,
        agent_id=agent_id,
        limit=limit,
    )

    if not memories:
        return f"No memories found for the user '{user_id}'."
    lines = [f"{len(memories)} recent memories for '{user_id}':\n"]
    for i, m in enumerate(memories, 1):
        lines.append(
            f"{i}. [{m.type.value}] {m.content}\n"
            f"   Importance: {m.importance:.2f} | "
            f"Accessed: {m.access_count}x | "
            f"Created: {m.created_at.strftime('%Y-%m-%d')}\n"
        )

    return "\n".join(lines)


@mcp.tool()
def consolidate_memories(
    user_id: str,
    similarity_threshold: float = 0.85,
) -> str:
    """
    Merge near duplicate memories for a user
    
    RUn this periodically to keep the memory store clean
    SImilar memories get merged into a single consolidated
    memory
    
    Args:
         user_id: WHo it belogns to
         similarity_threshold: How similar are the memories
    Returns:
        SUmmary of all the consolidated memories
    """

    config = ConsolidationConfig(similarity_threshold=similarity_threshold)
    result = consolidate(
        user_id=user_id,
        backend=_backend,
        config=config,
    )

    if result.memories_consolidated == 0:
        return f"Nothing to consolidate for user '{user_id}' at threshold {similarity_threshold}."

    return (
        f"Consolidated {result.memories_consolidated} memories "
        f"into {result.clusters_found} clusters for user '{user_id}'."
    )


@mcp.tool()
def delete_memory(
    memory_id: str,
    hard: bool = False,
) -> str:

    """
    Delete a memory by ID.

    Args:
        memory_id: The memory's unique ID.
        hard: If True, permanently delete. Default is soft-delete (recoverable).

    Returns:
        Confirmation of deletion.
    """
    deleted = _store.delete(memory_id, hard=hard)
    if deleted:
        action = "permanently deleted" if hard else "soft-deleted"
        return f"Memory {action}: {memory_id}"
    return f"Memory not found: {memory_id}"

@mcp.tool
def ingest_conversation(
    messages: list[dict],
    user_id: str,
    agent_id: str = "claude",
    use_llm: bool = False,
) -> str:
    """
    Extract and store memorable facts from a conversation automatically
    
    Use this at the end of a conversation to extract and persist evertything
    
    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."}.
        user_id: Who the conversation belongs to.
        agent_id: Which agent is ingesting (default: claude).
        use_llm: If True and ANTHROPIC_API_KEY is set, use Claude Haiku
                 for higher-quality extraction. Otherwise uses rule-based.

    Returns:
        Summary of what was extracted and stored.
    """

    from memvault import MemVault

    mc = MemVault(storage=_backend, embedder=_embedder)

    extractor = None
    if use_llm:
        try:
            from memvault.ingestion.anthropic_extractor import AnthropicExtractor
            extractor = AnthropicExtractor()

        except (ImportError, ValueError):
            pass


    stored = mc.ingest(
        messages,
        user_id=user_id,
        agent_id=agent_id,
        extractor=extractor,
    )

    if not stored:
        return f"No memorable facts found in the conversation for the user '{user_id}'."
    lines = [f"Extracted and stored {len(stored)} facts for '{user_id}' :\n"]
    for item in stored:
        lines.append(f" [{item.type.value}] {item.content[:80]}")

    return "\n".join(lines)    
#--------Entry Point---------

def main():
    mcp.run()

if __name__ == "__main__":
    main()
