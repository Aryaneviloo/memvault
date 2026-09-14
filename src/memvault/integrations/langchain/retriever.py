"""
LangChain retriever adapter for MemVault.

Plugs MemVault's semantic search into any LangChain RAG chain.

Usage:
    from memvault.integrations.langchain import MemVaultRetriever

    retriever = MemVaultRetriever(user_id="alice")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from memvault.core.models import MemoryType
from memvault.memvault import MemVault


class MemVaultRetriever(BaseRetriever):
    """
    LangChain retriever backed by MemVault semantic search.

    Drop-in replacement for any LangChain retriever.
    Returns memories ranked by embedding similarity + recency + importance.
    """

    user_id: str
    top_k: int = 5
    agent_id: str | None = None
    namespace: str = "default"
    memory_types: list[str] | None = None
    db_path: str = "memories.db"

    # Private — excluded from LangChain's Pydantic model
    _mc: MemVault | None = None

    def _get_mc(self) -> MemVault:
        if self._mc is None:
            self._mc = MemVault(db_path=self.db_path)
        return self._mc

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve semantically relevant memories for a query."""
        mc = self._get_mc()

        types = None
        if self.memory_types:
            types = [MemoryType(t) for t in self.memory_types]

        results = mc.recall(
            query,
            user_id=self.user_id,
            agent_id=self.agent_id,
            namespace=self.namespace,
            top_k=self.top_k,
            memory_types=types,
        )

        return [
            Document(
                page_content=r.item.content,
                metadata={
                    "memory_id": r.item.id,
                    "type": r.item.type.value,
                    "importance": r.item.importance,
                    "similarity": r.similarity,
                    "relevance": r.relevance,
                    "final_score": r.final_score,
                    "created_at": r.item.created_at.isoformat(),
                    "user_id": r.item.user_id,
                    "namespace": r.item.namespace,
                },
            )
            for r in results
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Async version — delegates to sync for now."""
        return self._get_relevant_documents(query, run_manager=run_manager)