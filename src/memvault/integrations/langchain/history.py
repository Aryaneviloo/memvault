"""
Langchain memory adapter for Memvault

Replacement for COnversation Buffer Memory with:
- SQLite/PostgreSQL
- Semantic retrieval
- Auto extraction of memorable facts via ingest()

Usage: 
 from memvault.integrations.langchain import MemVaultMemory
 
 memory = MemVaultMemory(user_id = "alice")
 chain = ConversationChain(llm = llm, memory = memory)
"""

from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from memvault.core.models import MemoryType
from memvault.memvault import MemVault


class MemVaultChatHistory(BaseChatMessageHistory):
    """
    Langchain history backed by Memvault
    Stores each message as working memory, retrieves in chronological order
    """
    def __init__(
            self,
            user_id: str,
            agent_id: str = "langchain-agent",
            namespace: str = "default",
            db_path: str = "memories.db",
    ) -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        self.namespace = namespace
        self._mc = MemVault(db_path = db_path)

    @property
    def messages(self) -> list[BaseMessage]:
        """Return message in chronological order"""
        memories = self._mc.recent(
            user_id=self.user_id,
            agent_id=self.agent_id,
            namespace=self.namespace,
            limit = 1000,
        )

        #Filter working memories that are chat messages
        chat_memories = [
            m for m in reversed(memories)
            if m.type == MemoryType.WORKING
            and m.source in ("human", "ai")
        ]

        result = []
        for m in chat_memories:
            if m.source == "human":
                result.append(HumanMessage(content=m.content))
            else:
                result.append(AIMessage(content=m.content))
        return result


    def add_message(self, message: BaseMessage) -> None:
        """Store a single message as a WORKING memory"""
        source = "human" if isinstance(message, HumanMessage) else "ai"
        self._mc.remember(
            message.content,
            user_id=self.user_id,
            agent_id=self.agent_id,
            namespace=self.namespace,
            memory_type=MemoryType.WORKING,
            source=source,
            importance = 0.3
        )

    def clear(self) -> None:
        """Hard deelte all messages for this user/namespace"""
        memories = self._mc.recent(
            user_id=self.user_id,
            agent_id=self.agent_id,
            namespace=self.namespace,
            limit = 10000,
        )
        for m in memories:
            if m.type == MemoryType.WORKING:
                self._mc.forget(m.id, hard=True)
