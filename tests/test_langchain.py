"""Tests for Langchain adapter"""

import pytest

from unittest.mock import patch, MagicMock

from memvault.integrations.langchain import MemVaultRetriever, MemVaultChatHistory
from memvault.core.models import MemoryItem, MemoryType
from memvault.core.retrieval import RetrievalResult
from datetime import datetime, timezone


from langchain_core.messages import HumanMessage, AIMessage


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)



def make_result(content: str) -> RetrievalResult:
    item = MemoryItem(
        agent_id="a",
        user_id="alice",
        type=MemoryType.SEMANTIC,
        content= content,
        created_at=NOW
    )
    item.embedding = [0,1]*384
    return RetrievalResult(
        item=item,
        similarity=0.85,
        relevance=0.7,
        final_score=0.77,
    )

# -- Retrieval test --

def test_retiever_returns_documents(tmp_path):
    r = MemVaultRetriever(user_id="alice", 
                          db_path=str(tmp_path / "test.db"))
    with patch.object(r._get_mc(), 
                      "recall", return_value=[make_result("User likes apples")]):
        docs = r._get_relevant_documents("personal", run_manager=MagicMock())
    assert len(docs) == 1
    assert docs[0].page_content == "User likes apples"
    assert docs[0].metadata["similarity"] == 0.85

def test_retriever_empty(tmp_path):
    r = MemVaultRetriever(user_id="aryan", db_path=str(tmp_path/"test.db"))
    with patch.object(r._get_mc(), "recall", return_value=[]):
        docs = r._get_relevant_documents("anything", run_manager=MagicMock())
    assert docs == []


#---Chat history---

def test_add_and_retrieve_messages(tmp_path):
    h = MemVaultChatHistory(user_id="aryan", db_path=str(tmp_path/"test.db"))
    h.add_message(HumanMessage(content="I like apple"))
    h.add_message(AIMessage(content="Good choice!"))
    msgs = h.messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[0].content == "I like apple"


def test_clear_removes_messages(tmp_path):
    h = MemVaultChatHistory(user_id="aryan",
                            db_path=str(tmp_path / "test.db"))
    h.add_message(HumanMessage(content="hello"))
    h.clear()
    assert h.messages == []

def test_messages_empty_on_fresh_history(tmp_path):
    h = MemVaultChatHistory(user_id="aryan", db_path=str(tmp_path/ "test.db"))
