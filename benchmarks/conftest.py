import os
import tempfile
import pytest
from memvault.core.models import MemoryItem, MemoryType
from memvault.embeddings.local import LocalEmbedder
from memvault.storage.base import EmbeddingStorageWrapper
from memvault.storage.memory import InMemoryStorage
from memvault.storage.sqlite import SQLiteStorage


@pytest.fixture
def sample_item():
    """Provides a standard MemoryItem to insert."""
    return MemoryItem(
        agent_id="bench-agent",
        user_id="bench-user",
        type=MemoryType.SEMANTIC,
        content="User prefers Python over JavaScript",
        importance=0.8,
    )


@pytest.fixture
def in_memory_setup():
    """Sets up an in-memory storage instance."""
    backend = InMemoryStorage()
    embedder = LocalEmbedder()
    wrapper = EmbeddingStorageWrapper(backend=backend, embedder=embedder)
    return wrapper, backend, embedder


@pytest.fixture
def sqlite_setup():
    """Sets up a temporary SQLite database for benchmarking."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    backend = SQLiteStorage(db_path)
    embedder = LocalEmbedder()
    wrapper = EmbeddingStorageWrapper(backend=backend, embedder=embedder)

    yield wrapper, backend, embedder

    # Close SQLite database connection prior to file removal
    if hasattr(backend, "close"):
        backend.close()
    elif hasattr(backend, "conn") and backend.conn:
        backend.conn.close()

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass  # Windows OS temporary file handle cleanup