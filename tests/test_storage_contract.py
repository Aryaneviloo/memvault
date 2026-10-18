import os

import pytest

from memvault.core.models import MemoryItem, MemoryQuery, MemoryType
from memvault.storage.memory import InMemoryStorage
from memvault.storage.sqlite import SQLiteStorage


def _get_backends():
    """Return list of backend identifiers to test against."""
    backends = ["memory", "sqlite"]

    # Only include postgres if running (check env var set in CI/docker)
    pg_dsn = os.environ.get("TEST_POSTGRES_DSN")
    if pg_dsn:
        backends.append("postgres")

    return backends


@pytest.fixture(params=_get_backends())
def store(request):
    """Run every test against all available backends."""
    if request.param == "memory":
        yield InMemoryStorage()

    elif request.param == "sqlite":
        yield SQLiteStorage(":memory:")

    elif request.param == "postgres":
        from memvault.storage.postgres import PostgresStorage

        pg_dsn = os.environ.get("TEST_POSTGRES_DSN")
        pg_store = PostgresStorage(pg_dsn)
        yield pg_store
        # Cleanup after each test
        with pg_store._conn.cursor() as cur:
            cur.execute("DELETE FROM memories")
        pg_store._conn.commit()


@pytest.fixture
def sample_item():
    return MemoryItem(
        agent_id="agent-1",
        user_id="user-1",
        type=MemoryType.EPISODIC,
        content="User likes to code",
        tags=["preference", "language"],
        metadata={"source": "chat"},
    )


def test_insert_and_get(store, sample_item):
    store.insert(sample_item)
    fetched = store.get(sample_item.id)
    assert fetched is not None
    assert fetched.id == sample_item.id
    assert fetched.content == "User likes to code"
    assert fetched.tags == ["preference", "language"]
    assert fetched.metadata == {"source": "chat"}
    assert fetched.type == MemoryType.EPISODIC


def test_get_nonexistent_returns_none(store):
    assert store.get("does-not-exist") is None


def test_update_changes_content_and_timestamp(store, sample_item):
    store.insert(sample_item)
    original_updated_at = sample_item.updated_at
    sample_item.content = "User hates to code"
    updated = store.update(sample_item)

    assert updated.content == "User hates to code"
    assert updated.updated_at > original_updated_at

    refetched = store.get(sample_item.id)
    assert refetched.content == "User hates to code"


def test_soft_delete_hides_item_from_get(store, sample_item):
    store.insert(sample_item)
    deleted = store.delete(sample_item.id)

    assert deleted is True
    assert store.get(sample_item.id) is None


def test_delete_nonexistent_returns_false(store):
    assert store.delete("does-not-exist") is False


def test_search_matches_content_substring(store, sample_item):
    store.insert(sample_item)
    results = store.search(MemoryQuery(text="code", user_id="user-1"))
    assert len(results) == 1
    assert results[0].id == sample_item.id


def test_search_respects_user_isolation(store, sample_item):
    store.insert(sample_item)
    results = store.search(MemoryQuery(text="code", user_id="someone_else"))
    assert results == []


def test_search_excludes_soft_deleted(store, sample_item):
    store.insert(sample_item)
    store.delete(sample_item.id)

    results = store.search(MemoryQuery(text="code", user_id="user-1"))
    assert results == []


def test_search_filters_by_type(store, sample_item):
    store.insert(sample_item)
    other = MemoryItem(
        agent_id="agent-1",
        user_id="user-1",
        type=MemoryType.SEMANTIC,
        content="User likes Python too",
    )
    store.insert(other)

    results = store.search(
        MemoryQuery(text="python", user_id="user-1", types=[MemoryType.SEMANTIC])
    )
    assert len(results) == 1
    assert results[0].id == other.id


def test_list_recent_orders_by_creation_time(store):
    older = MemoryItem(agent_id="a", user_id="u", type=MemoryType.EPISODIC, content="first")
    newer = MemoryItem(agent_id="a", user_id="u", type=MemoryType.EPISODIC, content="second")

    store.insert(older)
    store.insert(newer)

    results = store.list_recent(user_id="u")
    assert results[0].id == newer.id
    assert results[1].id == older.id


def test_list_recent_respects_limit(store):
    for i in range(5):
        store.insert(
            MemoryItem(agent_id="a", 
                       user_id="u", 
                       type=MemoryType.EPISODIC, 
                       content=f"memory {i}",
                       )
        )

    results = store.list_recent(user_id="u", limit=3)
    assert len(results) == 3


def test_search_filters_by_agent_id(store):
    store.insert(
        MemoryItem(
            agent_id="agent-A", user_id="u", type=MemoryType.EPISODIC, content="shared topic"
        )
    )
    store.insert(
        MemoryItem(
            agent_id="agent-B", user_id="u", type=MemoryType.EPISODIC, content="shared topic"
        )
    )

    results = store.search(MemoryQuery(text="shared", user_id="u", agent_id="agent-A"))
    assert len(results) == 1
    assert results[0].agent_id == "agent-A"


def test_list_recent_filters_by_agent_id(store):
    store.insert(MemoryItem(agent_id="agent-A", user_id="u", type=MemoryType.EPISODIC, content="x"))
    store.insert(MemoryItem(agent_id="agent-B", user_id="u", type=MemoryType.EPISODIC, content="y"))

    results = store.list_recent(user_id="u", agent_id="agent-A")
    assert len(results) == 1
    assert results[0].agent_id == "agent-A"

def _mem(user_id = "u",
         agent_id = "a",
         namespace = "default",
         content = "X"):
    return MemoryItem(
        user_id=user_id,
        agent_id=agent_id,
        namespace=namespace,
        type = MemoryType.WORKING,
        content = content,
    )

def test_list_namespaces_sorted_and_deduped(store):
    store.insert(_mem(namespace="zeta"))
    store.insert(_mem(namespace="alpha", content ="1"))
    store.insert(_mem(namespace="alpha", content="2"))

    assert store.list_namespaces(user_id="u") == ["alpha", "zeta"]


def test_list_namespaces_isolated_by_user(store):
    store.insert(_mem(user_id="u", namespace="mine"))
    store.insert(_mem(user_id = "other", namespace="theirs"))

    assert store.list_namespaces(user_id="u") == ["mine"]

def test_list_namespaces_filters_by_agent_id(store):
    store.insert(_mem(agent_id="agent-A", namespace="chat-1"))
    store.insert(_mem(agent_id="agent-B", namespace="chat-2"))

    assert store.list_namespaces(user_id="u", agent_id="agent-A") == ["chat-1"]
    assert store.list_namespaces(user_id="u") == ["chat-1", "chat-2"]


def test_list_namespaces_ignores_soft_deleted(store):
    gone = store.insert(_mem(namespace="gone"))
    store.insert(_mem(namespace="kept"))
    store.delete(gone.id)  # soft delete

    assert store.list_namespaces(user_id="u") == ["kept"]

    

def test_hard_delete_permanently_removes_item(store, sample_item):
    store.insert(sample_item)
    deleted = store.delete(sample_item.id, hard=True)

    assert deleted is True
    assert store.get(sample_item.id) is None


def test_sqlite_persists(tmp_path):
    """Data written by one connection should be readbale by new connetion"""

    db_file = os.path.join(tmp_path, "test-memories.db")

    store1 = SQLiteStorage(db_file)
    item = MemoryItem(
        agent_id="a", 
        user_id="u", 
        type=MemoryType.EPISODIC, 
        content="persisted memory",
    )
    store1.insert(item)

    store2 = SQLiteStorage(db_file)
    fetched = store2.get(item.id)

    assert fetched is not None
    assert fetched.content == "persisted memory"


def test_search_on_empty_store_finds_empty_list(store):
    results = store.search(MemoryQuery(text="anything", user_id="u"))
    assert results == []


def test_list_recent_on_empty_returns_empty_list(store):
    results = store.list_recent(user_id="u")
    assert results == []
