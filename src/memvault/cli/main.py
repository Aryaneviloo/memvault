import typer
from rich import print as rprint
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from memvault.core.consolidation import ConsolidationConfig, consolidate
from memvault.core.models import MemoryItem, MemoryQuery, MemoryType
from memvault.core.retrieval import retrieve
from memvault.embeddings.local import LocalEmbedder
from memvault.storage.base import EmbeddingStorageWrapper
from memvault.storage.sqlite import SQLiteStorage

app = typer.Typer(
    name="memvault",
    help="Open-source memory infrastructure for AI agents.",
    add_completion=False,
)

console = Console()

def _get_embedder() -> LocalEmbedder:
    """Build the embedder, exiting cleanly if the [local] extra is missing."""
    try:
        return LocalEmbedder()
    except ImportError as e:
        rprint(f"[red]✗[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None

def _get_backend(db_path: str = "memories.db") -> SQLiteStorage:
    """Storage only. No embedder needed."""
    return SQLiteStorage(db_path)



def _get_store(db_path: str = "memories.db") -> EmbeddingStorageWrapper:
    """Full stack: embedder + SQLite."""
    embedder = _get_embedder()
    return EmbeddingStorageWrapper(backend=_get_backend(db_path), embedder=embedder)


@app.command()
def remember(
    content: str = typer.Argument(..., help="The memory content to store"),
    user_id: str = typer.Option("default-user", "--user", "-u", help="User ID"),
    agent_id: str = typer.Option("default-agent", "--agent", "-a", help="Agent ID"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Memory namespace"),
    importance: float = typer.Option(0.5, "--importance", "-i", help="Importance (0.0-1.0)"),
    memory_type: str = typer.Option("episodic", "--type", "-t", help="Memory type"),
    db_path: str = typer.Option("memories.db", "--db", help="Path to SQLite database"),
) -> None:
    """Store a new memory."""

    try:
        mem_type = MemoryType(memory_type)
    except ValueError:
        valid = [t.value for t in MemoryType]
        typer.echo(f"Invalid type '{memory_type}'. Valid types: {valid}", err=True)
        raise typer.Exit(code=1) from None

    store = _get_store(db_path)
    item = MemoryItem(
        agent_id=agent_id,
        user_id=user_id,
        namespace=namespace,
        type=mem_type,
        content=content,
        importance=importance,
    )
    stored = store.insert(item)
    rprint(f"[green]✓[/green] Memory stored with ID: [bold]{stored.id}[/bold]")


@app.command()
def recall(
    query: str = typer.Argument(..., help="What to search for"),
    user_id: str = typer.Option("default-user", "--user", "-u", help="User ID"),
    agent_id: str | None = typer.Option(None, "--agent", "-a", help="Agent ID"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Namespace"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    db_path: str = typer.Option("memories.db", "--db", help="Path to SQLite database"),
) -> None:
    """Search memories semantically."""

    embedder = _get_embedder()
    backend = _get_backend(db_path)

    memory_query = MemoryQuery(
        text=query,
        user_id=user_id,
        agent_id=agent_id,
        namespace=namespace,
        top_k=top_k,
    )

    results = retrieve(query=memory_query, 
                       backend=backend, 
                       embedder=embedder)

    if not results:
        rprint("[yellow]No memories found.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Results for: '{query}'", show_lines=True)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Content", style="white")
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Importance", style="yellow", width=10)
    table.add_column("ID", style="dim", width=36)

    for r in results:
        table.add_row(
            str(r.final_score),
            r.item.content,
            r.item.type.value,
            str(round(r.item.importance, 2)),
            r.item.id,
        )

    console.print(table)


@app.command()
def forget(
    memory_id: str = typer.Argument(..., help="ID of the memory to delete"),
    hard: bool = typer.Option(False, "--hard", help="Permanently delete (no recovery)"),
    db_path: str = typer.Option("memories.db", "--db", help="Path to SQLite database"),
) -> None:
    """Delete a memory by ID."""

    backend = _get_backend(db_path)

    if hard:
        confirmed = typer.confirm(
            f"Permanently delete memory {memory_id!r}? This cannot be undone."
        )
        if not confirmed:
            rprint("[yellow]Aborted.[/yellow]")
            raise typer.Exit()

    deleted = backend.delete(memory_id, hard=hard)
    if deleted:
        action = "permanently deleted" if hard else "soft-deleted"
        rprint(f"[green]✓[/green] Memory {action}: [bold]{memory_id}[/bold]")
    else:
        rprint(f"[red]✗[/red] Memory not found: [bold]{memory_id}[/bold]")
        raise typer.Exit(code=1) from None


@app.command(name="consolidate")
def consolidate_cmd(
    user_id: str = typer.Option("default-user", "--user", "-u", help="User ID"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Namespace"),
    threshold: float = typer.Option(0.85, "--threshold", "-t", help="Similarity threshold"),
    db_path: str = typer.Option("memories.db", "--db", help="Path to SQLite database"),
) -> None:
    """Merge near-duplicate memories for a user."""

    backend = _get_backend(db_path)
    config = ConsolidationConfig(similarity_threshold=threshold)

    with console.status("Running consolidation..."):
        result = consolidate(
            user_id=user_id,
            backend=backend,
            namespace=namespace,
            config=config,
        )

    if result.memories_consolidated == 0:
        rprint("[yellow]No memories to consolidate.[/yellow]")
    else:
        rprint(
            f"[green]✓[/green] Consolidated [bold]{result.memories_consolidated}[/bold]"
            f" memories into [bold]{result.clusters_found}[/bold] clusters."
        )

@app.command()
def doctor(
    db_path: str = typer.Option("memories.db", "--db", help="Path to SQLite database"),
) -> None:
    """Check the health of the memvault installation."""

    console.print("[bold]MemVault Doctor[/bold]\n")
    ok = True

    try:
        SQLiteStorage(db_path)
        rprint("[green]✓[/green] Storage backend: SQLite")
    except Exception as e:
        ok = False
        rprint(f"[red]✗[/red] Storage backend failed: {escape(str(e))}")

    try:
        embedder = LocalEmbedder()
        vec = embedder.embed("test")
        assert len(vec) == 384
        rprint("[green]✓[/green] Embedding model: BGE-small (384 dims)")
    except Exception as e:
        ok = False
        rprint(f"[red]✗[/red] Embedding model failed: {escape(str(e))}")

    if ok:
        rprint("\n[bold green]All systems operational.[/bold green]")
    else:
        rprint("\n[bold red]Problems found.[/bold red]")
        raise typer.Exit(code=1)

def main():
    app()
