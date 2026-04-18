import typer
import asyncio
from rich.console import Console
from rich.table import Table

from src.app.core.config import settings
from src.app.clients.llm_gateway import LLMGateway
from src.app.pipeline.retrieval.strategies.qdrant_hybrid import QdrantHybridStrategy
from src.app.schemas.retrieval import RetrievalRequest
from qdrant_client import AsyncQdrantClient
from src.cli.utils import setup_windows_asyncio

app = typer.Typer(help="Query the RAG retrieval engine.")
console = Console()


@app.command("hybrid")
def search_hybrid(
        query: str = typer.Argument(..., help="The natural language query"),
        limit: int = typer.Option(5, "--limit", "-l")
):
    """Test the Qdrant Hybrid (Dense + Sparse) retrieval strategy."""

    async def _run():
        # Initialize clients
        q_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        gateway = LLMGateway()

        strategy = QdrantHybridStrategy(client=q_client, llm_gateway=gateway)
        request = RetrievalRequest(query_text=query, top_k=limit)

        console.print(f"[bold cyan]🔍 Searching for:[/bold cyan] {query}")

        results = await strategy.retrieve(request)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(title="Qdrant Hybrid Search Results")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Unit ID", style="magenta")
        table.add_column("Content Snippet", style="white")

        for res in results:
            table.add_row(
                f"{res.score:.4f}",
                res.unit_id,
                (res.content[:100] + "...") if len(res.content) > 100 else res.content
            )

        console.print(table)
        await q_client.close()

    setup_windows_asyncio()
    asyncio.run(_run())