import typer
import asyncio
from typing import List

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.pipeline.ingestion import IngestionService
from app.core.worker_app import task_app
from cli.utils import setup_windows_asyncio

app = typer.Typer(help="Trigger data ingestion and background tasks.")

@app.command("all")
def ingest_all(data_dir: str = typer.Option(settings.RAW_DATA_DIR, "--dir", "-d")):
    """Parses raw text, saves to Postgres, and queues Vector & Graph tasks."""
    async def _run():
        try:
            async with task_app.open_async():
                async with AsyncSessionLocal.begin() as pg_session:
                    service = IngestionService(pg_session=pg_session)
                    await service.process_directory(data_dir, trigger_tasks=True)
            typer.secho("✅ Full ingestion pipeline triggered successfully.", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"❌ Ingestion failed: {e}", fg=typer.colors.RED)

    setup_windows_asyncio()
    asyncio.run(_run())


@app.command("postgres")
def ingest_postgres(data_dir: str = typer.Option(settings.RAW_DATA_DIR, "--dir", "-d")):
    """Parses text and saves to Postgres ONLY. Bypasses ML and Graph tasks."""
    async def _run():
        try:
            async with AsyncSessionLocal.begin() as pg_session:
                service = IngestionService(pg_session=pg_session)
                await service.process_directory(data_dir, trigger_tasks=False)
            typer.secho("✅ Postgres-only ingestion complete.", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"❌ Ingestion failed: {e}", fg=typer.colors.RED)

    setup_windows_asyncio()
    asyncio.run(_run())


@app.command("vector")
def run_vector_task(unit_ids: List[str] = typer.Argument(..., help="List of Postgres IDs (e.g. art_5)")):
    """Manually triggers the Qdrant Upsert task for specific IDs."""
    from app.pipeline.ingestion.tasks import ingest_vectors_batch_task
    async def _run():
        typer.secho(f"🚀 Forcing Vector Task for {len(unit_ids)} units...", fg=typer.colors.CYAN)
        await ingest_vectors_batch_task(unit_ids=unit_ids)
        typer.secho("✅ Vector task complete.", fg=typer.colors.GREEN)

    setup_windows_asyncio()
    asyncio.run(_run())


@app.command("graph")
def run_graph_task(unit_ids: List[str] = typer.Argument(..., help="List of Postgres IDs (e.g. art_5)")):
    """Manually triggers the Neo4j Upsert task for specific IDs."""
    from app.pipeline.ingestion.tasks import ingest_graph_batch_task
    async def _run():
        typer.secho(f"🚀 Forcing Graph Task for {len(unit_ids)} units...", fg=typer.colors.CYAN)
        await ingest_graph_batch_task(unit_ids=unit_ids)
        typer.secho("✅ Graph task complete.", fg=typer.colors.GREEN)

    setup_windows_asyncio()
    asyncio.run(_run())