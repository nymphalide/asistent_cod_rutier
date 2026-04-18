import typer
import asyncio
from src.app.core.config import settings
from src.cli.utils import setup_windows_asyncio

app = typer.Typer(help="Check system health and statistics.")


@app.command("stats")
def stats():
    """Displays current record counts across Postgres, Qdrant, and Neo4j."""
    from sqlalchemy import select, func
    from src.app.db.session import AsyncSessionLocal
    from src.app.db.models import LawUnit
    from qdrant_client import AsyncQdrantClient
    from neo4j import AsyncGraphDatabase # type: ignore

    async def _run():
        typer.secho("--- Database Statistics ---", bold=True)

        try:
            async with AsyncSessionLocal() as db:
                pg_count = await db.scalar(select(func.count()).select_from(LawUnit))
                typer.secho(f"🐘 Postgres LawUnits: {pg_count}", fg=typer.colors.CYAN)
        except Exception:
            typer.secho("🐘 Postgres: Unreachable", fg=typer.colors.RED)

        q_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        try:
            if await q_client.collection_exists(settings.QDRANT_COLLECTION):
                info = await q_client.get_collection(settings.QDRANT_COLLECTION)
                typer.secho(f"🔵 Qdrant Vectors: {info.points_count}", fg=typer.colors.CYAN)
            else:
                typer.secho("🔵 Qdrant: Collection does not exist", fg=typer.colors.YELLOW)
        except Exception:
            typer.secho("🔵 Qdrant: Unreachable", fg=typer.colors.RED)
        finally:
            await q_client.close()

        n_driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        try:
            async with n_driver.session() as session:
                result = await session.run("MATCH (n) RETURN count(n) as count")
                record = await result.single()
                typer.secho(f"🕸️ Neo4j Nodes: {record['count']}", fg=typer.colors.CYAN)
        except Exception:
            typer.secho("🕸️ Neo4j: Unreachable", fg=typer.colors.RED)
        finally:
            await n_driver.close()

    setup_windows_asyncio()
    asyncio.run(_run())


@app.command("reconcile")
def reconcile(auto_fix: bool = typer.Option(False, "--fix", help="Automatically queue missing chunks")):
    """Detects data drift and optionally requeues missing entities."""
    from sqlalchemy import select
    from src.app.db.session import AsyncSessionLocal
    from src.app.db.models import LawUnit
    from src.app.core.worker_app import task_app
    from src.app.core.config import settings
    from qdrant_client import AsyncQdrantClient
    from neo4j import AsyncGraphDatabase # type: ignore
    import asyncio
    from src.cli.utils import setup_windows_asyncio  # Adjust import path if needed

    async def _run():
        typer.secho("🔍 Starting Data Reconciliation...", bold=True)

        # 1. Get Source of Truth (Postgres)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(LawUnit.id))
            pg_ids = {row[0] for row in result.all()}
        typer.secho(f"🐘 Postgres Source of Truth: {len(pg_ids)} units", fg=typer.colors.CYAN)

        # 2. Get Qdrant IDs (Using PyCharm-friendly pagination)
        qdrant_ids = set()
        q_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        try:
            if await q_client.collection_exists(settings.QDRANT_COLLECTION):
                current_offset = None
                is_first_batch = True

                while is_first_batch or current_offset is not None:
                    is_first_batch = False
                    records, current_offset = await q_client.scroll(
                        collection_name=settings.QDRANT_COLLECTION,
                        limit=1000,
                        with_payload=["unit_id"],
                        with_vectors=False,
                        offset=current_offset
                    )

                    for r in records:
                        if "unit_id" in r.payload:
                            qdrant_ids.add(r.payload["unit_id"])
        finally:
            await q_client.close()

        # 3. Get Neo4j IDs
        neo4j_ids = set()
        n_driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        try:
            async with n_driver.session() as session:
                result = await session.run("MATCH (n:LawUnit) RETURN n.id as id")
                async for record in result:
                    neo4j_ids.add(record["id"])
        finally:
            await n_driver.close()

        # 4. Calculate Drift (Set Subtraction)
        missing_in_qdrant = list(pg_ids - qdrant_ids)
        missing_in_neo4j = list(pg_ids - neo4j_ids)

        if not missing_in_qdrant and not missing_in_neo4j:
            typer.secho("✅ System is perfectly synced! No drift detected.", fg=typer.colors.GREEN)
            return

        if missing_in_qdrant:
            typer.secho(f"⚠️  Qdrant is missing {len(missing_in_qdrant)} units.", fg=typer.colors.YELLOW)
        if missing_in_neo4j:
            typer.secho(f"⚠️  Neo4j is missing {len(missing_in_neo4j)} units.", fg=typer.colors.YELLOW)

        # 5. Requeue if requested
        if auto_fix:
            typer.secho("⚙️  Auto-fix enabled. Queuing missing tasks...", bold=True)
            async with task_app.open_async():

                # Chunk into batches of 50 to match your standard ingestion flow
                def chunker(seq, size):
                    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

                if missing_in_qdrant:
                    for batch in chunker(missing_in_qdrant, 50):
                        await task_app.configure_task(name="ingest_vectors_batch",
                                                      task_kwargs={"unit_ids": batch}).defer_async()
                    typer.secho("✅ Queued Vector recovery tasks.", fg=typer.colors.GREEN)

                if missing_in_neo4j:
                    for batch in chunker(missing_in_neo4j, 50):
                        await task_app.configure_task(name="ingest_graph_batch",
                                                      task_kwargs={"unit_ids": batch}).defer_async()
                    typer.secho("✅ Queued Graph recovery tasks.", fg=typer.colors.GREEN)
        else:
            typer.secho("💡 Run with '--fix' to automatically queue background tasks for the missing data.", italic=True)

    setup_windows_asyncio()
    asyncio.run(_run())