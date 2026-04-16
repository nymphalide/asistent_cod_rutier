import typer
import asyncio
from app.core.config import settings
from cli.utils import setup_windows_asyncio

app = typer.Typer(help="Check system health and statistics.")


@app.command("stats")
def stats():
    """Displays current record counts across Postgres, Qdrant, and Neo4j."""
    from sqlalchemy import select, func
    from app.db.session import AsyncSessionLocal
    from app.db.models import LawUnit
    from qdrant_client import AsyncQdrantClient
    from neo4j import AsyncGraphDatabase

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