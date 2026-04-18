import typer
import asyncio
from src.app.core.config import settings
from src.cli.utils import setup_windows_asyncio

app = typer.Typer(help="Danger zone: Wipe databases and collections.")


@app.command("postgres")
def wipe_postgres():
    """DANGER: Wipes all tables from PostgreSQL."""
    typer.confirm("Are you sure you want to wipe PostgreSQL?", abort=True)
    from src.app.db.session import engine, Base

    async def _run():
        typer.secho("Wiping Postgres...", fg=typer.colors.YELLOW)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        typer.secho("✅ Postgres successfully wiped and reset.", fg=typer.colors.GREEN)

    setup_windows_asyncio()
    asyncio.run(_run())


@app.command("qdrant")
def wipe_qdrant():
    """DANGER: Wipes the RAG collection from Qdrant."""
    typer.confirm("Are you sure you want to wipe Qdrant?", abort=True)
    from qdrant_client import AsyncQdrantClient

    async def _run():
        typer.secho("Wiping Qdrant...", fg=typer.colors.YELLOW)
        q_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        try:
            if await q_client.collection_exists(settings.QDRANT_COLLECTION):
                await q_client.delete_collection(settings.QDRANT_COLLECTION)
                typer.secho("✅ Qdrant collection successfully deleted.", fg=typer.colors.GREEN)
            else:
                typer.secho("🔵 Qdrant collection did not exist.", fg=typer.colors.YELLOW)
        finally:
            await q_client.close()

    setup_windows_asyncio()
    asyncio.run(_run())


@app.command("neo4j")
def wipe_neo4j():
    """DANGER: Wipes all nodes and relationships from Neo4j."""
    typer.confirm("Are you sure you want to wipe Neo4j?", abort=True)
    from neo4j import AsyncGraphDatabase # type: ignore

    async def _run():
        typer.secho("Wiping Neo4j...", fg=typer.colors.YELLOW)
        n_driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        try:
            async with n_driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            typer.secho("✅ Neo4j graph successfully wiped.", fg=typer.colors.GREEN)
        finally:
            await n_driver.close()

    setup_windows_asyncio()
    asyncio.run(_run())


@app.command("all")
def wipe_all():
    """DANGER: Wipes Postgres, Qdrant, AND Neo4j simultaneously."""
    typer.confirm("Are you sure you want to completely wipe ALL databases?", abort=True)
    from qdrant_client import AsyncQdrantClient
    from neo4j import AsyncGraphDatabase # type: ignore
    from src.app.db.session import engine, Base

    async def _run():
        typer.secho("Initiating full system wipe...", fg=typer.colors.RED, bold=True)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            typer.secho(" - Postgres wiped.", fg=typer.colors.YELLOW)

        q_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        try:
            if await q_client.collection_exists(settings.QDRANT_COLLECTION):
                await q_client.delete_collection(settings.QDRANT_COLLECTION)
            typer.secho(" - Qdrant wiped.", fg=typer.colors.YELLOW)
        finally:
            await q_client.close()

        n_driver = AsyncGraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        try:
            async with n_driver.session() as session:
                await session.run("MATCH (n) DETACH DELETE n")
            typer.secho(" - Neo4j wiped.", fg=typer.colors.YELLOW)
        finally:
            await n_driver.close()

        typer.secho("✅ All databases successfully wiped.", fg=typer.colors.GREEN)

    setup_windows_asyncio()
    asyncio.run(_run())