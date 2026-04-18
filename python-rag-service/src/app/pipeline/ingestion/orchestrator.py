import glob
import os
from typing import List
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from src.app.core.worker_app import task_app
from src.app.db import LawUnitRepository
from src.app.pipeline.ingestion.parser import TrafficCodeParser
from src.app.schemas.law_unit import LawUnitCreate

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Facade Pattern for orchestrating the RAG ETL process.
    Implements the Transactional Outbox Pattern to decouple databases.
    """

    def __init__(self, pg_session: AsyncSession):
        self.pg_session = pg_session
        self.pg_repo = LawUnitRepository(self.pg_session)

    async def process_directory(self, raw_data_dir: str, trigger_tasks: bool = True) -> None:
        logger.info(f"--- Starting Ingestion from: {raw_data_dir} ---")

        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

        file_paths = glob.glob(os.path.join(raw_data_dir, "*.txt"))
        file_paths.sort()

        if not file_paths:
            logger.warning("No .txt files found.")
            return

        all_units: List[LawUnitCreate] = []

        # 1. Parse Files (Deterministic extraction)
        logger.info(f"Parsing {len(file_paths)} files...")
        for file_path in file_paths:
            parser = TrafficCodeParser(file_path)
            all_units.extend(parser.parse())

        if not all_units:
            return

        # 2. Transactional Outbox Pattern logic
        batch_size = 50
        for i in range(0, len(all_units), batch_size):
            batch = all_units[i:i + batch_size]

            # Upsert relational data
            await self.pg_repo.bulk_upsert(batch)

            if trigger_tasks:
                unit_ids = [u.id for u in batch]

                # Extract the raw psycopg AsyncConnection from the active SQLAlchemy session
                sa_conn = await self.pg_session.connection()
                raw_psycopg_conn = (await sa_conn.get_raw_connection()).driver_connection

                # Defer the tasks natively WHILE preserving the Outbox Pattern
                # by explicitly passing the locked SQLAlchemy connection
                await task_app.configure_task(
                    name="ingest_vectors_batch",
                    connection=raw_psycopg_conn
                ).defer_async(unit_ids=unit_ids)

                await task_app.configure_task(
                    name="ingest_graph_batch",
                    connection=raw_psycopg_conn
                ).defer_async(unit_ids=unit_ids)

                logger.info(f"Scheduled decoupled Vector and Graph tasks for batch of {len(batch)} units.")
            else:
                logger.info(f"Postgres-only mode: Skipped background tasks for {len(batch)} units.")

        logger.info("Ingestion complete. Background workers will handle AI enrichment.")
