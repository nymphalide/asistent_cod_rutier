import os
import glob
import logging
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.law_unit import LawUnitCreate
from app.pipeline.ingestion.parser import TrafficCodeParser
from app.db.repository import LawUnitRepository
from app.db.queue import task_app

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Facade Pattern for orchestrating the RAG ETL process.
    Implements the Transactional Outbox Pattern to decouple databases.
    """

    def __init__(self, pg_session: AsyncSession):
        self.pg_session = pg_session
        self.pg_repo = LawUnitRepository(self.pg_session)

    async def process_directory(self, raw_data_dir: str) -> None:
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

            unit_ids = [u.id for u in batch]

            # Defer the async task to the Procrastinate queue
            await task_app.configure(
                task_kwargs={"unit_ids": unit_ids}
            ).defer_async(name="process_knowledge_batch")

            logger.info(f"Scheduled worker task for batch of {len(batch)} units.")

        logger.info("Ingestion complete. Background workers will handle AI enrichment.")