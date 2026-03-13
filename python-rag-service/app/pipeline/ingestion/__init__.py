import os
import glob
import logging
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.law_unit import LawUnitCreate, LawUnitEnriched
from app.pipeline.ingestion.parser import TrafficCodeParser
from app.pipeline.ingestion.enricher import EnricherService
from app.db.repository import LawUnitRepository
from app.db.vector import QdrantRepository

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Facade Pattern for orchestrating the RAG ETL process.
    Uses strict Dependency Injection for database connections.
    """

    def __init__(self, pg_session: AsyncSession, qdrant_repo: QdrantRepository):
        self.pg_session = pg_session
        self.pg_repo = LawUnitRepository(self.pg_session)
        self.qdrant_repo = qdrant_repo
        self.enricher = EnricherService()

    async def process_directory(self, raw_data_dir: str) -> List[LawUnitEnriched]:
        logger.info(f"--- Starting Ingestion from: {raw_data_dir} ---")

        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"Directory not found: {raw_data_dir}")

        # 1. Gather Files
        file_paths = glob.glob(os.path.join(raw_data_dir, "*.txt"))
        file_paths.sort()

        if not file_paths:
            logger.warning("No .txt files found.")
            return []

        all_units: List[LawUnitCreate] = []

        # 2. Parse Files
        logger.info(f"Parsing {len(file_paths)} files...")
        for file_path in file_paths:
            logger.info(f" -> Processing: {os.path.basename(file_path)}")
            parser = TrafficCodeParser(file_path)
            file_units = parser.parse()
            all_units.extend(file_units)

        logger.info(f"Parsing complete. Extracted {len(all_units)} units.")

        # 3. Enrich (Embeddings & Hypothetical Questions)
        logger.info("Enriching data with Embeddings and Questions...")
        enriched_units = await self.enricher.enrich_batch(all_units)

        # 4. Save to Database (Saga Pattern / Distributed Transaction)
        logger.info("Writing to Databases...")

        # We do not use 'try...except' to commit/rollback here.
        # The Unit of Work pattern dictates that the caller managing the session handles the transaction.
        count = await self.pg_repo.bulk_upsert(enriched_units)
        await self.qdrant_repo.bulk_upsert(enriched_units)

        logger.info(f"SUCCESS: Mapped vectors to Qdrant and prepped {count} units for Postgres commit.")

        return enriched_units