import asyncio
import os
import logging

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.pipeline.ingestion import IngestionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    raw_data_dir = settings.RAW_DATA_DIR

    try:
        # Unit of Work Pattern: If anything fails here, Postgres rolls back.
        async with AsyncSessionLocal.begin() as pg_session:
            service = IngestionService(pg_session=pg_session)
            await service.process_directory(raw_data_dir)

    except Exception as e:
        logger.error(f"PIPELINE FAILED: Transaction rolled back. Error: {e}")


if __name__ == "__main__":
    # Ensure Windows uses the correct asyncio event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())