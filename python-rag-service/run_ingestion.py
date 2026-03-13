import asyncio
import os
import logging
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.vector import QdrantRepository
from app.pipeline.ingestion import IngestionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 1. Directory in which the raw data sits
    raw_data_dir = settings.RAW_DATA_DIR

    # 2. Initialize the Qdrant Client (External Connection)
    logger.info("Connecting to Vector Database...")
    qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    qdrant_repo = QdrantRepository(client=qdrant_client)
    await qdrant_repo.initialize_collection()

    try:
        # 3. Initialize the Postgres Session (Unit of Work Pattern)
        # .begin() automatically manages the transaction (commits on success, rolls back on error)
        async with AsyncSessionLocal.begin() as pg_session:

            # Inject dependencies into the Facade
            service = IngestionService(pg_session=pg_session, qdrant_repo=qdrant_repo)

            # Execute the pipeline
            await service.process_directory(raw_data_dir)

    except Exception as e:
        logger.error(f"PIPELINE FAILED: Distributed transaction rolled back. Error: {e}")
    finally:
        # Clean up the network connection pool
        await qdrant_client.close()
        logger.info("Connections closed.")


if __name__ == "__main__":
    # Ensure Windows uses the correct asyncio event loop policy for certain networking tasks
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())