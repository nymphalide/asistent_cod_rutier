import asyncio
import os
import logging

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.pipeline.ingestion import IngestionService
from app.db.queue import task_app  # IMPORTĂM task_app aici

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    raw_data_dir = settings.RAW_DATA_DIR

    try:
        # RESOURCE MANAGEMENT PATTERN: Deschidem conexiunea Procrastinate
        # Aceasta este "ușa" prin care trimitem biletele către coada de așteptare.
        async with task_app.open_async():

            # UNIT OF WORK PATTERN: Începem tranzacția PostgreSQL
            async with AsyncSessionLocal.begin() as pg_session:
                service = IngestionService(pg_session=pg_session)
                await service.process_directory(raw_data_dir)

            logger.info("✅ Ingestia a reușit și sarcinile au fost puse în coadă.")

    except Exception as e:
        logger.error(f"❌ PIPELINE FAILED: Tranzacția a fost anulată. Eroare: {e}")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())