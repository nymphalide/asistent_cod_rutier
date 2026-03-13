import os
import sys
import pytest
import asyncio
from sqlalchemy import select, func

# 1. Setup the Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pipeline.ingestion import run_ingestion_pipeline
from app.db.session import AsyncSessionLocal
from app.db.models import LawUnit
from app.core.custom_types import UnitType

RAW_DATA_DIR = "data/raw_text"


# This decorator tells pytest to spin up an event loop for this specific test
@pytest.mark.asyncio
async def test_ingestion_execution():
    """
    Integration Test:
    1. Checks if raw data exists.
    2. Runs the full ingestion pipeline (Parser -> DB).
    3. Connects to the DB to verify data was actually saved.
    """

    # A. Pre-check
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        with open(os.path.join(RAW_DATA_DIR, "dummy_test.txt"), "w", encoding="utf-8") as f:
            f.write("CAPITOLUL I: DISPOZITII GENERALE\nArt. 1.\n(1) Circulatia pe drumurile publice...")
        print(f"Created dummy test file in {RAW_DATA_DIR}")

    # B. Execution (Now awaited!)
    processed_units = await run_ingestion_pipeline(RAW_DATA_DIR)

    # C. Assertions (In-Memory)
    assert isinstance(processed_units, list)
    assert len(processed_units) > 0, "Pipeline returned empty list"

    print(f"\n[TEST] In-memory checks passed. Processed {len(processed_units)} units.")

    # D. Database Verification
    # Using the Context Manager pattern to safely open and close the async connection
    async with AsyncSessionLocal() as db:
        # 1. Check total count using SQLAlchemy 2.0 syntax
        stmt_count = select(func.count()).select_from(LawUnit)
        result_count = await db.execute(stmt_count)
        count = result_count.scalar_one()

        assert count > 0, "Database table 'law_units' is empty!"

        # 2. Check a specific unit type exists
        stmt_chapter = select(LawUnit).where(LawUnit.unit_type == UnitType.CHAPTER)
        result_chapter = await db.execute(stmt_chapter)
        chapter = result_chapter.scalar_one_or_none()

        assert chapter is not None, "No chapters found in DB"

        print(f"[TEST] Database verification passed. Found {count} rows in 'law_units'.")
        print(f"[TEST] Sample Chapter: {chapter.content}")


# The Entry Point Pattern for running the script directly (outside of pytest)
if __name__ == "__main__":
    asyncio.run(test_ingestion_execution())