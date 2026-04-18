# app/api/dependencies.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from src.app.db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides an async database session per request.
    Implements the Unit of Work pattern: automatically commits on success,
    and rolls back on any exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            # Hand the session to the FastAPI endpoint
            yield session

            # If the endpoint finishes without errors, commit the transaction
            await session.commit()

        except Exception:
            # If the endpoint crashes for ANY reason, undo everything
            await session.rollback()
            raise