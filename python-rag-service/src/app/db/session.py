from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from src.app.core.config import settings

# create_async_engine replaces create_engine
engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)

# async_sessionmaker replaces sessionmaker
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

Base = declarative_base()