"""
Async PostgreSQL database configuration and session management.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy import event, pool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import StaticPool

from app.config import settings

logger = logging.getLogger(__name__)

# Create the declarative base
Base = declarative_base()

# Global variables for database
async_engine: Optional[AsyncEngine] = None
async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


def create_engine() -> AsyncEngine:
    """Create async database engine with optimized configuration."""
    engine_kwargs = {
        "echo": settings.database_echo,
        "future": True,
        "pool_pre_ping": True,
        "pool_recycle": 3600,  # Recycle connections every hour
    }

    # Configure connection pool based on environment
    if settings.is_production:
        engine_kwargs.update({
            "poolclass": pool.QueuePool,
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_max_overflow,
            "pool_timeout": 30,
        })
    elif settings.is_testing:
        # Use static pool for testing to avoid connection issues
        engine_kwargs.update({
            "poolclass": StaticPool,
            "connect_args": {"check_same_thread": False},
        })
    else:
        # Development configuration
        engine_kwargs.update({
            "pool_size": 5,
            "max_overflow": 10,
        })

    engine = create_async_engine(settings.database_url, **engine_kwargs)

    # Add event listeners for connection management
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set SQLite pragma for better performance (if using SQLite)."""
        if "sqlite" in settings.database_url:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    @event.listens_for(engine.sync_engine, "checkout")
    def check_connection(dbapi_connection, connection_record, connection_proxy):
        """Check connection health on checkout."""
        pass  # Could add connection health checks here

    return engine


def create_session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create async session maker."""
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )


async def init_db() -> None:
    """Initialize database connection."""
    global async_engine, async_session_maker

    try:
        logger.info("Initializing database connection...")
        async_engine = create_engine()
        async_session_maker = create_session_maker(async_engine)

        # Test the connection
        async with async_engine.begin() as conn:
            # Execute a simple query to test connection
            await conn.execute("SELECT 1")

        logger.info("Database connection initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db() -> None:
    """Close database connection."""
    global async_engine

    if async_engine:
        logger.info("Closing database connection...")
        await async_engine.dispose()
        logger.info("Database connection closed")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.

    Yields:
        AsyncSession: Database session

    Raises:
        Exception: If session creation fails
    """
    if not async_session_maker:
        raise RuntimeError("Database not initialized")

    async with async_session_maker() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database session.

    Yields:
        AsyncSession: Database session
    """
    if not async_session_maker:
        raise RuntimeError("Database not initialized")

    async with async_session_maker() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database context error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


class DatabaseManager:
    """Database manager for handling connections and sessions."""

    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_maker: Optional[async_sessionmaker[AsyncSession]] = None

    async def connect(self) -> None:
        """Connect to database."""
        self.engine = create_engine()
        self.session_maker = create_session_maker(self.engine)

        # Test connection
        async with self.engine.begin() as conn:
            await conn.execute("SELECT 1")

        logger.info("Database manager connected")

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_maker = None
            logger.info("Database manager disconnected")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager."""
        if not self.session_maker:
            raise RuntimeError("Database manager not connected")

        async with self.session_maker() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Database session error: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.engine:
                return False

            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
            return True

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


# Utility functions for testing
async def create_test_engine() -> AsyncEngine:
    """Create engine for testing."""
    return create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables in the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables(engine: AsyncEngine) -> None:
    """Drop all tables in the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Database utilities for migrations and management
class DatabaseUtils:
    """Utility functions for database operations."""

    @staticmethod
    async def check_connection(engine: AsyncEngine) -> bool:
        """Check if database connection is working."""
        try:
            async with engine.begin() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    @staticmethod
    async def get_table_names(engine: AsyncEngine) -> list[str]:
        """Get list of table names in the database."""
        async with engine.begin() as conn:
            result = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
                if "sqlite" in str(engine.url)
                else "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            )
            return [row[0] for row in result.fetchall()]

    @staticmethod
    async def get_database_size(engine: AsyncEngine) -> Optional[int]:
        """Get database size in bytes."""
        try:
            async with engine.begin() as conn:
                if "postgresql" in str(engine.url):
                    result = await conn.execute(
                        "SELECT pg_database_size(current_database())"
                    )
                    return result.scalar()
                elif "sqlite" in str(engine.url):
                    # SQLite doesn't have built-in size function
                    return None
            return None
        except Exception:
            return None