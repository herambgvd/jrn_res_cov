"""
Redis connection and cache management.
"""

import json
import logging
import pickle
from typing import Any, Optional, Union, Dict, List
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

from app.config import settings

logger = logging.getLogger(__name__)

# Global Redis instances
redis_client: Optional[Redis] = None
redis_cache: Optional[Redis] = None


class RedisManager:
    """Redis connection and cache management."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client: Optional[Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Create Redis client
            self.client = Redis(
                connection_pool=self.connection_pool,
                socket_keepalive=True,
                socket_keepalive_options={},
            )

            # Test connection
            await self.client.ping()
            logger.info(f"Redis connected successfully to {self.redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            if not self.client:
                return False
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


class CacheManager:
    """High-level cache management with serialization support."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_ttl = settings.cache_default_ttl

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with automatic deserialization.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            value = await self.redis.get(key)
            if value is None:
                return default

            # Try to deserialize JSON first
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # If JSON fails, try pickle
                try:
                    return pickle.loads(value.encode())
                except (pickle.PickleError, AttributeError):
                    # Return as string if all else fails
                    return value

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default

    async def set(
            self,
            key: str,
            value: Any,
            ttl: Optional[int] = None,
            serialize_method: str = "json"
    ) -> bool:
        """
        Set value in cache with automatic serialization.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize_method: Serialization method ("json" or "pickle")

        Returns:
            True if successful
        """
        try:
            ttl = ttl or self.default_ttl

            # Serialize value
            if serialize_method == "json":
                try:
                    serialized_value = json.dumps(value)
                except (TypeError, ValueError):
                    # Fall back to pickle if JSON fails
                    serialized_value = pickle.dumps(value)
            else:
                serialized_value = pickle.dumps(value)

            await self.redis.setex(key, ttl, serialized_value)
            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter in cache."""
        try:
            # Use pipeline for atomic operation
            pipe = self.redis.pipeline()
            pipe.incr(key, amount)
            if ttl:
                pipe.expire(key, ttl)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return 0

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            values = await self.redis.mget(keys)
            result = {}

            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        try:
                            result[key] = pickle.loads(value.encode())
                        except (pickle.PickleError, AttributeError):
                            result[key] = value

            return result

        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            return {}

    async def set_many(
            self,
            mapping: Dict[str, Any],
            ttl: Optional[int] = None,
            serialize_method: str = "json"
    ) -> bool:
        """Set multiple values in cache."""
        try:
            ttl = ttl or self.default_ttl
            pipe = self.redis.pipeline()

            for key, value in mapping.items():
                # Serialize value
                if serialize_method == "json":
                    try:
                        serialized_value = json.dumps(value)
                    except (TypeError, ValueError):
                        serialized_value = pickle.dumps(value)
                else:
                    serialized_value = pickle.dumps(value)

                pipe.setex(key, ttl, serialized_value)

            await pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache delete_pattern error for pattern {pattern}: {e}")
            return 0

    async def clear_cache(self) -> bool:
        """Clear all cache (use with caution)."""
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.redis.info()
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}


class SessionManager:
    """Redis-based session management."""

    def __init__(self, redis_client: Redis, prefix: str = "session:"):
        self.redis = redis_client
        self.prefix = prefix
        self.default_ttl = 3600  # 1 hour

    def _make_key(self, session_id: str) -> str:
        """Generate session key."""
        return f"{self.prefix}{session_id}"

    async def create_session(
            self,
            session_id: str,
            data: Dict[str, Any],
            ttl: Optional[int] = None
    ) -> bool:
        """Create new session."""
        try:
            key = self._make_key(session_id)
            ttl = ttl or self.default_ttl
            serialized_data = json.dumps(data)
            await self.redis.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Session create error: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        try:
            key = self._make_key(session_id)
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Session get error: {e}")
            return None

    async def update_session(
            self,
            session_id: str,
            data: Dict[str, Any],
            ttl: Optional[int] = None
    ) -> bool:
        """Update session data."""
        try:
            key = self._make_key(session_id)

            # Get current TTL if not provided
            if ttl is None:
                ttl = await self.redis.ttl(key)
                if ttl <= 0:
                    ttl = self.default_ttl

            serialized_data = json.dumps(data)
            await self.redis.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Session update error: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        try:
            key = self._make_key(session_id)
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Session delete error: {e}")
            return False

    async def extend_session(self, session_id: str, ttl: int) -> bool:
        """Extend session TTL."""
        try:
            key = self._make_key(session_id)
            result = await self.redis.expire(key, ttl)
            return result
        except Exception as e:
            logger.error(f"Session extend error: {e}")
            return False


# Global instances
redis_manager: Optional[RedisManager] = None
cache_manager: Optional[CacheManager] = None
session_manager: Optional[SessionManager] = None


async def init_redis() -> None:
    """Initialize Redis connections."""
    global redis_manager, cache_manager, session_manager, redis_client

    try:
        # Initialize Redis manager
        redis_manager = RedisManager(settings.redis_url)
        await redis_manager.connect()

        # Set global client
        redis_client = redis_manager.client

        # Initialize cache and session managers
        cache_manager = CacheManager(redis_client)
        session_manager = SessionManager(redis_client)

        logger.info("Redis initialization completed")

    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        raise


async def close_redis() -> None:
    """Close Redis connections."""
    global redis_manager

    if redis_manager:
        await redis_manager.disconnect()


@asynccontextmanager
async def get_redis():
    """Get Redis client context manager."""
    if not redis_client:
        raise RuntimeError("Redis not initialized")
    yield redis_client


# Utility functions for common cache operations
async def cache_key_builder(*args, prefix: str = "", separator: str = ":") -> str:
    """Build cache key from arguments."""
    parts = [str(arg) for arg in args if arg]
    if prefix:
        parts.insert(0, prefix)
    return separator.join(parts)


def cache_decorator(ttl: int = None, key_prefix: str = ""):
    """Decorator for caching function results."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not cache_manager:
                return await func(*args, **kwargs)

            # Build cache key
            key_parts = [key_prefix, func.__name__] + list(map(str, args))
            for k, v in sorted(kwargs.items()):
                key_parts.extend([k, str(v)])
            cache_key = await cache_key_builder(*key_parts)

            # Try to get from cache
            result = await cache_manager.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator