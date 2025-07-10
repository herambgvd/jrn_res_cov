"""
Celery configuration and app initialization.
"""

import logging
from typing import Any, Dict

from celery import Celery
from celery.signals import worker_ready, worker_shutting_down
from kombu import Queue

from app.config import settings

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "ai_resume_platform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.resume_tasks",
        "app.tasks.analysis_tasks",
        "app.tasks.email_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer=settings.celery_task_serializer,
    accept_content=settings.celery_accept_content,
    result_serializer=settings.celery_result_serializer,
    timezone=settings.celery_timezone,
    enable_utc=True,

    # Task execution
    task_track_started=True,
    task_time_limit=settings.task_time_limit,
    task_soft_time_limit=settings.task_soft_time_limit,

    # Task routing and queues
    task_routes={
        "app.tasks.resume_tasks.*": {"queue": "resume"},
        "app.tasks.analysis_tasks.*": {"queue": "analysis"},
        "app.tasks.email_tasks.*": {"queue": "email"},
    },

    # Queue definitions
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("resume", routing_key="resume"),
        Queue("analysis", routing_key="analysis"),
        Queue("email", routing_key="email"),
        Queue("priority", routing_key="priority"),
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",

    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,

    # Result backend settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },

    # Retry configuration
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=settings.task_retry_delay,
    task_max_retries=settings.task_max_retries,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,

    # Beat schedule (for periodic tasks)
    beat_schedule={
        "cleanup-expired-files": {
            "task": "app.tasks.resume_tasks.cleanup_expired_files",
            "schedule": 3600.0,  # Run every hour
        },
        "cleanup-temp-files": {
            "task": "app.tasks.resume_tasks.cleanup_temp_files",
            "schedule": 1800.0,  # Run every 30 minutes
        },
        "update-analysis-cache": {
            "task": "app.tasks.analysis_tasks.update_analysis_cache",
            "schedule": 7200.0,  # Run every 2 hours
        },
    },
    beat_scheduler="celery.beat:PersistentScheduler",
)


class CeleryConfig:
    """Celery configuration class."""

    # Task discovery
    imports = [
        "app.tasks.resume_tasks",
        "app.tasks.analysis_tasks",
        "app.tasks.email_tasks",
    ]

    # Task execution
    task_always_eager = settings.is_testing
    task_eager_propagates = settings.is_testing

    # Monitoring and logging
    worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
    worker_task_log_format = "[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s"

    # Security
    worker_enable_remote_control = not settings.is_production

    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get Celery configuration as dictionary."""
        return {
            # Broker and backend
            "broker_url": settings.celery_broker_url,
            "result_backend": settings.celery_result_backend,

            # Serialization
            "task_serializer": settings.celery_task_serializer,
            "result_serializer": settings.celery_result_serializer,
            "accept_content": settings.celery_accept_content,

            # Timezone
            "timezone": settings.celery_timezone,
            "enable_utc": True,

            # Task settings
            "task_track_started": True,
            "task_acks_late": True,
            "task_reject_on_worker_lost": True,
            "task_time_limit": settings.task_time_limit,
            "task_soft_time_limit": settings.task_soft_time_limit,

            # Worker settings
            "worker_prefetch_multiplier": 1,
            "worker_max_tasks_per_child": 1000,
            "worker_send_task_events": True,

            # Result settings
            "result_expires": 3600,
            "task_send_sent_event": True,

            # Retry settings
            "task_default_retry_delay": settings.task_retry_delay,
            "task_max_retries": settings.task_max_retries,
        }


def create_celery_app() -> Celery:
    """Create and configure Celery app."""
    app = Celery("ai_resume_platform")

    # Update configuration
    config = CeleryConfig.get_config()
    app.conf.update(config)

    # Set up task discovery
    app.autodiscover_tasks([
        "app.tasks.resume_tasks",
        "app.tasks.analysis_tasks",
        "app.tasks.email_tasks",
    ])

    return app


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal."""
    logger.info(f"Celery worker ready: {sender}")


@worker_shutting_down.connect
def worker_shutting_down_handler(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    logger.info(f"Celery worker shutting down: {sender}")


# Task decorator with default settings
def task(*args, **kwargs):
    """Task decorator with default settings."""
    kwargs.setdefault("bind", True)
    kwargs.setdefault("autoretry_for", (Exception,))
    kwargs.setdefault("retry_kwargs", {"max_retries": settings.task_max_retries})
    kwargs.setdefault("retry_backoff", True)
    kwargs.setdefault("retry_jitter", True)

    return celery_app.task(*args, **kwargs)


# Priority task decorator
def priority_task(*args, **kwargs):
    """Priority task decorator."""
    kwargs.setdefault("queue", "priority")
    kwargs.setdefault("priority", 9)
    return task(*args, **kwargs)


# Long running task decorator
def long_task(*args, **kwargs):
    """Long running task decorator."""
    kwargs.setdefault("time_limit", 1800)  # 30 minutes
    kwargs.setdefault("soft_time_limit", 1700)  # 28 minutes
    return task(*args, **kwargs)


class TaskStatus:
    """Task status constants."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskManager:
    """Task management utilities."""

    @staticmethod
    def get_task_info(task_id: str) -> Dict[str, Any]:
        """Get task information by ID."""
        task_result = celery_app.AsyncResult(task_id)
        return {
            "id": task_id,
            "status": task_result.status,
            "result": task_result.result,
            "traceback": task_result.traceback,
            "date_done": task_result.date_done,
        }

    @staticmethod
    def revoke_task(task_id: str, terminate: bool = False) -> bool:
        """Revoke a task."""
        try:
            celery_app.control.revoke(task_id, terminate=terminate)
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task {task_id}: {e}")
            return False

    @staticmethod
    def get_active_tasks() -> list:
        """Get list of active tasks."""
        try:
            inspect = celery_app.control.inspect()
            active = inspect.active()
            return active or []
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return []

    @staticmethod
    def get_worker_stats() -> Dict[str, Any]:
        """Get worker statistics."""
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            return stats or {}
        except Exception as e:
            logger.error(f"Failed to get worker stats: {e}")
            return {}

    @staticmethod
    def purge_queue(queue_name: str) -> int:
        """Purge messages from queue."""
        try:
            result = celery_app.control.purge()
            return result
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return 0


# Global task manager instance
task_manager = TaskManager()


# Health check for Celery
async def celery_health_check() -> Dict[str, Any]:
    """Check Celery health."""
    try:
        # Check if workers are available
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if not stats:
            return {
                "status": "unhealthy",
                "message": "No workers available",
                "workers": 0
            }

        worker_count = len(stats)
        return {
            "status": "healthy",
            "message": f"{worker_count} worker(s) available",
            "workers": worker_count,
            "worker_stats": stats
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": str(e),
            "workers": 0
        }


# Export the configured Celery app
__all__ = ["celery_app", "task", "priority_task", "long_task", "task_manager"]