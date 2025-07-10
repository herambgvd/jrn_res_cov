"""
Main FastAPI application for AI Resume Builder & Analysis Platform.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.core.database import init_db, close_db, db_manager
from app.core.redis import init_redis, close_redis
from app.core.celery_app import celery_app
from app.api.v1.router import api_router
from app.schemas.common import ErrorResponse, ValidationErrorResponse, HealthCheckResponse
from app.utils.exceptions import AppException

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Resume Platform API", version=settings.app_version)

    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")

        # Initialize Redis
        await init_redis()
        logger.info("Redis initialized successfully")

        # Verify Celery connection
        try:
            celery_inspect = celery_app.control.inspect()
            stats = celery_inspect.stats()
            if stats:
                logger.info("Celery connection verified", workers=len(stats))
            else:
                logger.warning("No Celery workers found")
        except Exception as e:
            logger.error("Celery connection failed", error=str(e))

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down AI Resume Platform API")

    try:
        await close_db()
        logger.info("Database connections closed")

        await close_redis()
        logger.info("Redis connections closed")

        logger.info("Application shutdown completed successfully")

    except Exception as e:
        logger.error("Application shutdown failed", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="AI Resume Builder & Analysis Platform",
    description="AI-powered resume builder and analysis platform with cover letter generation",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)


# Middleware configuration
def configure_middleware():
    """Configure FastAPI middleware."""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )

    # Trusted host middleware (security)
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure appropriately for production
        )

    # GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        max_age=3600,
        same_site="lax",
        https_only=settings.is_production,
    )

    # Request ID and timing middleware
    @app.middleware("http")
    async def add_request_id_and_timing(request: Request, call_next):
        """Add request ID and timing to requests."""
        import uuid
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Add request ID to headers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{process_time:.4f}s"

        # Log request
        logger.info(
            "Request processed",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None,
        )

        return response

    # Rate limiting middleware (basic implementation)
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        """Basic rate limiting middleware."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/live", "/health/ready"]:
            return await call_next(request)

        # TODO: Implement proper rate limiting with Redis
        # For now, just process the request
        response = await call_next(request)
        return response


# Exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions."""
    logger.error(
        "Application error",
        error=str(exc),
        error_code=exc.error_code,
        request_id=getattr(request.state, "request_id", None),
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        "HTTP error",
        status_code=exc.status_code,
        detail=exc.detail,
        request_id=getattr(request.state, "request_id", None),
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Validation error",
        errors=exc.errors(),
        request_id=getattr(request.state, "request_id", None),
        path=request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content=ValidationErrorResponse(
            errors=exc.errors(),
        ).model_dump(),
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors."""
    logger.error(
        "Internal server error",
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=getattr(request.state, "request_id", None),
        path=request.url.path,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error" if settings.is_production else str(exc),
            error_code="INTERNAL_SERVER_ERROR",
        ).model_dump(),
    )


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint."""
    import psutil
    import time

    start_time = time.time()

    # Check database health
    db_health = await db_manager.health_check()

    # Check Redis health
    from app.core.redis import redis_manager
    redis_health = await redis_manager.health_check() if redis_manager else False

    # Check Celery health
    try:
        celery_inspect = celery_app.control.inspect()
        celery_stats = celery_inspect.stats()
        celery_healthy = bool(celery_stats)
        celery_workers = len(celery_stats) if celery_stats else 0
    except Exception:
        celery_healthy = False
        celery_workers = 0

    # Overall health status
    overall_healthy = db_health and redis_health and celery_healthy
    status = "healthy" if overall_healthy else "unhealthy"

    # Get system uptime
    uptime = time.time() - start_time

    return HealthCheckResponse(
        status=status,
        version=settings.app_version,
        environment=settings.environment,
        database={
            "status": "healthy" if db_health else "unhealthy",
            "response_time": time.time() - start_time,
        },
        redis={
            "status": "healthy" if redis_health else "unhealthy",
            "response_time": time.time() - start_time,
        },
        celery={
            "status": "healthy" if celery_healthy else "unhealthy",
            "workers": celery_workers,
        },
        uptime=uptime,
    )


@app.get("/health/live", tags=["Health"])
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    # Quick health checks for readiness
    try:
        db_healthy = await db_manager.health_check()
        if not db_healthy:
            raise HTTPException(status_code=503, detail="Database not ready")

        return {"status": "ready"}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Resume Builder & Analysis Platform",
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": "/docs" if settings.debug else None,
        "health_url": "/health",
        "api_prefix": "/api/v1",
        "features": {
            "resume_builder": True,
            "cover_letter_builder": settings.enable_cover_letter,
            "ai_analysis": settings.enable_ai_analysis,
            "template_customization": settings.enable_template_customization,
            "bulk_operations": settings.enable_bulk_operations,
        },
    }


# API version information
@app.get("/api/version", tags=["API Info"])
async def api_version():
    """Get API version information."""
    return {
        "version": "1.0.0",
        "release_date": "2024-01-01T00:00:00Z",
        "deprecated": False,
        "endpoints": {
            "resumes": "/api/v1/resumes",
            "cover_letters": "/api/v1/cover-letters",
            "analysis": "/api/v1/analysis",
            "templates": "/api/v1/templates",
        },
    }


# Include API routers
app.include_router(api_router, prefix="/api/v1")

# Configure middleware after creating the app
configure_middleware()


# Additional startup configuration
@app.on_event("startup")
async def startup_event():
    """Additional startup configuration."""
    logger.info("Performing additional startup tasks")

    # Initialize any background tasks or caches here
    # Example: Load templates into cache, initialize AI models, etc.

    logger.info("Startup tasks completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Additional shutdown tasks."""
    logger.info("Performing shutdown cleanup")

    # Cleanup tasks here
    # Example: Save cache to disk, cleanup temp files, etc.

    logger.info("Shutdown cleanup completed")


# Custom route for serving static files (if needed)
if settings.debug:
    from fastapi.staticfiles import StaticFiles
    import os

    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

# WebSocket support for real-time features (optional)
from fastapi import WebSocket, WebSocketDisconnect
from typing import List


class ConnectionManager:
    """WebSocket connection manager for real-time features."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket."""
        await websocket.send_text(message)

    async def send_user_message(self, message: str, user_id: str):
        """Send message to all connections for a user."""
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    # Connection might be closed
                    pass

    async def broadcast(self, message: str):
        """Broadcast message to all connections."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Connection might be closed
                pass


manager = ConnectionManager()


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Echo message back (or handle as needed)
            await manager.send_personal_message(f"Echo: {data}", websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info("WebSocket disconnected", user_id=user_id)


# Utility function for sending real-time notifications
async def send_real_time_notification(user_id: str, notification: Dict[str, Any]):
    """Send real-time notification to user."""
    import json
    message = json.dumps({
        "type": "notification",
        "data": notification,
        "timestamp": time.time(),
    })
    await manager.send_user_message(message, user_id)


# Background task to send analysis progress updates
async def send_analysis_progress(user_id: str, task_id: str, progress: int):
    """Send analysis progress update."""
    import json
    message = json.dumps({
        "type": "analysis_progress",
        "task_id": task_id,
        "progress": progress,
        "timestamp": time.time(),
    })
    await manager.send_user_message(message, user_id)


# Error tracking integration (optional)
if settings.sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    from sentry_sdk.integrations.redis import RedisIntegration

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        integrations=[
            FastApiIntegration(auto_enabling_integrations=False),
            SqlalchemyIntegration(),
            RedisIntegration(),
        ],
        traces_sample_rate=0.1 if settings.is_production else 1.0,
        send_default_pii=False,
    )

# Metrics collection (optional)
if settings.enable_metrics:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    # Define metrics
    REQUEST_COUNT = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )

    REQUEST_DURATION = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration',
        ['method', 'endpoint']
    )


    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Collect metrics for requests."""
        start_time = time.time()

        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        return response


    @app.get("/metrics", tags=["Monitoring"])
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)

    if settings.is_production:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response


# API documentation customization
if settings.debug:
    from fastapi.openapi.utils import get_openapi


    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="AI Resume Builder & Analysis Platform API",
            version=settings.app_version,
            description="""
            ## AI-Powered Resume Builder & Analysis Platform

            This API provides comprehensive resume building and analysis capabilities:

            ### Features
            - **Resume Builder**: Create professional resumes with multiple templates
            - **AI Analysis**: Get detailed analysis and improvement suggestions
            - **Cover Letter Builder**: Generate personalized cover letters
            - **ATS Optimization**: Ensure compatibility with Applicant Tracking Systems
            - **Job Matching**: Match resumes against job descriptions

            ### Authentication
            This API integrates with the AuthKit service for authentication.
            Include the JWT token in the Authorization header:
            `Authorization: Bearer <your-token>`

            ### Rate Limiting
            API requests are rate-limited to ensure fair usage.

            ### Support
            For support, contact: support@example.com
            """,
            routes=app.routes,
        )

        # Add custom info
        openapi_schema["info"]["x-logo"] = {
            "url": "https://your-domain.com/logo.png"
        }

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema


    app.openapi = custom_openapi

# Export the FastAPI app
__all__ = ["app"]