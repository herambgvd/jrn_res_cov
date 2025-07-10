"""
Common Pydantic schemas used across the application.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar

from pydantic import BaseModel, Field, ConfigDict

T = TypeVar('T')


# Response wrapper schemas
class ResponseWrapper(BaseModel, Generic[T]):
    """Generic response wrapper."""
    success: bool = True
    message: str = "Success"
    data: Optional[T] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: List[T]
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    size: int = Field(..., ge=1, le=100, description="Number of items per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationErrorResponse(BaseModel):
    """Validation error response schema."""
    success: bool = False
    message: str = "Validation error"
    errors: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Page size")

    def get_offset(self) -> int:
        """Get the offset for database queries."""
        return (self.page - 1) * self.size


class SortParams(BaseModel):
    """Sorting parameters."""
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: str = Field("desc", regex="^(asc|desc)$", description="Sort order")


class FilterParams(BaseModel):
    """Base filtering parameters."""
    created_after: Optional[datetime] = Field(None, description="Filter items created after this date")
    created_before: Optional[datetime] = Field(None, description="Filter items created before this date")
    updated_after: Optional[datetime] = Field(None, description="Filter items updated after this date")
    updated_before: Optional[datetime] = Field(None, description="Filter items updated before this date")


# Search schemas
class SearchParams(BaseModel):
    """Search parameters."""
    query: Optional[str] = Field(None, min_length=1, description="Search query")
    fields: Optional[List[str]] = Field(None, description="Fields to search in")
    fuzzy: bool = Field(False, description="Enable fuzzy matching")
    highlight: bool = Field(False, description="Highlight search terms in results")


class SearchResponse(BaseModel, Generic[T]):
    """Search response with highlighting and facets."""
    items: List[T]
    total: int
    query: str
    took: float = Field(..., description="Search time in milliseconds")
    highlights: Optional[Dict[str, List[str]]] = None
    facets: Optional[Dict[str, Dict[str, int]]] = None
    suggestions: Optional[List[str]] = None


# File upload schemas
class FileUploadResponse(BaseModel):
    """File upload response schema."""
    file_id: uuid.UUID
    filename: str
    file_size: int
    file_type: str
    file_path: str
    upload_url: Optional[str] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class FileMetadata(BaseModel):
    """File metadata schema."""
    filename: str
    file_size: int
    file_type: str
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Health check schemas
class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Health status (healthy, unhealthy, degraded)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    database: Dict[str, Any] = Field(..., description="Database health status")
    redis: Dict[str, Any] = Field(..., description="Redis health status")
    celery: Dict[str, Any] = Field(..., description="Celery health status")
    external_services: Optional[Dict[str, Any]] = None
    uptime: float = Field(..., description="Uptime in seconds")


class DatabaseHealth(BaseModel):
    """Database health schema."""
    status: str
    response_time: float = Field(..., description="Response time in milliseconds")
    connection_pool: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None


class RedisHealth(BaseModel):
    """Redis health schema."""
    status: str
    response_time: float = Field(..., description="Response time in milliseconds")
    memory_usage: Optional[str] = None
    connected_clients: Optional[int] = None
    last_error: Optional[str] = None


class CeleryHealth(BaseModel):
    """Celery health schema."""
    status: str
    workers: int = Field(..., description="Number of active workers")
    queues: Dict[str, int] = Field(..., description="Queue lengths")
    active_tasks: int = Field(..., description="Number of active tasks")
    last_error: Optional[str] = None


# Task status schemas
class TaskStatus(BaseModel):
    """Background task status schema."""
    task_id: uuid.UUID
    status: str = Field(..., description="Task status (pending, running, success, failure)")
    progress: int = Field(0, ge=0, le=100, description="Progress percentage")
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


class TaskResult(BaseModel):
    """Background task result schema."""
    task_id: uuid.UUID
    status: str
    result: Any
    traceback: Optional[str] = None
    date_done: Optional[datetime] = None


# Notification schemas
class NotificationBase(BaseModel):
    """Base notification schema."""
    title: str = Field(..., max_length=200)
    message: str = Field(..., max_length=1000)
    type: str = Field(..., description="Notification type (info, success, warning, error)")
    priority: str = Field("normal", description="Priority (low, normal, high, urgent)")


class NotificationCreate(NotificationBase):
    """Schema for creating notifications."""
    user_id: uuid.UUID = Field(..., description="Target user ID")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional notification data")
    scheduled_at: Optional[datetime] = Field(None, description="When to send the notification")
    expires_at: Optional[datetime] = Field(None, description="When the notification expires")


class NotificationResponse(NotificationBase):
    """Schema for notification response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    data: Optional[Dict[str, Any]] = None
    is_read: bool = False
    is_sent: bool = False
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    read_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None


# Analytics schemas
class AnalyticsEvent(BaseModel):
    """Analytics event schema."""
    event_name: str = Field(..., max_length=100)
    user_id: Optional[uuid.UUID] = None
    session_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalyticsMetrics(BaseModel):
    """Analytics metrics schema."""
    metric_name: str
    value: float
    unit: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# User context schemas (for integration with auth service)
class UserContext(BaseModel):
    """User context from auth service."""
    user_id: uuid.UUID
    email: str
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_verified: bool = True


class UserProfile(BaseModel):
    """User profile schema."""
    id: uuid.UUID
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    phone: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True
    is_verified: bool = True
    created_at: datetime
    updated_at: datetime


# Audit schemas
class AuditLog(BaseModel):
    """Audit log schema."""
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    action: str = Field(..., max_length=100)
    resource_type: str = Field(..., max_length=50)
    resource_id: Optional[uuid.UUID] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Configuration schemas
class AppSettings(BaseModel):
    """Application settings schema."""
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    rate_limits: Dict[str, int] = Field(default_factory=dict)
    ai_settings: Dict[str, Any] = Field(default_factory=dict)
    email_settings: Dict[str, Any] = Field(default_factory=dict)
    storage_settings: Dict[str, Any] = Field(default_factory=dict)


# Batch operation schemas
class BatchOperation(BaseModel):
    """Batch operation schema."""
    operation_type: str = Field(..., description="Type of batch operation")
    items: List[Dict[str, Any]] = Field(..., min_items=1, description="Items to process")
    options: Optional[Dict[str, Any]] = Field(None, description="Operation options")


class BatchOperationResult(BaseModel):
    """Batch operation result schema."""
    operation_id: uuid.UUID
    operation_type: str
    total_items: int
    successful_items: int
    failed_items: int
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = Field(..., description="Operation status")


# Export/Import schemas
class ExportRequest(BaseModel):
    """Export request schema."""
    export_type: str = Field(..., description="Type of export")
    format: str = Field(..., description="Export format")
    filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
    options: Optional[Dict[str, Any]] = Field(None, description="Export options")


class ExportResponse(BaseModel):
    """Export response schema."""
    export_id: uuid.UUID
    download_url: str
    file_name: str
    file_size: int
    format: str
    expires_at: datetime
    created_at: datetime


class ImportRequest(BaseModel):
    """Import request schema."""
    import_type: str = Field(..., description="Type of import")
    file_url: str = Field(..., description="URL of file to import")
    options: Optional[Dict[str, Any]] = Field(None, description="Import options")
    validation_options: Optional[Dict[str, Any]] = Field(None, description="Validation options")


class ImportResponse(BaseModel):
    """Import response schema."""
    import_id: uuid.UUID
    status: str = Field(..., description="Import status")
    total_records: int
    imported_records: int
    failed_records: int
    validation_errors: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    completed_at: Optional[datetime] = None


# Rate limiting schemas
class RateLimitInfo(BaseModel):
    """Rate limit information schema."""
    limit: int = Field(..., description="Rate limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="When the limit resets")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")


# Generic utility schemas
class IDResponse(BaseModel):
    """Simple ID response schema."""
    id: uuid.UUID


class MessageResponse(BaseModel):
    """Simple message response schema."""
    message: str


class CountResponse(BaseModel):
    """Simple count response schema."""
    count: int


class StatusResponse(BaseModel):
    """Simple status response schema."""
    status: str
    message: Optional[str] = None


class BooleanResponse(BaseModel):
    """Simple boolean response schema."""
    success: bool
    message: Optional[str] = None


# API versioning schemas
class APIVersion(BaseModel):
    """API version information schema."""
    version: str = Field(..., description="API version")
    release_date: datetime = Field(..., description="Release date")
    deprecated: bool = Field(False, description="Whether this version is deprecated")
    deprecation_date: Optional[datetime] = Field(None, description="When this version will be deprecated")
    end_of_life: Optional[datetime] = Field(None, description="End of life date")
    changelog_url: Optional[str] = Field(None, description="URL to changelog")


# Feature flag schemas
class FeatureFlag(BaseModel):
    """Feature flag schema."""
    name: str = Field(..., max_length=100)
    enabled: bool = Field(..., description="Whether the feature is enabled")
    description: Optional[str] = Field(None, description="Feature description")
    rollout_percentage: int = Field(100, ge=0, le=100, description="Rollout percentage")
    target_users: Optional[List[uuid.UUID]] = Field(None, description="Specific target users")
    target_roles: Optional[List[str]] = Field(None, description="Target roles")
    start_date: Optional[datetime] = Field(None, description="When to start the feature")
    end_date: Optional[datetime] = Field(None, description="When to end the feature")


# Webhook schemas
class WebhookEvent(BaseModel):
    """Webhook event schema."""
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: str = Field(..., max_length=100)
    resource_type: str = Field(..., max_length=50)
    resource_id: uuid.UUID
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0", description="Event schema version")


class WebhookDelivery(BaseModel):
    """Webhook delivery schema."""
    delivery_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    webhook_url: str
    event: WebhookEvent
    status: str = Field(..., description="Delivery status")
    attempts: int = Field(0, description="Number of delivery attempts")
    last_attempt_at: Optional[datetime] = None
    next_attempt_at: Optional[datetime] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None