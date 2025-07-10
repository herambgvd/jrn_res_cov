"""
Custom exception classes for the AI Resume Platform.
"""

from typing import Any, Dict, List, Optional
from fastapi import HTTPException


class AppException(Exception):
    """Base application exception."""

    def __init__(
            self,
            message: str,
            status_code: int = 500,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(AppException):
    """Exception for validation errors."""

    def __init__(
            self,
            message: str = "Validation failed",
            field_errors: Optional[Dict[str, List[str]]] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        self.field_errors = field_errors or {}
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationException(AppException):
    """Exception for authentication errors."""

    def __init__(
            self,
            message: str = "Authentication failed",
            details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationException(AppException):
    """Exception for authorization errors."""

    def __init__(
            self,
            message: str = "Access forbidden",
            required_permissions: Optional[List[str]] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if required_permissions:
            details["required_permissions"] = required_permissions

        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )


class ResourceNotFoundException(AppException):
    """Exception for resource not found errors."""

    def __init__(
            self,
            resource_type: str,
            resource_id: Optional[str] = None,
            message: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        if not message:
            if resource_id:
                message = f"{resource_type} with ID {resource_id} not found"
            else:
                message = f"{resource_type} not found"

        details = details or {}
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })

        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details=details
        )


class ConflictException(AppException):
    """Exception for resource conflict errors."""

    def __init__(
            self,
            message: str = "Resource conflict",
            conflicting_field: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if conflicting_field:
            details["conflicting_field"] = conflicting_field

        super().__init__(
            message=message,
            status_code=409,
            error_code="RESOURCE_CONFLICT",
            details=details
        )


class RateLimitException(AppException):
    """Exception for rate limiting errors."""

    def __init__(
            self,
            message: str = "Rate limit exceeded",
            retry_after: Optional[int] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class ServiceUnavailableException(AppException):
    """Exception for service unavailable errors."""

    def __init__(
            self,
            service_name: str,
            message: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        if not message:
            message = f"{service_name} service is currently unavailable"

        details = details or {}
        details["service_name"] = service_name

        super().__init__(
            message=message,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            details=details
        )


class ExternalServiceException(AppException):
    """Exception for external service errors."""

    def __init__(
            self,
            service_name: str,
            error_message: str,
            status_code: int = 502,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details.update({
            "service_name": service_name,
            "external_error": error_message
        })

        super().__init__(
            message=f"External service error: {service_name}",
            status_code=status_code,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details
        )


class FileProcessingException(AppException):
    """Exception for file processing errors."""

    def __init__(
            self,
            message: str = "File processing failed",
            file_name: Optional[str] = None,
            file_type: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if file_name:
            details["file_name"] = file_name
        if file_type:
            details["file_type"] = file_type

        super().__init__(
            message=message,
            status_code=422,
            error_code="FILE_PROCESSING_ERROR",
            details=details
        )


class AIServiceException(AppException):
    """Exception for AI service errors."""

    def __init__(
            self,
            message: str = "AI service error",
            model_name: Optional[str] = None,
            operation: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            status_code=503,
            error_code="AI_SERVICE_ERROR",
            details=details
        )


class AnalysisException(AppException):
    """Exception for analysis processing errors."""

    def __init__(
            self,
            message: str = "Analysis failed",
            analysis_type: Optional[str] = None,
            target_id: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if analysis_type:
            details["analysis_type"] = analysis_type
        if target_id:
            details["target_id"] = target_id

        super().__init__(
            message=message,
            status_code=422,
            error_code="ANALYSIS_ERROR",
            details=details
        )


class TemplateException(AppException):
    """Exception for template processing errors."""

    def __init__(
            self,
            message: str = "Template processing failed",
            template_id: Optional[str] = None,
            template_type: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if template_id:
            details["template_id"] = template_id
        if template_type:
            details["template_type"] = template_type

        super().__init__(
            message=message,
            status_code=422,
            error_code="TEMPLATE_ERROR",
            details=details
        )


class DatabaseException(AppException):
    """Exception for database errors."""

    def __init__(
            self,
            message: str = "Database operation failed",
            operation: Optional[str] = None,
            table: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if operation:
            details["operation"] = operation
        if table:
            details["table"] = table

        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details=details
        )


class CacheException(AppException):
    """Exception for cache errors."""

    def __init__(
            self,
            message: str = "Cache operation failed",
            cache_key: Optional[str] = None,
            operation: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if cache_key:
            details["cache_key"] = cache_key
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            status_code=500,
            error_code="CACHE_ERROR",
            details=details
        )


class ConfigurationException(AppException):
    """Exception for configuration errors."""

    def __init__(
            self,
            message: str = "Configuration error",
            config_key: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if config_key:
            details["config_key"] = config_key

        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class BusinessLogicException(AppException):
    """Exception for business logic violations."""

    def __init__(
            self,
            message: str,
            rule: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if rule:
            details["violated_rule"] = rule

        super().__init__(
            message=message,
            status_code=400,
            error_code="BUSINESS_LOGIC_ERROR",
            details=details
        )


# Custom exception for quota/limit errors
class QuotaExceededException(AppException):
    """Exception for quota exceeded errors."""

    def __init__(
            self,
            resource_type: str,
            current_usage: int,
            limit: int,
            message: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        if not message:
            message = f"{resource_type} quota exceeded: {current_usage}/{limit}"

        details = details or {}
        details.update({
            "resource_type": resource_type,
            "current_usage": current_usage,
            "limit": limit
        })

        super().__init__(
            message=message,
            status_code=402,  # Payment Required
            error_code="QUOTA_EXCEEDED",
            details=details
        )


# Utility functions for exception handling
def handle_database_error(e: Exception, operation: str = "unknown", table: str = "unknown") -> AppException:
    """Convert database errors to AppException."""
    error_message = str(e)

    if "duplicate" in error_message.lower() or "unique" in error_message.lower():
        return ConflictException(
            message="Resource already exists",
            details={"operation": operation, "table": table, "db_error": error_message}
        )
    elif "foreign key" in error_message.lower():
        return ValidationException(
            message="Invalid reference to related resource",
            details={"operation": operation, "table": table, "db_error": error_message}
        )
    elif "not found" in error_message.lower():
        return ResourceNotFoundException(
            resource_type=table,
            details={"operation": operation, "db_error": error_message}
        )
    else:
        return DatabaseException(
            message="Database operation failed",
            operation=operation,
            table=table,
            details={"db_error": error_message}
        )


def handle_ai_service_error(e: Exception, model_name: str = "unknown", operation: str = "unknown") -> AppException:
    """Convert AI service errors to AppException."""
    error_message = str(e)

    if "quota" in error_message.lower() or "rate limit" in error_message.lower():
        return RateLimitException(
            message="AI service rate limit exceeded",
            details={"model_name": model_name, "operation": operation, "ai_error": error_message}
        )
    elif "timeout" in error_message.lower():
        return ServiceUnavailableException(
            service_name="AI Service",
            message="AI service timeout",
            details={"model_name": model_name, "operation": operation, "ai_error": error_message}
        )
    elif "unauthorized" in error_message.lower() or "authentication" in error_message.lower():
        return ConfigurationException(
            message="AI service authentication failed",
            details={"model_name": model_name, "operation": operation, "ai_error": error_message}
        )
    else:
        return AIServiceException(
            message="AI service error",
            model_name=model_name,
            operation=operation,
            details={"ai_error": error_message}
        )


def handle_file_processing_error(e: Exception, file_name: str = "unknown", file_type: str = "unknown") -> AppException:
    """Convert file processing errors to AppException."""
    error_message = str(e)

    if "size" in error_message.lower():
        return ValidationException(
            message="File size exceeds limit",
            details={"file_name": file_name, "file_type": file_type, "file_error": error_message}
        )
    elif "format" in error_message.lower() or "type" in error_message.lower():
        return ValidationException(
            message="Unsupported file format",
            details={"file_name": file_name, "file_type": file_type, "file_error": error_message}
        )
    elif "corrupt" in error_message.lower() or "invalid" in error_message.lower():
        return FileProcessingException(
            message="File is corrupted or invalid",
            file_name=file_name,
            file_type=file_type,
            details={"file_error": error_message}
        )
    else:
        return FileProcessingException(
            message="File processing failed",
            file_name=file_name,
            file_type=file_type,
            details={"file_error": error_message}
        )


# Exception hierarchy for easy catching
class ClientError(AppException):
    """Base class for 4xx client errors."""
    pass


class ServerError(AppException):
    """Base class for 5xx server errors."""
    pass


# Update existing exceptions to inherit from appropriate base classes
class ValidationException(ClientError):
    """Exception for validation errors."""
    pass


class AuthenticationException(ClientError):
    """Exception for authentication errors."""
    pass


class AuthorizationException(ClientError):
    """Exception for authorization errors."""
    pass


class ResourceNotFoundException(ClientError):
    """Exception for resource not found errors."""
    pass


class ConflictException(ClientError):
    """Exception for resource conflict errors."""
    pass


class RateLimitException(ClientError):
    """Exception for rate limiting errors."""
    pass


class ServiceUnavailableException(ServerError):
    """Exception for service unavailable errors."""
    pass


class ExternalServiceException(ServerError):
    """Exception for external service errors."""
    pass


class DatabaseException(ServerError):
    """Exception for database errors."""
    pass


class CacheException(ServerError):
    """Exception for cache errors."""
    pass


class ConfigurationException(ServerError):
    """Exception for configuration errors."""
    pass