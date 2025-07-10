"""
API dependencies for authentication, authorization, and service injection.
"""

import uuid
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings
from app.schemas.common import UserContext
from app.services.resume_service import ResumeService
from app.services.cover_letter_service import CoverLetterService
from app.services.analysis_service import AnalysisService
from app.services.ai_service import AIService
from app.services.pdf_service import PDFService
from app.utils.exceptions import AppException

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserContext:
    """
    Get current user from JWT token via AuthKit service.

    This function integrates with the existing AuthKit service to validate
    JWT tokens and retrieve user information.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # Call AuthKit service to validate token and get user info
        async with httpx.AsyncClient(timeout=settings.user_service_timeout) as client:
            response = await client.get(
                f"{settings.user_service_url}/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Authentication service unavailable"
                )

            user_data = response.json()

            return UserContext(
                user_id=uuid.UUID(user_data["id"]),
                email=user_data["email"],
                roles=user_data.get("roles", []),
                permissions=user_data.get("permissions", []),
                is_active=user_data.get("is_active", True),
                is_verified=user_data.get("is_verified", True)
            )

    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error"
        )


def require_permissions(*required_permissions: str):
    """
    Dependency to require specific permissions.

    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(
            user: UserContext = Depends(require_permissions("admin.read"))
        ):
            ...
    """

    def permission_checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )

        if not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is not verified"
            )

        # Check if user has any of the required permissions
        user_permissions = set(user.permissions)
        required_permissions_set = set(required_permissions)

        if not required_permissions_set.intersection(user_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        return user

    return permission_checker


def require_roles(*required_roles: str):
    """
    Dependency to require specific roles.

    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(
            user: UserContext = Depends(require_roles("admin", "superuser"))
        ):
            ...
    """

    def role_checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )

        # Check if user has any of the required roles
        user_roles = set(user.roles)
        required_roles_set = set(required_roles)

        if not required_roles_set.intersection(user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role privileges"
            )

        return user

    return role_checker


def require_admin():
    """Convenience dependency for admin-only endpoints."""
    return require_roles("admin", "superuser")


def require_verified_user():
    """Dependency to require verified user account."""

    def verification_checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account verification required"
            )
        return user

    return verification_checker


# Service dependencies
def get_ai_service() -> AIService:
    """Get AI service instance."""
    return AIService()


def get_pdf_service() -> PDFService:
    """Get PDF service instance."""
    return PDFService()


def get_resume_service(
        ai_service: AIService = Depends(get_ai_service),
        pdf_service: PDFService = Depends(get_pdf_service)
) -> ResumeService:
    """Get resume service instance."""
    return ResumeService(ai_service=ai_service, pdf_service=pdf_service)


def get_cover_letter_service(
        ai_service: AIService = Depends(get_ai_service),
        pdf_service: PDFService = Depends(get_pdf_service)
) -> CoverLetterService:
    """Get cover letter service instance."""
    return CoverLetterService(ai_service=ai_service, pdf_service=pdf_service)


def get_analysis_service(
        ai_service: AIService = Depends(get_ai_service)
) -> AnalysisService:
    """Get analysis service instance."""
    return AnalysisService(ai_service=ai_service)


# Rate limiting dependency
async def rate_limit_check(
        user: UserContext = Depends(get_current_user)
) -> UserContext:
    """
    Check rate limits for the current user.

    This is a placeholder implementation. In production, you would
    integrate with Redis to track request rates per user.
    """
    # TODO: Implement actual rate limiting with Redis
    # For now, just return the user
    return user


# Feature flag dependencies
def require_feature(feature_name: str):
    """
    Dependency to require a specific feature to be enabled.

    Usage:
        @router.get("/ai-analysis")
        async def ai_analysis(
            _: None = Depends(require_feature("enable_ai_analysis"))
        ):
            ...
    """

    def feature_checker():
        feature_enabled = getattr(settings, feature_name, False)
        if not feature_enabled:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature not available"
            )
        return None

    return feature_checker


def require_ai_analysis():
    """Require AI analysis feature to be enabled."""
    return require_feature("enable_ai_analysis")


def require_cover_letter():
    """Require cover letter feature to be enabled."""
    return require_feature("enable_cover_letter")


def require_bulk_operations():
    """Require bulk operations feature to be enabled."""
    return require_feature("enable_bulk_operations")


# Pagination dependencies
class PaginationParams:
    """Pagination parameters."""

    def __init__(
            self,
            page: int = 1,
            size: int = 20,
            max_size: int = 100
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page must be >= 1"
            )

        if size < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Size must be >= 1"
            )

        if size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Size must be <= {max_size}"
            )

        self.page = page
        self.size = size
        self.offset = (page - 1) * size


def get_pagination_params(
        page: int = 1,
        size: int = 20
) -> PaginationParams:
    """Get validated pagination parameters."""
    return PaginationParams(page=page, size=size)


# File upload dependencies
def validate_file_upload(
        content_type: str,
        file_size: int,
        allowed_types: list = None,
        max_size: int = None
):
    """
    Validate file upload parameters.

    Args:
        content_type: MIME type of the uploaded file
        file_size: Size of the uploaded file in bytes
        allowed_types: List of allowed MIME types
        max_size: Maximum file size in bytes
    """
    if allowed_types is None:
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain"
        ]

    if max_size is None:
        max_size = settings.max_file_size

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
        )

    if file_size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size too large. Maximum size: {max_size_mb:.1f}MB"
        )


# Request context dependency
class RequestContext:
    """Request context information."""

    def __init__(self, user: UserContext, request_id: str = None):
        self.user = user
        self.request_id = request_id or str(uuid.uuid4())


def get_request_context(
        user: UserContext = Depends(get_current_user)
) -> RequestContext:
    """Get request context with user and request ID."""
    return RequestContext(user=user)


# Background task dependencies
def get_background_task_context(
        user: UserContext = Depends(get_current_user)
) -> dict:
    """Get context for background tasks."""
    return {
        "user_id": str(user.user_id),
        "user_email": user.email,
        "user_roles": user.roles,
        "user_permissions": user.permissions
    }


# Error handling for service dependencies
def handle_service_errors(func):
    """Decorator to handle common service errors."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AppException as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal service error"
            )

    return wrapper