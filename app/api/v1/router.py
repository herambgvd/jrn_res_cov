"""
Main API router that includes all endpoint modules.
"""

from fastapi import APIRouter

from app.api.v1 import resume, cover_letter, analysis, templates
from app.config import settings

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    resume.router,
    prefix="/resumes",
    tags=["Resumes"],
    responses={404: {"description": "Resume not found"}},
)

api_router.include_router(
    cover_letter.router,
    prefix="/cover-letters",
    tags=["Cover Letters"],
    responses={404: {"description": "Cover letter not found"}},
)

api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=["Analysis"],
    responses={404: {"description": "Analysis not found"}},
)

# Include templates router if template customization is enabled
if settings.enable_template_customization:
    api_router.include_router(
        templates.router,
        prefix="/templates",
        tags=["Templates"],
        responses={404: {"description": "Template not found"}},
    )

# Add any additional routers here as needed
# For example: jobs router, users router (if managing users locally), etc.