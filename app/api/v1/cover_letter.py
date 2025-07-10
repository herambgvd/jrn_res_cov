"""
Cover Letter API endpoints.
"""

import uuid
# Import required modules for datetime operations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks

from app.api.deps import get_current_user, get_cover_letter_service
from app.models.cover_letter import CoverLetterStatus, CoverLetterType, CoverLetterTone
from app.schemas.common import ResponseWrapper, MessageResponse, TaskStatus
from app.schemas.common import UserContext
from app.schemas.cover_letter import (
    CoverLetterCreate, CoverLetterUpdate, CoverLetterResponse,
    CoverLetterGenerationRequest, CoverLetterGenerationResponse,
    CoverLetterTemplateListResponse,
    CoverLetterOptimizationRequest, CoverLetterOptimizationResponse,
    CoverLetterExportRequest, CoverLetterExportResponse,
    CoverLetterAnalysisRequest, JobDescriptionCreate, JobDescriptionResponse,
    CoverLetterSummary, CoverLetterStatsResponse,
    BulkCoverLetterCreate, BulkCoverLetterResponse,
    TemplateSuggestionRequest, TemplateSuggestionResponse
)
from app.services.cover_letter_service import CoverLetterService
from app.utils.exceptions import AppException

router = APIRouter()


@router.post("/", response_model=ResponseWrapper[CoverLetterResponse], status_code=201)
async def create_cover_letter(
        cover_letter_data: CoverLetterCreate,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Create a new cover letter.

    - **title**: Cover letter title (required)
    - **content**: Cover letter content (required)
    - **type**: Cover letter type (standard, email, networking, etc.)
    - **tone**: Cover letter tone (professional, conversational, etc.)
    - **job_title**: Target job title (optional)
    - **company_name**: Target company name (optional)
    """
    try:
        cover_letter = await cover_letter_service.create_cover_letter(
            current_user.user_id, cover_letter_data
        )
        return ResponseWrapper(
            data=cover_letter,
            message="Cover letter created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/generate", response_model=ResponseWrapper[CoverLetterGenerationResponse])
async def generate_ai_cover_letter(
        generation_request: CoverLetterGenerationRequest,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Generate a cover letter using AI.

    - **job_description**: Job description (required)
    - **job_title**: Job title (required)
    - **company_name**: Company name (required)
    - **hiring_manager_name**: Hiring manager name (optional)
    - **resume_id**: Resume to base content on (optional)
    - **template_id**: Template to use (optional)
    - **tone**: Desired tone
    - **personalization_level**: Level of personalization (1-5)
    """
    try:
        result = await cover_letter_service.generate_ai_cover_letter(
            current_user.user_id, generation_request
        )
        return ResponseWrapper(
            data=result,
            message="Cover letter generated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=ResponseWrapper[Dict[str, Any]])
async def list_cover_letters(
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Page size"),
        status: Optional[CoverLetterStatus] = Query(None, description="Filter by status"),
        type: Optional[CoverLetterType] = Query(None, description="Filter by type"),
        company_name: Optional[str] = Query(None, description="Filter by company name"),
        search: Optional[str] = Query(None, description="Search in title, description, job title"),
        sort_by: str = Query("updated_at", description="Sort field"),
        sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    List user's cover letters with pagination and filtering.

    - **page**: Page number (default: 1)
    - **size**: Items per page (default: 20, max: 100)
    - **status**: Filter by cover letter status
    - **type**: Filter by cover letter type
    - **company_name**: Filter by company name
    - **search**: Search in title, description, job title, company name
    - **sort_by**: Field to sort by (default: updated_at)
    - **sort_order**: Sort order asc/desc (default: desc)
    """
    try:
        result = await cover_letter_service.list_cover_letters(
            user_id=current_user.user_id,
            page=page,
            size=size,
            status=status,
            type=type,
            company_name=company_name,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return ResponseWrapper(
            data=result,
            message="Cover letters retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{cover_letter_id}", response_model=ResponseWrapper[CoverLetterResponse])
async def get_cover_letter(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID"),
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Get a specific cover letter by ID.

    - **cover_letter_id**: UUID of the cover letter to retrieve
    """
    try:
        cover_letter = await cover_letter_service.get_cover_letter(
            cover_letter_id, current_user.user_id
        )
        return ResponseWrapper(
            data=cover_letter,
            message="Cover letter retrieved successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{cover_letter_id}", response_model=ResponseWrapper[CoverLetterResponse])
async def update_cover_letter(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID"),
        update_data: CoverLetterUpdate = ...,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Update a cover letter.

    - **cover_letter_id**: UUID of the cover letter to update
    - **update_data**: Fields to update (partial update supported)
    """
    try:
        cover_letter = await cover_letter_service.update_cover_letter(
            cover_letter_id, current_user.user_id, update_data
        )
        return ResponseWrapper(
            data=cover_letter,
            message="Cover letter updated successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{cover_letter_id}", response_model=ResponseWrapper[MessageResponse])
async def delete_cover_letter(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID"),
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Delete a cover letter (soft delete).

    - **cover_letter_id**: UUID of the cover letter to delete
    """
    try:
        await cover_letter_service.delete_cover_letter(cover_letter_id, current_user.user_id)
        return ResponseWrapper(
            data=MessageResponse(message="Cover letter deleted successfully"),
            message="Cover letter deleted successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{cover_letter_id}/duplicate", response_model=ResponseWrapper[CoverLetterResponse])
async def duplicate_cover_letter(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID to duplicate"),
        new_title: Optional[str] = Query(None, description="Title for the duplicate"),
        new_company: Optional[str] = Query(None, description="Company name for the duplicate"),
        new_job_title: Optional[str] = Query(None, description="Job title for the duplicate"),
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Duplicate an existing cover letter.

    - **cover_letter_id**: UUID of the cover letter to duplicate
    - **new_title**: Optional new title for the duplicate
    - **new_company**: Optional new company name
    - **new_job_title**: Optional new job title
    """
    try:
        cover_letter = await cover_letter_service.duplicate_cover_letter(
            cover_letter_id, current_user.user_id, new_title, new_company, new_job_title
        )
        return ResponseWrapper(
            data=cover_letter,
            message="Cover letter duplicated successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{cover_letter_id}/optimize", response_model=ResponseWrapper[CoverLetterOptimizationResponse])
async def optimize_cover_letter(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID"),
        optimization_request: CoverLetterOptimizationRequest = ...,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Optimize cover letter content using AI.

    - **cover_letter_id**: UUID of the cover letter to optimize
    - **optimization_type**: Type of optimization (keyword, tone, length, ats)
    - **job_description**: Optional job description for optimization context
    - **target_keywords**: Specific keywords to optimize for
    - **optimization_level**: Optimization intensity (1-5)
    """
    try:
        result = await cover_letter_service.optimize_cover_letter(
            cover_letter_id, current_user.user_id, optimization_request
        )
        return ResponseWrapper(
            data=result,
            message="Cover letter optimized successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{cover_letter_id}/export", response_model=ResponseWrapper[CoverLetterExportResponse])
async def export_cover_letter(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID"),
        export_request: CoverLetterExportRequest = ...,
        background_tasks: BackgroundTasks = ...,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Export cover letter in specified format.

    - **cover_letter_id**: UUID of the cover letter to export
    - **format**: Export format (pdf, docx, html)
    - **template_id**: Optional template override
    - **include_contact_info**: Whether to include contact information
    - **letterhead**: Custom letterhead settings
    """
    try:
        # This would be implemented in the service
        export_result = {
            "file_url": f"/downloads/cover-letter-{cover_letter_id}.{export_request.format}",
            "file_name": f"cover-letter-{cover_letter_id}.{export_request.format}",
            "file_size": 1024000,  # Placeholder
            "format": export_request.format,
            "expires_at": datetime.utcnow() + timedelta(hours=24)
        }

        return ResponseWrapper(
            data=export_result,
            message="Cover letter export initiated successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Template endpoints
@router.get("/templates/", response_model=ResponseWrapper[CoverLetterTemplateListResponse])
async def get_cover_letter_templates(
        type: Optional[CoverLetterType] = Query(None, description="Filter by type"),
        tone: Optional[CoverLetterTone] = Query(None, description="Filter by tone"),
        industry: Optional[str] = Query(None, description="Filter by target industry"),
        is_premium: Optional[bool] = Query(None, description="Filter by premium status"),
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Page size"),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Get available cover letter templates.

    - **type**: Filter by template type
    - **tone**: Filter by template tone
    - **industry**: Filter by target industry
    - **is_premium**: Filter by premium status
    - **page**: Page number
    - **size**: Items per page
    """
    try:
        templates = await cover_letter_service.get_templates(
            type=type,
            tone=tone,
            industry=industry,
            is_premium=is_premium,
            page=page,
            size=size
        )

        result = CoverLetterTemplateListResponse(
            templates=templates,
            total=len(templates),
            page=page,
            size=size,
            pages=1  # Templates are cached, so we return all in one page for now
        )

        return ResponseWrapper(
            data=result,
            message="Templates retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/templates/suggest", response_model=ResponseWrapper[TemplateSuggestionResponse])
async def suggest_templates(
        suggestion_request: TemplateSuggestionRequest,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Get template suggestions based on job description and preferences.

    - **job_description**: Job description for analysis
    - **job_title**: Job title
    - **company_name**: Company name
    - **industry**: Industry (optional)
    - **experience_level**: Experience level (optional)
    - **preferred_tone**: Preferred tone (optional)
    """
    try:
        # This would be implemented using AI to analyze the job and suggest templates
        suggestions = {
            "suggested_templates": [],
            "match_scores": {},
            "reasoning": {},
            "best_match": None
        }

        return ResponseWrapper(
            data=suggestions,
            message="Template suggestions generated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Job description management
@router.post("/job-descriptions/", response_model=ResponseWrapper[JobDescriptionResponse])
async def save_job_description(
        job_data: JobDescriptionCreate,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Save a job description for future use.

    - **title**: Job title (required)
    - **company_name**: Company name (required)
    - **description**: Job description (required)
    - **requirements**: Job requirements (optional)
    - **responsibilities**: Job responsibilities (optional)
    - **location**: Job location (optional)
    """
    try:
        job_description = await cover_letter_service.save_job_description(
            current_user.user_id, job_data
        )
        return ResponseWrapper(
            data=job_description,
            message="Job description saved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Bulk operations
@router.post("/bulk/create", response_model=ResponseWrapper[BulkCoverLetterResponse])
async def bulk_create_cover_letters(
        bulk_request: BulkCoverLetterCreate,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Create multiple cover letters based on a base template and job descriptions.

    - **base_cover_letter_id**: Base cover letter to use as template
    - **job_descriptions**: List of job descriptions to create cover letters for
    - **customization_level**: Level of customization per job (1-5)
    - **include_company_research**: Whether to include company research
    """
    try:
        if len(bulk_request.job_descriptions) > 20:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 job descriptions allowed in bulk operation"
            )

        # This would queue a background task for bulk creation
        task_id = uuid.uuid4()  # Placeholder

        result = BulkCoverLetterResponse(
            task_id=task_id,
            total_jobs=len(bulk_request.job_descriptions),
            estimated_completion_time=len(bulk_request.job_descriptions) * 30,  # 30 seconds per job
            status="pending"
        )

        return ResponseWrapper(
            data=result,
            message="Bulk cover letter creation queued successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Analysis endpoints
@router.get("/{cover_letter_id}/analysis", response_model=ResponseWrapper[Dict[str, Any]])
async def get_cover_letter_analysis(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID"),
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Get the latest analysis for a cover letter.

    - **cover_letter_id**: UUID of the cover letter
    """
    try:
        # This would be implemented in the service
        analysis = {
            "content_quality_score": 85.0,
            "personalization_score": 78.0,
            "keyword_match_score": 82.0,
            "tone_consistency_score": 90.0,
            "overall_score": 83.8,
            "suggestions": [],
            "analysis_date": datetime.utcnow()
        }

        return ResponseWrapper(
            data=analysis,
            message="Cover letter analysis retrieved successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{cover_letter_id}/analyze", response_model=ResponseWrapper[TaskStatus])
async def request_cover_letter_analysis(
        cover_letter_id: uuid.UUID = Path(..., description="Cover letter ID"),
        analysis_request: CoverLetterAnalysisRequest = ...,
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Request analysis of a cover letter.

    - **cover_letter_id**: UUID of the cover letter to analyze
    - **analysis_types**: Types of analysis to perform
    - **job_description**: Optional job description for matching
    - **include_suggestions**: Whether to include improvement suggestions
    """
    try:
        # This would queue an analysis task
        task_id = uuid.uuid4()  # Placeholder

        task_status = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0,
            created_at=datetime.utcnow()
        )

        return ResponseWrapper(
            data=task_status,
            message="Cover letter analysis queued successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Statistics and summary endpoints
@router.get("/stats/overview", response_model=ResponseWrapper[CoverLetterStatsResponse])
async def get_cover_letter_statistics(
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Get user's cover letter statistics and overview.
    """
    try:
        stats = {
            "total_cover_letters": 0,
            "active_cover_letters": 0,
            "draft_cover_letters": 0,
            "ai_generated_count": 0,
            "total_views": 0,
            "total_downloads": 0,
            "average_score": None,
            "average_personalization_score": None,
            "most_used_template": None,
            "most_common_tone": None,
            "recent_activity": []
        }

        return ResponseWrapper(
            data=stats,
            message="Statistics retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=ResponseWrapper[List[CoverLetterSummary]])
async def get_cover_letter_summaries(
        limit: int = Query(10, ge=1, le=50, description="Number of summaries to return"),
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Get lightweight cover letter summaries for quick overview.

    - **limit**: Maximum number of summaries to return
    """
    try:
        cover_letters = await cover_letter_service.list_cover_letters(
            user_id=current_user.user_id,
            page=1,
            size=limit,
            sort_by="updated_at",
            sort_order="desc"
        )

        summaries = [
            CoverLetterSummary(
                id=cl["id"],
                title=cl["title"],
                status=cl["status"],
                type=cl["type"],
                tone=cl["tone"],
                job_title=cl.get("job_title"),
                company_name=cl.get("company_name"),
                overall_score=cl.get("overall_score"),
                view_count=cl.get("view_count", 0),
                created_at=cl["created_at"],
                updated_at=cl["updated_at"]
            )
            for cl in cover_letters["cover_letters"]
        ]

        return ResponseWrapper(
            data=summaries,
            message="Cover letter summaries retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Search and filtering
@router.get("/search", response_model=ResponseWrapper[Dict[str, Any]])
async def search_cover_letters(
        query: str = Query(..., min_length=1, description="Search query"),
        filters: Optional[Dict[str, Any]] = Query(None, description="Additional filters"),
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Page size"),
        current_user: UserContext = Depends(get_current_user),
        cover_letter_service: CoverLetterService = Depends(get_cover_letter_service)
):
    """
    Advanced search for cover letters.

    - **query**: Search query
    - **filters**: Additional filters (JSON object)
    - **page**: Page number
    - **size**: Items per page
    """
    try:
        # This would implement advanced search functionality
        result = await cover_letter_service.list_cover_letters(
            user_id=current_user.user_id,
            search=query,
            page=page,
            size=size
        )

        return ResponseWrapper(
            data=result,
            message="Search completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
