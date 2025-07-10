"""
Resume API endpoints.
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from app.schemas.resume import (
    ResumeCreate, ResumeUpdate, ResumeResponse, ResumeListResponse,
    ResumeTemplateResponse, ResumeTemplateListResponse, ResumeExportRequest,
    ResumeExportResponse, ResumeAnalysisRequest, ResumeAnalysisResponse,
    ResumeShareCreate, ResumeShareResponse, ResumeSkillCreate, ResumeSkillResponse,
    ResumeSummary, ResumeStatsResponse
)
from app.schemas.common import (
    ResponseWrapper, PaginatedResponse, MessageResponse, TaskStatus
)
from app.models.resume import ResumeStatus, TemplateCategory
from app.services.resume_service import ResumeService
from app.api.deps import get_current_user, get_resume_service
from app.schemas.common import UserContext
from app.utils.exceptions import AppException

router = APIRouter()


@router.post("/", response_model=ResponseWrapper[ResumeResponse], status_code=201)
async def create_resume(
        resume_data: ResumeCreate,
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Create a new resume.

    - **title**: Resume title (required)
    - **personal_info**: Personal information including name, email, etc. (required)
    - **template_id**: Template to use (optional)
    - **work_experience**: Work experience entries (optional)
    - **education**: Education entries (optional)
    - **skills**: Skills list (optional)
    """
    try:
        resume = await resume_service.create_resume(current_user.user_id, resume_data)
        return ResponseWrapper(
            data=resume,
            message="Resume created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=ResponseWrapper[ResumeListResponse])
async def list_resumes(
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Page size"),
        status: Optional[ResumeStatus] = Query(None, description="Filter by status"),
        search: Optional[str] = Query(None, description="Search in title, description, job title"),
        sort_by: str = Query("updated_at", description="Sort field"),
        sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    List user's resumes with pagination and filtering.

    - **page**: Page number (default: 1)
    - **size**: Items per page (default: 20, max: 100)
    - **status**: Filter by resume status
    - **search**: Search in title, description, target job title
    - **sort_by**: Field to sort by (default: updated_at)
    - **sort_order**: Sort order asc/desc (default: desc)
    """
    try:
        result = await resume_service.list_resumes(
            user_id=current_user.user_id,
            page=page,
            size=size,
            status=status,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return ResponseWrapper(
            data=result,
            message="Resumes retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{resume_id}", response_model=ResponseWrapper[ResumeResponse])
async def get_resume(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Get a specific resume by ID.

    - **resume_id**: UUID of the resume to retrieve
    """
    try:
        resume = await resume_service.get_resume(resume_id, current_user.user_id)
        return ResponseWrapper(
            data=resume,
            message="Resume retrieved successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{resume_id}", response_model=ResponseWrapper[ResumeResponse])
async def update_resume(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        update_data: ResumeUpdate = ...,
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Update a resume.

    - **resume_id**: UUID of the resume to update
    - **update_data**: Fields to update (partial update supported)
    """
    try:
        resume = await resume_service.update_resume(
            resume_id, current_user.user_id, update_data
        )
        return ResponseWrapper(
            data=resume,
            message="Resume updated successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{resume_id}", response_model=ResponseWrapper[MessageResponse])
async def delete_resume(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Delete a resume (soft delete).

    - **resume_id**: UUID of the resume to delete
    """
    try:
        await resume_service.delete_resume(resume_id, current_user.user_id)
        return ResponseWrapper(
            data=MessageResponse(message="Resume deleted successfully"),
            message="Resume deleted successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{resume_id}/duplicate", response_model=ResponseWrapper[ResumeResponse])
async def duplicate_resume(
        resume_id: uuid.UUID = Path(..., description="Resume ID to duplicate"),
        new_title: Optional[str] = Query(None, description="Title for the duplicate"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Duplicate an existing resume.

    - **resume_id**: UUID of the resume to duplicate
    - **new_title**: Optional new title for the duplicate
    """
    try:
        resume = await resume_service.duplicate_resume(
            resume_id, current_user.user_id, new_title
        )
        return ResponseWrapper(
            data=resume,
            message="Resume duplicated successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{resume_id}/publish", response_model=ResponseWrapper[ResumeResponse])
async def publish_resume(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Publish a resume (make it active and generate PDF).

    - **resume_id**: UUID of the resume to publish
    """
    try:
        resume = await resume_service.publish_resume(resume_id, current_user.user_id)
        return ResponseWrapper(
            data=resume,
            message="Resume published successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{resume_id}/export", response_model=ResponseWrapper[ResumeExportResponse])
async def export_resume(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        export_request: ResumeExportRequest = ...,
        background_tasks: BackgroundTasks = ...,
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Export resume in specified format.

    - **resume_id**: UUID of the resume to export
    - **format**: Export format (pdf, docx, html)
    - **template_id**: Optional template override
    - **custom_settings**: Custom export settings
    """
    try:
        export_result = await resume_service.export_resume(
            resume_id, current_user.user_id, export_request
        )
        return ResponseWrapper(
            data=export_result,
            message="Resume export initiated successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{resume_id}/analysis", response_model=ResponseWrapper[Dict[str, Any]])
async def get_resume_analysis(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Get the latest analysis for a resume.

    - **resume_id**: UUID of the resume
    """
    try:
        analysis = await resume_service.get_resume_analysis(resume_id, current_user.user_id)
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="No analysis found for this resume"
            )

        return ResponseWrapper(
            data=analysis,
            message="Resume analysis retrieved successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{resume_id}/analyze", response_model=ResponseWrapper[TaskStatus])
async def request_resume_analysis(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        analysis_request: ResumeAnalysisRequest = ...,
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Request analysis of a resume.

    - **resume_id**: UUID of the resume to analyze
    - **analysis_types**: Types of analysis to perform
    - **job_description**: Optional job description for matching
    - **include_suggestions**: Whether to include improvement suggestions
    """
    try:
        task_id = await resume_service.request_analysis(
            resume_id=resume_id,
            user_id=current_user.user_id,
            analysis_types=analysis_request.analysis_types,
            job_description=analysis_request.job_description
        )

        task_status = TaskStatus(
            task_id=uuid.UUID(task_id),
            status="pending",
            progress=0,
            created_at=datetime.utcnow()
        )

        return ResponseWrapper(
            data=task_status,
            message="Resume analysis queued successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Skills management endpoints
@router.post("/{resume_id}/skills", response_model=ResponseWrapper[ResumeSkillResponse])
async def add_skill_to_resume(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        skill_data: ResumeSkillCreate = ...,
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Add a skill to a resume.

    - **resume_id**: UUID of the resume
    - **skill_name**: Name of the skill (required)
    - **skill_category**: Category of the skill (optional)
    - **proficiency_level**: Proficiency level 1-5 (optional)
    - **years_experience**: Years of experience (optional)
    """
    try:
        skill = await resume_service.add_skill(resume_id, current_user.user_id, skill_data)
        return ResponseWrapper(
            data=skill,
            message="Skill added successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Sharing endpoints
@router.post("/{resume_id}/share", response_model=ResponseWrapper[ResumeShareResponse])
async def create_resume_share_link(
        resume_id: uuid.UUID = Path(..., description="Resume ID"),
        share_data: ResumeShareCreate = ...,
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Create a shareable link for a resume.

    - **resume_id**: UUID of the resume to share
    - **share_type**: Type of sharing (public, private, password)
    - **password**: Password for protected shares (optional)
    - **allow_download**: Whether to allow downloads
    - **expires_at**: Expiration date (optional)
    """
    try:
        share_link = await resume_service.create_share_link(
            resume_id, current_user.user_id, share_data
        )
        return ResponseWrapper(
            data=share_link,
            message="Share link created successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Template endpoints
@router.get("/templates/", response_model=ResponseWrapper[ResumeTemplateListResponse])
async def get_resume_templates(
        category: Optional[TemplateCategory] = Query(None, description="Filter by category"),
        is_premium: Optional[bool] = Query(None, description="Filter by premium status"),
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Page size"),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Get available resume templates.

    - **category**: Filter by template category
    - **is_premium**: Filter by premium status
    - **page**: Page number
    - **size**: Items per page
    """
    try:
        templates = await resume_service.get_templates(
            category=category.value if category else None,
            is_premium=is_premium,
            page=page,
            size=size
        )

        result = ResumeTemplateListResponse(
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


@router.get("/templates/{template_id}", response_model=ResponseWrapper[ResumeTemplateResponse])
async def get_resume_template(
        template_id: uuid.UUID = Path(..., description="Template ID"),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Get a specific resume template.

    - **template_id**: UUID of the template
    """
    try:
        template = await resume_service.get_template(template_id)
        return ResponseWrapper(
            data=template,
            message="Template retrieved successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Statistics and summary endpoints
@router.get("/stats/overview", response_model=ResponseWrapper[ResumeStatsResponse])
async def get_resume_statistics(
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Get user's resume statistics and overview.
    """
    try:
        # This would be implemented in the resume service
        stats = {
            "total_resumes": 0,
            "active_resumes": 0,
            "draft_resumes": 0,
            "total_views": 0,
            "total_downloads": 0,
            "average_score": None,
            "most_used_template": None,
            "recent_activity": []
        }

        return ResponseWrapper(
            data=stats,
            message="Statistics retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=ResponseWrapper[List[ResumeSummary]])
async def get_resume_summaries(
        limit: int = Query(10, ge=1, le=50, description="Number of summaries to return"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Get lightweight resume summaries for quick overview.

    - **limit**: Maximum number of summaries to return
    """
    try:
        resumes = await resume_service.list_resumes(
            user_id=current_user.user_id,
            page=1,
            size=limit,
            sort_by="updated_at",
            sort_order="desc"
        )

        summaries = [
            ResumeSummary(
                id=resume.id,
                title=resume.title,
                status=resume.status,
                template_id=resume.template_id,
                overall_score=resume.overall_score,
                view_count=resume.view_count,
                created_at=resume.created_at,
                updated_at=resume.updated_at
            )
            for resume in resumes.resumes
        ]

        return ResponseWrapper(
            data=summaries,
            message="Resume summaries retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch operations (if enabled)
@router.post("/batch/analyze", response_model=ResponseWrapper[TaskStatus])
async def batch_analyze_resumes(
        resume_ids: List[uuid.UUID] = Query(..., description="Resume IDs to analyze"),
        analysis_types: List[str] = Query(["comprehensive"], description="Analysis types"),
        job_description: Optional[str] = Query(None, description="Job description for context"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Analyze multiple resumes in batch.

    - **resume_ids**: List of resume IDs to analyze
    - **analysis_types**: Types of analysis to perform
    - **job_description**: Optional job description for matching
    """
    try:
        if len(resume_ids) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 resumes allowed in batch operation"
            )

        # This would queue a batch analysis task
        task_id = uuid.uuid4()  # Placeholder

        task_status = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0,
            created_at=datetime.utcnow()
        )

        return ResponseWrapper(
            data=task_status,
            message="Batch analysis queued successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# File upload endpoint for resume parsing
@router.post("/upload", response_model=ResponseWrapper[ResumeResponse])
async def upload_and_parse_resume(
        file: UploadFile = File(..., description="Resume file (PDF, DOCX, TXT)"),
        title: Optional[str] = Form(None, description="Resume title"),
        current_user: UserContext = Depends(get_current_user),
        resume_service: ResumeService = Depends(get_resume_service)
):
    """
    Upload and parse a resume file.

    - **file**: Resume file to upload and parse
    - **title**: Optional title for the resume
    """
    try:
        # Validate file type
        allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                         "text/plain"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload PDF, DOCX, or TXT files."
            )

        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 10MB."
            )

        # This would be implemented to parse the uploaded file
        # and create a resume from the extracted content
        parsed_data = ResumeCreate(
            title=title or f"Imported Resume - {file.filename}",
            personal_info={"first_name": "Imported", "last_name": "User", "email": "imported@example.com"},
            description="Resume imported from uploaded file"
        )

        resume = await resume_service.create_resume(current_user.user_id, parsed_data)

        return ResponseWrapper(
            data=resume,
            message="Resume uploaded and parsed successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))