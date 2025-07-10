"""
Resume service with business logic for resume operations.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db_context
from app.core.redis import cache_manager
from app.models.resume import (
    Resume, ResumeTemplate, ResumeAnalysis, ResumeVersion,
    ResumeSkill, ResumeShare, ResumeStatus
)
from app.schemas.resume import (
    ResumeCreate, ResumeUpdate, ResumeResponse, ResumeListResponse,
    ResumeTemplateCreate, ResumeTemplateUpdate, ResumeTemplateResponse,
    ResumeAnalysisCreate, ResumeVersionCreate, ResumeSkillCreate,
    ResumeShareCreate, ResumeExportRequest
)
from app.utils.exceptions import AppException
from app.services.ai_service import AIService
from app.services.pdf_service import PDFService
from app.tasks.resume_tasks import generate_resume_pdf_task, analyze_resume_task


class ResumeService:
    """Service for resume operations."""

    def __init__(self, ai_service: AIService = None, pdf_service: PDFService = None):
        self.ai_service = ai_service or AIService()
        self.pdf_service = pdf_service or PDFService()

    async def create_resume(
            self,
            user_id: uuid.UUID,
            resume_data: ResumeCreate
    ) -> ResumeResponse:
        """Create a new resume."""
        async with get_db_context() as db:
            # Validate template if provided
            if resume_data.template_id:
                template = await self._get_template_by_id(db, resume_data.template_id)
                if not template or not template.is_active:
                    raise AppException(
                        message="Invalid or inactive template",
                        status_code=400,
                        error_code="INVALID_TEMPLATE"
                    )

            # Create resume instance
            resume = Resume(
                user_id=user_id,
                title=resume_data.title,
                description=resume_data.description,
                template_id=resume_data.template_id,
                personal_info=resume_data.personal_info,
                work_experience=resume_data.work_experience,
                education=resume_data.education,
                skills=resume_data.skills,
                projects=resume_data.projects,
                certifications=resume_data.certifications,
                languages=resume_data.languages,
                awards=resume_data.awards,
                references=resume_data.references,
                custom_sections=resume_data.custom_sections,
                target_job_title=resume_data.target_job_title,
                target_industry=resume_data.target_industry,
                keywords=resume_data.keywords,
                is_public=resume_data.is_public,
                status=ResumeStatus.DRAFT
            )

            db.add(resume)
            await db.commit()
            await db.refresh(resume)

            # Generate initial version
            await self._create_version(db, resume, "Initial version", user_id)

            # Cache the resume
            await self._cache_resume(resume)

            # Trigger background analysis if AI is enabled
            if resume_data.template_id:
                analyze_resume_task.delay(str(resume.id))

            return await self._to_response(resume)

    async def get_resume(self, resume_id: uuid.UUID, user_id: uuid.UUID) -> ResumeResponse:
        """Get a resume by ID."""
        # Try cache first
        cache_key = f"resume:{resume_id}"
        cached_resume = await cache_manager.get(cache_key) if cache_manager else None

        if cached_resume:
            # Verify user access
            if cached_resume.get("user_id") != str(user_id):
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )
            return ResumeResponse(**cached_resume)

        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            if resume.user_id != user_id and not resume.is_public:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Increment view count
            resume.view_count += 1
            await db.commit()

            # Cache the resume
            await self._cache_resume(resume)

            return await self._to_response(resume)

    async def update_resume(
            self,
            resume_id: uuid.UUID,
            user_id: uuid.UUID,
            update_data: ResumeUpdate
    ) -> ResumeResponse:
        """Update a resume."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Store old data for version tracking
            old_data = await self._serialize_resume_data(resume)

            # Update fields
            update_fields = update_data.model_dump(exclude_unset=True)
            for field, value in update_fields.items():
                setattr(resume, field, value)

            resume.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(resume)

            # Create new version if significant changes
            if self._has_significant_changes(update_fields):
                await self._create_version(
                    db, resume, "Updated resume", user_id, old_data
                )

            # Clear cache
            await self._clear_resume_cache(resume_id)

            # Trigger re-analysis if content changed
            if any(field in update_fields for field in [
                'work_experience', 'education', 'skills', 'projects'
            ]):
                analyze_resume_task.delay(str(resume.id))

            return await self._to_response(resume)

    async def delete_resume(self, resume_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Delete a resume (soft delete)."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            resume.status = ResumeStatus.DELETED
            resume.deleted_at = datetime.utcnow()
            await db.commit()

            # Clear cache
            await self._clear_resume_cache(resume_id)

            return True

    async def list_resumes(
            self,
            user_id: uuid.UUID,
            page: int = 1,
            size: int = 20,
            status: Optional[ResumeStatus] = None,
            search: Optional[str] = None,
            sort_by: str = "updated_at",
            sort_order: str = "desc"
    ) -> ResumeListResponse:
        """List user's resumes with pagination."""
        async with get_db_context() as db:
            # Build query
            query = db.query(Resume).filter(
                Resume.user_id == user_id,
                Resume.deleted_at.is_(None)
            )

            # Add filters
            if status:
                query = query.filter(Resume.status == status)

            if search:
                search_filter = or_(
                    Resume.title.ilike(f"%{search}%"),
                    Resume.description.ilike(f"%{search}%"),
                    Resume.target_job_title.ilike(f"%{search}%")
                )
                query = query.filter(search_filter)

            # Add sorting
            if sort_order == "desc":
                query = query.order_by(desc(getattr(Resume, sort_by)))
            else:
                query = query.order_by(getattr(Resume, sort_by))

            # Get total count
            total = await db.scalar(
                query.with_only_columns(func.count(Resume.id))
            )

            # Apply pagination
            offset = (page - 1) * size
            resumes = await db.scalars(
                query.offset(offset).limit(size)
            )

            # Convert to response
            resume_responses = []
            for resume in resumes:
                resume_responses.append(await self._to_response(resume))

            return ResumeListResponse(
                resumes=resume_responses,
                total=total,
                page=page,
                size=size,
                pages=(total + size - 1) // size
            )

    async def duplicate_resume(
            self,
            resume_id: uuid.UUID,
            user_id: uuid.UUID,
            new_title: Optional[str] = None
    ) -> ResumeResponse:
        """Duplicate an existing resume."""
        async with get_db_context() as db:
            original = await db.get(Resume, resume_id)

            if not original or original.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Create duplicate
            duplicate_data = await self._serialize_resume_data(original)
            duplicate_data["title"] = new_title or f"{original.title} (Copy)"
            duplicate_data["status"] = ResumeStatus.DRAFT
            duplicate_data["is_public"] = False
            duplicate_data["share_token"] = None

            # Remove computed fields
            for field in ["id", "created_at", "updated_at", "view_count",
                          "download_count", "share_count", "file_path", "pdf_path"]:
                duplicate_data.pop(field, None)

            duplicate = Resume(**duplicate_data)
            db.add(duplicate)
            await db.commit()
            await db.refresh(duplicate)

            return await self._to_response(duplicate)

    async def publish_resume(self, resume_id: uuid.UUID, user_id: uuid.UUID) -> ResumeResponse:
        """Publish a resume (make it active and potentially public)."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Validate resume completeness
            if not self._is_resume_complete(resume):
                raise AppException(
                    message="Resume is incomplete and cannot be published",
                    status_code=400,
                    error_code="INCOMPLETE_RESUME"
                )

            resume.status = ResumeStatus.PUBLISHED
            await db.commit()

            # Generate PDF if not exists
            if not resume.pdf_path:
                generate_resume_pdf_task.delay(str(resume.id))

            # Clear cache
            await self._clear_resume_cache(resume_id)

            return await self._to_response(resume)

    async def export_resume(
            self,
            resume_id: uuid.UUID,
            user_id: uuid.UUID,
            export_request: ResumeExportRequest
    ) -> Dict[str, Any]:
        """Export resume in specified format."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Generate export based on format
            if export_request.format.lower() == "pdf":
                return await self.pdf_service.generate_pdf(
                    resume, export_request.template_id, export_request.custom_settings
                )
            elif export_request.format.lower() == "docx":
                return await self.pdf_service.generate_docx(resume, export_request.custom_settings)
            elif export_request.format.lower() == "html":
                return await self.pdf_service.generate_html(resume, export_request.template_id)
            else:
                raise AppException(
                    message="Unsupported export format",
                    status_code=400,
                    error_code="UNSUPPORTED_FORMAT"
                )

    # Template management methods
    async def get_templates(
            self,
            category: Optional[str] = None,
            is_premium: Optional[bool] = None,
            page: int = 1,
            size: int = 20
    ) -> List[ResumeTemplateResponse]:
        """Get available resume templates."""
        cache_key = f"templates:{category}:{is_premium}:{page}:{size}"
        cached_templates = await cache_manager.get(cache_key) if cache_manager else None

        if cached_templates:
            return [ResumeTemplateResponse(**t) for t in cached_templates]

        async with get_db_context() as db:
            query = db.query(ResumeTemplate).filter(ResumeTemplate.is_active == True)

            if category:
                query = query.filter(ResumeTemplate.category == category)

            if is_premium is not None:
                query = query.filter(ResumeTemplate.is_premium == is_premium)

            query = query.order_by(ResumeTemplate.sort_order, ResumeTemplate.created_at)

            # Apply pagination
            offset = (page - 1) * size
            templates = await db.scalars(query.offset(offset).limit(size))

            template_responses = []
            for template in templates:
                template_responses.append(await self._template_to_response(template))

            # Cache templates
            if cache_manager:
                await cache_manager.set(
                    cache_key,
                    [t.model_dump() for t in template_responses],
                    ttl=3600  # 1 hour
                )

            return template_responses

    async def get_template(self, template_id: uuid.UUID) -> ResumeTemplateResponse:
        """Get a specific template."""
        cache_key = f"template:{template_id}"
        cached_template = await cache_manager.get(cache_key) if cache_manager else None

        if cached_template:
            return ResumeTemplateResponse(**cached_template)

        async with get_db_context() as db:
            template = await self._get_template_by_id(db, template_id)

            if not template:
                raise AppException(
                    message="Template not found",
                    status_code=404,
                    error_code="TEMPLATE_NOT_FOUND"
                )

            response = await self._template_to_response(template)

            # Cache template
            if cache_manager:
                await cache_manager.set(cache_key, response.model_dump(), ttl=3600)

            return response

    # Analysis methods
    async def get_resume_analysis(
            self,
            resume_id: uuid.UUID,
            user_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Get latest analysis for a resume."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Get latest analysis
            analysis = await db.scalar(
                db.query(ResumeAnalysis)
                .filter(ResumeAnalysis.resume_id == resume_id)
                .order_by(desc(ResumeAnalysis.created_at))
                .limit(1)
            )

            if not analysis:
                return None

            return {
                "id": analysis.id,
                "analysis_type": analysis.analysis_type,
                "overall_score": analysis.overall_score,
                "ats_score": analysis.ats_score,
                "content_score": analysis.content_score,
                "format_score": analysis.format_score,
                "keyword_analysis": analysis.keyword_analysis,
                "content_analysis": analysis.content_analysis,
                "format_analysis": analysis.format_analysis,
                "suggestions": analysis.suggestions,
                "created_at": analysis.created_at,
            }

    async def request_analysis(
            self,
            resume_id: uuid.UUID,
            user_id: uuid.UUID,
            analysis_types: List[str] = None,
            job_description: Optional[str] = None
    ) -> str:
        """Request resume analysis."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Queue analysis task
            task = analyze_resume_task.delay(
                str(resume_id),
                analysis_types or ["comprehensive"],
                job_description
            )

            return str(task.id)

    # Skills management
    async def add_skill(
            self,
            resume_id: uuid.UUID,
            user_id: uuid.UUID,
            skill_data: ResumeSkillCreate
    ) -> Dict[str, Any]:
        """Add a skill to resume."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Check if skill already exists
            existing = await db.scalar(
                db.query(ResumeSkill).filter(
                    and_(
                        ResumeSkill.resume_id == resume_id,
                        ResumeSkill.skill_name == skill_data.skill_name
                    )
                )
            )

            if existing:
                raise AppException(
                    message="Skill already exists",
                    status_code=400,
                    error_code="SKILL_EXISTS"
                )

            skill = ResumeSkill(
                resume_id=resume_id,
                **skill_data.model_dump()
            )

            db.add(skill)
            await db.commit()
            await db.refresh(skill)

            # Clear resume cache
            await self._clear_resume_cache(resume_id)

            return {
                "id": skill.id,
                "skill_name": skill.skill_name,
                "skill_category": skill.skill_category,
                "proficiency_level": skill.proficiency_level,
                "years_experience": skill.years_experience,
                "is_featured": skill.is_featured,
                "sort_order": skill.sort_order,
            }

    # Sharing methods
    async def create_share_link(
            self,
            resume_id: uuid.UUID,
            user_id: uuid.UUID,
            share_data: ResumeShareCreate
    ) -> Dict[str, Any]:
        """Create a shareable link for resume."""
        async with get_db_context() as db:
            resume = await db.get(Resume, resume_id)

            if not resume or resume.user_id != user_id:
                raise AppException(
                    message="Resume not found",
                    status_code=404,
                    error_code="RESUME_NOT_FOUND"
                )

            # Generate share token
            import secrets
            share_token = secrets.token_urlsafe(32)

            # Hash password if provided
            password_hash = None
            if share_data.password:
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                password_hash = pwd_context.hash(share_data.password)

            share = ResumeShare(
                resume_id=resume_id,
                share_token=share_token,
                share_type=share_data.share_type,
                password_hash=password_hash,
                allow_download=share_data.allow_download,
                expires_at=share_data.expires_at,
                created_by=user_id
            )

            db.add(share)
            await db.commit()
            await db.refresh(share)

            return {
                "id": share.id,
                "share_token": share.share_token,
                "share_url": f"/shared/resume/{share.share_token}",
                "share_type": share.share_type,
                "allow_download": share.allow_download,
                "expires_at": share.expires_at,
                "created_at": share.created_at,
            }

    # Helper methods
    async def _get_template_by_id(self, db: AsyncSession, template_id: uuid.UUID) -> Optional[ResumeTemplate]:
        """Get template by ID."""
        return await db.get(ResumeTemplate, template_id)

    async def _to_response(self, resume: Resume) -> ResumeResponse:
        """Convert resume model to response."""
        return ResumeResponse(
            id=resume.id,
            user_id=resume.user_id,
            title=resume.title,
            description=resume.description,
            status=resume.status,
            template_id=resume.template_id,
            personal_info=resume.personal_info,
            work_experience=resume.work_experience,
            education=resume.education,
            skills=resume.skills,
            projects=resume.projects,
            certifications=resume.certifications,
            languages=resume.languages,
            awards=resume.awards,
            references=resume.references,
            custom_sections=resume.custom_sections,
            custom_css=resume.custom_css,
            theme_settings=resume.theme_settings,
            target_job_title=resume.target_job_title,
            target_industry=resume.target_industry,
            keywords=resume.keywords,
            file_path=resume.file_path,
            pdf_path=resume.pdf_path,
            file_size=resume.file_size,
            view_count=resume.view_count,
            download_count=resume.download_count,
            share_count=resume.share_count,
            ats_score=resume.ats_score,
            content_score=resume.content_score,
            format_score=resume.format_score,
            overall_score=resume.overall_score,
            last_analyzed_at=resume.last_analyzed_at,
            is_public=resume.is_public,
            is_featured=resume.is_featured,
            share_token=resume.share_token,
            created_at=resume.created_at,
            updated_at=resume.updated_at,
            deleted_at=resume.deleted_at,
        )

    async def _template_to_response(self, template: ResumeTemplate) -> ResumeTemplateResponse:
        """Convert template model to response."""
        return ResumeTemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            category=template.category,
            html_template=template.html_template,
            css_styles=template.css_styles,
            preview_image=template.preview_image,
            default_settings=template.default_settings,
            customizable_fields=template.customizable_fields,
            supported_sections=template.supported_sections,
            is_active=template.is_active,
            is_premium=template.is_premium,
            is_featured=template.is_featured,
            usage_count=template.usage_count,
            rating=template.rating,
            sort_order=template.sort_order,
            tags=template.tags,
            created_by=template.created_by,
            created_at=template.created_at,
            updated_at=template.updated_at,
        )

    async def _cache_resume(self, resume: Resume):
        """Cache resume data."""
        if cache_manager:
            cache_key = f"resume:{resume.id}"
            resume_data = (await self._to_response(resume)).model_dump()
            await cache_manager.set(cache_key, resume_data, ttl=1800)  # 30 minutes

    async def _clear_resume_cache(self, resume_id: uuid.UUID):
        """Clear resume cache."""
        if cache_manager:
            cache_key = f"resume:{resume_id}"
            await cache_manager.delete(cache_key)

    async def _serialize_resume_data(self, resume: Resume) -> Dict[str, Any]:
        """Serialize resume data for versioning."""
        return {
            "title": resume.title,
            "description": resume.description,
            "personal_info": resume.personal_info,
            "work_experience": resume.work_experience,
            "education": resume.education,
            "skills": resume.skills,
            "projects": resume.projects,
            "certifications": resume.certifications,
            "languages": resume.languages,
            "awards": resume.awards,
            "references": resume.references,
            "custom_sections": resume.custom_sections,
            "template_id": resume.template_id,
            "target_job_title": resume.target_job_title,
            "target_industry": resume.target_industry,
            "keywords": resume.keywords,
        }

    async def _create_version(
            self,
            db: AsyncSession,
            resume: Resume,
            description: str,
            user_id: uuid.UUID,
            old_data: Optional[Dict[str, Any]] = None
    ):
        """Create a new version of the resume."""
        # Get latest version number
        latest_version = await db.scalar(
            db.query(func.max(ResumeVersion.version_number))
            .filter(ResumeVersion.resume_id == resume.id)
        ) or 0

        version = ResumeVersion(
            resume_id=resume.id,
            version_number=latest_version + 1,
            change_description=description,
            resume_data=old_data or await self._serialize_resume_data(resume),
            created_by=user_id
        )

        db.add(version)
        await db.commit()

    def _has_significant_changes(self, update_fields: Dict[str, Any]) -> bool:
        """Check if update has significant changes requiring versioning."""
        significant_fields = {
            "work_experience", "education", "skills", "projects",
            "certifications", "personal_info", "title"
        }
        return any(field in update_fields for field in significant_fields)

    def _is_resume_complete(self, resume: Resume) -> bool:
        """Check if resume is complete enough to publish."""
        # Check required fields
        if not resume.personal_info.get("first_name") or not resume.personal_info.get("last_name"):
            return False

        if not resume.personal_info.get("email"):
            return False

        # Must have at least one work experience or education
        if not resume.work_experience and not resume.education:
            return False

        return True