"""
Cover letter service with business logic for cover letter operations.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_context
from app.core.redis import cache_manager
from app.models.cover_letter import (
    CoverLetter, CoverLetterTemplate, CoverLetterAnalysis,
    CoverLetterVersion, JobDescription, CoverLetterStatus,
    CoverLetterType, CoverLetterTone
)
from app.schemas.cover_letter import (
    CoverLetterCreate, CoverLetterUpdate, CoverLetterResponse,
    CoverLetterGenerationRequest, CoverLetterGenerationResponse,
    CoverLetterTemplateResponse, JobDescriptionCreate,
    CoverLetterOptimizationRequest, CoverLetterOptimizationResponse
)
from app.utils.exceptions import AppException
from app.services.ai_service import AIService
from app.services.pdf_service import PDFService
from app.tasks.resume_tasks import generate_cover_letter_pdf_task


class CoverLetterService:
    """Service for cover letter operations."""

    def __init__(self, ai_service: AIService = None, pdf_service: PDFService = None):
        self.ai_service = ai_service or AIService()
        self.pdf_service = pdf_service or PDFService()

    async def create_cover_letter(
            self,
            user_id: uuid.UUID,
            cover_letter_data: CoverLetterCreate
    ) -> CoverLetterResponse:
        """Create a new cover letter."""
        async with get_db_context() as db:
            # Validate template if provided
            if cover_letter_data.template_id:
                template = await self._get_template_by_id(db, cover_letter_data.template_id)
                if not template or not template.is_active:
                    raise AppException(
                        message="Invalid or inactive template",
                        status_code=400,
                        error_code="INVALID_TEMPLATE"
                    )

            # Validate resume if provided
            if cover_letter_data.resume_id:
                from app.models.resume import Resume
                resume = await db.get(Resume, cover_letter_data.resume_id)
                if not resume or resume.user_id != user_id:
                    raise AppException(
                        message="Invalid resume",
                        status_code=400,
                        error_code="INVALID_RESUME"
                    )

            # Create cover letter instance
            cover_letter = CoverLetter(
                user_id=user_id,
                resume_id=cover_letter_data.resume_id,
                title=cover_letter_data.title,
                description=cover_letter_data.description,
                type=cover_letter_data.type,
                tone=cover_letter_data.tone,
                content=cover_letter_data.content,
                template_id=cover_letter_data.template_id,
                job_title=cover_letter_data.job_title,
                company_name=cover_letter_data.company_name,
                job_description=cover_letter_data.job_description,
                job_requirements=cover_letter_data.job_requirements,
                company_info=cover_letter_data.company_info,
                hiring_manager_name=cover_letter_data.hiring_manager_name,
                hiring_manager_title=cover_letter_data.hiring_manager_title,
                company_address=cover_letter_data.company_address,
                keywords=cover_letter_data.keywords,
                key_achievements=cover_letter_data.key_achievements,
                skills_highlighted=cover_letter_data.skills_highlighted,
                ai_generated=cover_letter_data.ai_generated,
                ai_prompt=cover_letter_data.ai_prompt,
                generation_settings=cover_letter_data.generation_settings,
                personalization_level=cover_letter_data.personalization_level,
                is_public=cover_letter_data.is_public,
                status=CoverLetterStatus.DRAFT
            )

            db.add(cover_letter)
            await db.commit()
            await db.refresh(cover_letter)

            # Generate initial version
            await self._create_version(db, cover_letter, "Initial version", user_id)

            # Cache the cover letter
            await self._cache_cover_letter(cover_letter)

            return await self._to_response(cover_letter)

    async def generate_ai_cover_letter(
            self,
            user_id: uuid.UUID,
            generation_request: CoverLetterGenerationRequest
    ) -> CoverLetterGenerationResponse:
        """Generate a cover letter using AI."""
        import time
        start_time = time.time()

        async with get_db_context() as db:
            # Get resume data if provided
            resume_data = None
            if generation_request.resume_id:
                from app.models.resume import Resume
                resume = await db.get(Resume, generation_request.resume_id)
                if resume and resume.user_id == user_id:
                    resume_data = {
                        "work_experience": resume.work_experience,
                        "education": resume.education,
                        "skills": resume.skills,
                        "personal_info": resume.personal_info,
                    }

            # Get template if provided
            template = None
            if generation_request.template_id:
                template = await self._get_template_by_id(db, generation_request.template_id)

            # Generate content using AI service
            ai_response = await self.ai_service.generate_cover_letter(
                job_description=generation_request.job_description,
                job_title=generation_request.job_title,
                company_name=generation_request.company_name,
                hiring_manager_name=generation_request.hiring_manager_name,
                resume_data=resume_data,
                template=template,
                tone=generation_request.tone,
                personalization_level=generation_request.personalization_level,
                key_points=generation_request.key_points or [],
                company_research=generation_request.company_research or {},
                custom_prompt=generation_request.custom_prompt,
                generation_settings=generation_request.generation_settings or {}
            )

            # Create cover letter with generated content
            cover_letter_data = CoverLetterCreate(
                title=f"Cover Letter - {generation_request.company_name}",
                type=CoverLetterType.STANDARD,
                tone=generation_request.tone,
                content=ai_response["content"],
                job_title=generation_request.job_title,
                company_name=generation_request.company_name,
                hiring_manager_name=generation_request.hiring_manager_name,
                job_description=generation_request.job_description,
                resume_id=generation_request.resume_id,
                template_id=generation_request.template_id,
                ai_generated=True,
                ai_prompt=generation_request.custom_prompt or ai_response.get("prompt_used"),
                generation_settings=generation_request.generation_settings,
                personalization_level=generation_request.personalization_level,
                keywords=ai_response.get("keywords_used", []),
            )

            cover_letter = await self.create_cover_letter(user_id, cover_letter_data)

            generation_time = time.time() - start_time

            return CoverLetterGenerationResponse(
                cover_letter_id=cover_letter.id,
                content=ai_response["content"],
                opening_paragraph=ai_response.get("opening_paragraph", ""),
                body_paragraphs=ai_response.get("body_paragraphs", []),
                closing_paragraph=ai_response.get("closing_paragraph", ""),
                keywords_used=ai_response.get("keywords_used", []),
                personalization_score=ai_response.get("personalization_score", 85.0),
                generation_time=generation_time,
                ai_model_used=ai_response.get("model_used", "gpt-4")
            )

    async def get_cover_letter(
            self,
            cover_letter_id: uuid.UUID,
            user_id: uuid.UUID
    ) -> CoverLetterResponse:
        """Get a cover letter by ID."""
        # Try cache first
        cache_key = f"cover_letter:{cover_letter_id}"
        cached_cover_letter = await cache_manager.get(cache_key) if cache_manager else None

        if cached_cover_letter:
            # Verify user access
            if cached_cover_letter.get("user_id") != str(user_id):
                raise AppException(
                    message="Cover letter not found",
                    status_code=404,
                    error_code="COVER_LETTER_NOT_FOUND"
                )
            return CoverLetterResponse(**cached_cover_letter)

        async with get_db_context() as db:
            cover_letter = await db.get(CoverLetter, cover_letter_id)

            if not cover_letter:
                raise AppException(
                    message="Cover letter not found",
                    status_code=404,
                    error_code="COVER_LETTER_NOT_FOUND"
                )

            if cover_letter.user_id != user_id and not cover_letter.is_public:
                raise AppException(
                    message="Cover letter not found",
                    status_code=404,
                    error_code="COVER_LETTER_NOT_FOUND"
                )

            # Increment view count
            cover_letter.view_count += 1
            await db.commit()

            # Cache the cover letter
            await self._cache_cover_letter(cover_letter)

            return await self._to_response(cover_letter)

    async def update_cover_letter(
            self,
            cover_letter_id: uuid.UUID,
            user_id: uuid.UUID,
            update_data: CoverLetterUpdate
    ) -> CoverLetterResponse:
        """Update a cover letter."""
        async with get_db_context() as db:
            cover_letter = await db.get(CoverLetter, cover_letter_id)

            if not cover_letter or cover_letter.user_id != user_id:
                raise AppException(
                    message="Cover letter not found",
                    status_code=404,
                    error_code="COVER_LETTER_NOT_FOUND"
                )

            # Store old data for version tracking
            old_data = await self._serialize_cover_letter_data(cover_letter)

            # Update fields
            update_fields = update_data.model_dump(exclude_unset=True)
            for field, value in update_fields.items():
                setattr(cover_letter, field, value)

            cover_letter.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(cover_letter)

            # Create new version if significant changes
            if self._has_significant_changes(update_fields):
                await self._create_version(
                    db, cover_letter, "Updated cover letter", user_id, old_data
                )

            # Clear cache
            await self._clear_cover_letter_cache(cover_letter_id)

            return await self._to_response(cover_letter)

    async def delete_cover_letter(
            self,
            cover_letter_id: uuid.UUID,
            user_id: uuid.UUID
    ) -> bool:
        """Delete a cover letter (soft delete)."""
        async with get_db_context() as db:
            cover_letter = await db.get(CoverLetter, cover_letter_id)

            if not cover_letter or cover_letter.user_id != user_id:
                raise AppException(
                    message="Cover letter not found",
                    status_code=404,
                    error_code="COVER_LETTER_NOT_FOUND"
                )

            cover_letter.status = CoverLetterStatus.DELETED
            cover_letter.deleted_at = datetime.utcnow()
            await db.commit()

            # Clear cache
            await self._clear_cover_letter_cache(cover_letter_id)

            return True

    async def list_cover_letters(
            self,
            user_id: uuid.UUID,
            page: int = 1,
            size: int = 20,
            status: Optional[CoverLetterStatus] = None,
            type: Optional[CoverLetterType] = None,
            company_name: Optional[str] = None,
            search: Optional[str] = None,
            sort_by: str = "updated_at",
            sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """List user's cover letters with pagination."""
        async with get_db_context() as db:
            # Build query
            query = db.query(CoverLetter).filter(
                CoverLetter.user_id == user_id,
                CoverLetter.deleted_at.is_(None)
            )

            # Add filters
            if status:
                query = query.filter(CoverLetter.status == status)

            if type:
                query = query.filter(CoverLetter.type == type)

            if company_name:
                query = query.filter(CoverLetter.company_name.ilike(f"%{company_name}%"))

            if search:
                search_filter = or_(
                    CoverLetter.title.ilike(f"%{search}%"),
                    CoverLetter.description.ilike(f"%{search}%"),
                    CoverLetter.job_title.ilike(f"%{search}%"),
                    CoverLetter.company_name.ilike(f"%{search}%")
                )
                query = query.filter(search_filter)

            # Add sorting
            if sort_order == "desc":
                query = query.order_by(desc(getattr(CoverLetter, sort_by)))
            else:
                query = query.order_by(getattr(CoverLetter, sort_by))

            # Get total count
            total = await db.scalar(
                query.with_only_columns(func.count(CoverLetter.id))
            )

            # Apply pagination
            offset = (page - 1) * size
            cover_letters = await db.scalars(
                query.offset(offset).limit(size)
            )

            # Convert to response
            cover_letter_responses = []
            for cover_letter in cover_letters:
                cover_letter_responses.append(await self._to_response(cover_letter))

            return {
                "cover_letters": cover_letter_responses,
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            }

    async def optimize_cover_letter(
            self,
            cover_letter_id: uuid.UUID,
            user_id: uuid.UUID,
            optimization_request: CoverLetterOptimizationRequest
    ) -> CoverLetterOptimizationResponse:
        """Optimize cover letter content."""
        async with get_db_context() as db:
            cover_letter = await db.get(CoverLetter, cover_letter_id)

            if not cover_letter or cover_letter.user_id != user_id:
                raise AppException(
                    message="Cover letter not found",
                    status_code=404,
                    error_code="COVER_LETTER_NOT_FOUND"
                )

            # Get current scores for comparison
            before_scores = await self._analyze_cover_letter_quality(cover_letter)

            # Optimize using AI service
            optimization_result = await self.ai_service.optimize_cover_letter(
                content=cover_letter.content,
                optimization_type=optimization_request.optimization_type,
                job_description=optimization_request.job_description,
                target_keywords=optimization_request.target_keywords,
                target_tone=optimization_request.target_tone,
                target_length=optimization_request.target_length,
                preserve_sections=optimization_request.preserve_sections,
                optimization_level=optimization_request.optimization_level
            )

            # Calculate predicted scores
            after_scores = await self._predict_optimized_scores(
                before_scores, optimization_result
            )

            return CoverLetterOptimizationResponse(
                optimized_content=optimization_result["content"],
                changes_made=optimization_result["changes"],
                improvement_score=optimization_result["improvement_score"],
                optimization_summary=optimization_result["summary"],
                before_scores=before_scores,
                after_scores=after_scores
            )

    async def duplicate_cover_letter(
            self,
            cover_letter_id: uuid.UUID,
            user_id: uuid.UUID,
            new_title: Optional[str] = None,
            new_company: Optional[str] = None,
            new_job_title: Optional[str] = None
    ) -> CoverLetterResponse:
        """Duplicate an existing cover letter."""
        async with get_db_context() as db:
            original = await db.get(CoverLetter, cover_letter_id)

            if not original or original.user_id != user_id:
                raise AppException(
                    message="Cover letter not found",
                    status_code=404,
                    error_code="COVER_LETTER_NOT_FOUND"
                )

            # Create duplicate
            duplicate_data = await self._serialize_cover_letter_data(original)
            duplicate_data["title"] = new_title or f"{original.title} (Copy)"
            duplicate_data["company_name"] = new_company or original.company_name
            duplicate_data["job_title"] = new_job_title or original.job_title
            duplicate_data["status"] = CoverLetterStatus.DRAFT
            duplicate_data["is_public"] = False
            duplicate_data["share_token"] = None

            # Remove computed fields
            for field in ["id", "created_at", "updated_at", "view_count",
                          "download_count", "share_count", "file_path", "pdf_path"]:
                duplicate_data.pop(field, None)

            duplicate = CoverLetter(**duplicate_data)
            db.add(duplicate)
            await db.commit()
            await db.refresh(duplicate)

            return await self._to_response(duplicate)

    async def get_templates(
            self,
            type: Optional[CoverLetterType] = None,
            tone: Optional[CoverLetterTone] = None,
            industry: Optional[str] = None,
            is_premium: Optional[bool] = None,
            page: int = 1,
            size: int = 20
    ) -> List[CoverLetterTemplateResponse]:
        """Get available cover letter templates."""
        cache_key = f"cl_templates:{type}:{tone}:{industry}:{is_premium}:{page}:{size}"
        cached_templates = await cache_manager.get(cache_key) if cache_manager else None

        if cached_templates:
            return [CoverLetterTemplateResponse(**t) for t in cached_templates]

        async with get_db_context() as db:
            query = db.query(CoverLetterTemplate).filter(
                CoverLetterTemplate.is_active == True
            )

            if type:
                query = query.filter(CoverLetterTemplate.type == type)

            if tone:
                query = query.filter(CoverLetterTemplate.tone == tone)

            if industry:
                query = query.filter(
                    CoverLetterTemplate.target_industries.contains([industry])
                )

            if is_premium is not None:
                query = query.filter(CoverLetterTemplate.is_premium == is_premium)

            query = query.order_by(
                CoverLetterTemplate.sort_order,
                CoverLetterTemplate.created_at
            )

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

    async def save_job_description(
            self,
            user_id: uuid.UUID,
            job_data: JobDescriptionCreate
    ) -> Dict[str, Any]:
        """Save a job description for future use."""
        async with get_db_context() as db:
            job_description = JobDescription(
                user_id=user_id,
                title=job_data.title,
                company_name=job_data.company_name,
                description=job_data.description,
                requirements=job_data.requirements,
                responsibilities=job_data.responsibilities,
                location=job_data.location,
                job_type=job_data.job_type,
                experience_level=job_data.experience_level,
                industry=job_data.industry,
                department=job_data.department,
                salary_min=job_data.salary_min,
                salary_max=job_data.salary_max,
                currency=job_data.currency,
                external_url=job_data.external_url,
                external_id=job_data.external_id,
                source=job_data.source
            )

            # Extract keywords using AI service
            if self.ai_service:
                keywords = await self.ai_service.extract_job_keywords(job_data.description)
                job_description.extracted_keywords = keywords.get("keywords", [])
                job_description.skill_requirements = keywords.get("skills", [])
                job_description.education_requirements = keywords.get("education", [])

            db.add(job_description)
            await db.commit()
            await db.refresh(job_description)

            return {
                "id": job_description.id,
                "title": job_description.title,
                "company_name": job_description.company_name,
                "extracted_keywords": job_description.extracted_keywords,
                "skill_requirements": job_description.skill_requirements,
                "education_requirements": job_description.education_requirements,
                "created_at": job_description.created_at
            }

    # Helper methods
    async def _get_template_by_id(
            self,
            db: AsyncSession,
            template_id: uuid.UUID
    ) -> Optional[CoverLetterTemplate]:
        """Get template by ID."""
        return await db.get(CoverLetterTemplate, template_id)

    async def _to_response(self, cover_letter: CoverLetter) -> CoverLetterResponse:
        """Convert cover letter model to response."""
        return CoverLetterResponse(
            id=cover_letter.id,
            user_id=cover_letter.user_id,
            resume_id=cover_letter.resume_id,
            title=cover_letter.title,
            description=cover_letter.description,
            status=cover_letter.status,
            type=cover_letter.type,
            tone=cover_letter.tone,
            content=cover_letter.content,
            template_id=cover_letter.template_id,
            opening_paragraph=cover_letter.opening_paragraph,
            body_paragraphs=cover_letter.body_paragraphs,
            closing_paragraph=cover_letter.closing_paragraph,
            job_title=cover_letter.job_title,
            company_name=cover_letter.company_name,
            company_info=cover_letter.company_info,
            job_description=cover_letter.job_description,
            job_requirements=cover_letter.job_requirements,
            hiring_manager_name=cover_letter.hiring_manager_name,
            hiring_manager_title=cover_letter.hiring_manager_title,
            company_address=cover_letter.company_address,
            custom_css=cover_letter.custom_css,
            theme_settings=cover_letter.theme_settings,
            ai_generated=cover_letter.ai_generated,
            ai_prompt=cover_letter.ai_prompt,
            ai_model_used=cover_letter.ai_model_used,
            generation_settings=cover_letter.generation_settings,
            personalization_level=cover_letter.personalization_level,
            keywords=cover_letter.keywords,
            key_achievements=cover_letter.key_achievements,
            skills_highlighted=cover_letter.skills_highlighted,
            file_path=cover_letter.file_path,
            pdf_path=cover_letter.pdf_path,
            file_size=cover_letter.file_size,
            view_count=cover_letter.view_count,
            download_count=cover_letter.download_count,
            share_count=cover_letter.share_count,
            content_quality_score=cover_letter.content_quality_score,
            personalization_score=cover_letter.personalization_score,
            keyword_match_score=cover_letter.keyword_match_score,
            overall_score=cover_letter.overall_score,
            last_analyzed_at=cover_letter.last_analyzed_at,
            is_public=cover_letter.is_public,
            is_featured=cover_letter.is_featured,
            share_token=cover_letter.share_token,
            created_at=cover_letter.created_at,
            updated_at=cover_letter.updated_at,
            deleted_at=cover_letter.deleted_at,
        )

    async def _template_to_response(self, template: CoverLetterTemplate) -> CoverLetterTemplateResponse:
        """Convert template model to response."""
        return CoverLetterTemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            type=template.type,
            tone=template.tone,
            html_template=template.html_template,
            css_styles=template.css_styles,
            opening_template=template.opening_template,
            body_template=template.body_template,
            closing_template=template.closing_template,
            placeholders=template.placeholders,
            required_fields=template.required_fields,
            optional_fields=template.optional_fields,
            is_premium=template.is_premium,
            tags=template.tags,
            preview_image=template.preview_image,
            ai_prompt_template=template.ai_prompt_template,
            generation_settings=template.generation_settings,
            target_industries=template.target_industries,
            target_job_levels=template.target_job_levels,
            target_job_types=template.target_job_types,
            is_active=template.is_active,
            is_featured=template.is_featured,
            usage_count=template.usage_count,
            rating=template.rating,
            sort_order=template.sort_order,
            created_by=template.created_by,
            created_at=template.created_at,
            updated_at=template.updated_at,
        )

    async def _cache_cover_letter(self, cover_letter: CoverLetter):
        """Cache cover letter data."""
        if cache_manager:
            cache_key = f"cover_letter:{cover_letter.id}"
            cover_letter_data = (await self._to_response(cover_letter)).model_dump()
            await cache_manager.set(cache_key, cover_letter_data, ttl=1800)  # 30 minutes

    async def _clear_cover_letter_cache(self, cover_letter_id: uuid.UUID):
        """Clear cover letter cache."""
        if cache_manager:
            cache_key = f"cover_letter:{cover_letter_id}"
            await cache_manager.delete(cache_key)

    async def _serialize_cover_letter_data(self, cover_letter: CoverLetter) -> Dict[str, Any]:
        """Serialize cover letter data for versioning."""
        return {
            "title": cover_letter.title,
            "description": cover_letter.description,
            "type": cover_letter.type,
            "tone": cover_letter.tone,
            "content": cover_letter.content,
            "opening_paragraph": cover_letter.opening_paragraph,
            "body_paragraphs": cover_letter.body_paragraphs,
            "closing_paragraph": cover_letter.closing_paragraph,
            "job_title": cover_letter.job_title,
            "company_name": cover_letter.company_name,
            "company_info": cover_letter.company_info,
            "job_description": cover_letter.job_description,
            "hiring_manager_name": cover_letter.hiring_manager_name,
            "keywords": cover_letter.keywords,
            "template_id": cover_letter.template_id,
        }

    async def _create_version(
            self,
            db: AsyncSession,
            cover_letter: CoverLetter,
            description: str,
            user_id: uuid.UUID,
            old_data: Optional[Dict[str, Any]] = None
    ):
        """Create a new version of the cover letter."""
        # Get latest version number
        latest_version = await db.scalar(
            db.query(func.max(CoverLetterVersion.version_number))
            .filter(CoverLetterVersion.cover_letter_id == cover_letter.id)
        ) or 0

        version = CoverLetterVersion(
            cover_letter_id=cover_letter.id,
            version_number=latest_version + 1,
            change_description=description,
            cover_letter_data=old_data or await self._serialize_cover_letter_data(cover_letter),
            created_by=user_id
        )

        db.add(version)
        await db.commit()

    def _has_significant_changes(self, update_fields: Dict[str, Any]) -> bool:
        """Check if update has significant changes requiring versioning."""
        significant_fields = {
            "content", "opening_paragraph", "body_paragraphs",
            "closing_paragraph", "title", "tone", "type"
        }
        return any(field in update_fields for field in significant_fields)

    async def _analyze_cover_letter_quality(self, cover_letter: CoverLetter) -> Dict[str, float]:
        """Analyze cover letter quality and return scores."""
        if self.ai_service:
            return await self.ai_service.analyze_cover_letter_quality(cover_letter.content)

        # Fallback basic analysis
        return {
            "content_quality": 75.0,
            "personalization": 70.0,
            "keyword_match": 65.0,
            "tone_consistency": 80.0,
            "overall": 72.5
        }

    async def _predict_optimized_scores(
            self,
            before_scores: Dict[str, float],
            optimization_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict scores after optimization."""
        improvement = optimization_result.get("improvement_score", 10.0)

        after_scores = {}
        for metric, score in before_scores.items():
            # Apply improvement with diminishing returns
            max_improvement = min(improvement, 100 - score)
            after_scores[metric] = min(100, score + max_improvement * 0.8)

        return after_scores