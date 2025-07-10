"""
Template management service for resume and cover letter templates.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_context
from app.core.redis import cache_manager
from app.models.resume import ResumeTemplate, TemplateCategory
from app.models.cover_letter import CoverLetterTemplate, CoverLetterType, CoverLetterTone
from app.schemas.resume import (
    ResumeTemplateCreate, ResumeTemplateUpdate, ResumeTemplateResponse
)
from app.schemas.cover_letter import (
    CoverLetterTemplateCreate, CoverLetterTemplateUpdate, CoverLetterTemplateResponse
)
from app.utils.exceptions import AppException
from app.utils.formatters import TextFormatter


class TemplateService:
    """Service for managing resume and cover letter templates."""

    def __init__(self):
        self.cache_ttl = 3600  # 1 hour

    # Resume Template Methods
    async def create_resume_template(
            self,
            template_data: ResumeTemplateCreate,
            created_by: Optional[uuid.UUID] = None
    ) -> ResumeTemplateResponse:
        """Create a new resume template."""
        async with get_db_context() as db:
            # Validate template content
            await self._validate_template_content(
                template_data.html_template,
                template_data.css_styles,
                "resume"
            )

            template = ResumeTemplate(
                name=template_data.name,
                description=template_data.description,
                category=template_data.category,
                html_template=template_data.html_template,
                css_styles=template_data.css_styles,
                preview_image=template_data.preview_image,
                default_settings=template_data.default_settings,
                customizable_fields=template_data.customizable_fields,
                supported_sections=template_data.supported_sections,
                is_premium=template_data.is_premium,
                tags=template_data.tags,
                sort_order=template_data.sort_order,
                created_by=created_by
            )

            db.add(template)
            await db.commit()
            await db.refresh(template)

            # Clear template cache
            await self._clear_template_cache("resume")

            return await self._resume_template_to_response(template)

    async def update_resume_template(
            self,
            template_id: uuid.UUID,
            template_data: ResumeTemplateUpdate,
            user_id: Optional[uuid.UUID] = None
    ) -> ResumeTemplateResponse:
        """Update a resume template."""
        async with get_db_context() as db:
            template = await db.get(ResumeTemplate, template_id)

            if not template:
                raise AppException(
                    message="Template not found",
                    status_code=404,
                    error_code="TEMPLATE_NOT_FOUND"
                )

            # Check permissions (only creator or admin can update)
            if user_id and template.created_by and template.created_by != user_id:
                raise AppException(
                    message="Insufficient permissions to update template",
                    status_code=403,
                    error_code="INSUFFICIENT_PERMISSIONS"
                )

            # Update fields
            update_fields = template_data.model_dump(exclude_unset=True)
            for field, value in update_fields.items():
                setattr(template, field, value)

            # Validate updated content if HTML/CSS changed
            if "html_template" in update_fields or "css_styles" in update_fields:
                await self._validate_template_content(
                    template.html_template,
                    template.css_styles,
                    "resume"
                )

            template.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(template)

            # Clear template cache
            await self._clear_template_cache("resume")

            return await self._resume_template_to_response(template)

    async def get_resume_templates(
            self,
            category: Optional[TemplateCategory] = None,
            is_premium: Optional[bool] = None,
            is_active: bool = True,
            tags: Optional[List[str]] = None,
            search: Optional[str] = None,
            page: int = 1,
            size: int = 20
    ) -> Dict[str, Any]:
        """Get resume templates with filtering and pagination."""
        cache_key = f"resume_templates:{category}:{is_premium}:{is_active}:{page}:{size}:{search}"
        if tags:
            cache_key += f":{','.join(sorted(tags))}"

        # Try cache first
        cached_result = await cache_manager.get(cache_key) if cache_manager else None
        if cached_result:
            return cached_result

        async with get_db_context() as db:
            query = db.query(ResumeTemplate)

            # Apply filters
            if is_active is not None:
                query = query.filter(ResumeTemplate.is_active == is_active)

            if category:
                query = query.filter(ResumeTemplate.category == category)

            if is_premium is not None:
                query = query.filter(ResumeTemplate.is_premium == is_premium)

            if tags:
                # Filter by tags (templates that have any of the specified tags)
                tag_filters = []
                for tag in tags:
                    tag_filters.append(ResumeTemplate.tags.contains([tag]))
                query = query.filter(or_(*tag_filters))

            if search:
                search_filter = or_(
                    ResumeTemplate.name.ilike(f"%{search}%"),
                    ResumeTemplate.description.ilike(f"%{search}%")
                )
                query = query.filter(search_filter)

            # Order by sort_order and featured status
            query = query.order_by(
                ResumeTemplate.is_featured.desc(),
                ResumeTemplate.sort_order,
                ResumeTemplate.usage_count.desc(),
                ResumeTemplate.created_at
            )

            # Get total count
            total = await db.scalar(
                query.with_only_columns(func.count(ResumeTemplate.id))
            )

            # Apply pagination
            offset = (page - 1) * size
            templates = await db.scalars(query.offset(offset).limit(size))

            # Convert to response
            template_responses = []
            for template in templates:
                template_responses.append(await self._resume_template_to_response(template))

            result = {
                "templates": template_responses,
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            }

            # Cache result
            if cache_manager:
                await cache_manager.set(cache_key, result, ttl=self.cache_ttl)

            return result

    async def get_resume_template(self, template_id: uuid.UUID) -> ResumeTemplateResponse:
        """Get a specific resume template."""
        cache_key = f"resume_template:{template_id}"
        cached_template = await cache_manager.get(cache_key) if cache_manager else None

        if cached_template:
            return ResumeTemplateResponse(**cached_template)

        async with get_db_context() as db:
            template = await db.get(ResumeTemplate, template_id)

            if not template:
                raise AppException(
                    message="Template not found",
                    status_code=404,
                    error_code="TEMPLATE_NOT_FOUND"
                )

            response = await self._resume_template_to_response(template)

            # Cache template
            if cache_manager:
                await cache_manager.set(cache_key, response.model_dump(), ttl=self.cache_ttl)

            return response

    # Cover Letter Template Methods
    async def create_cover_letter_template(
            self,
            template_data: CoverLetterTemplateCreate,
            created_by: Optional[uuid.UUID] = None
    ) -> CoverLetterTemplateResponse:
        """Create a new cover letter template."""
        async with get_db_context() as db:
            # Validate template content
            await self._validate_template_content(
                template_data.html_template,
                template_data.css_styles,
                "cover_letter"
            )

            template = CoverLetterTemplate(
                name=template_data.name,
                description=template_data.description,
                type=template_data.type,
                tone=template_data.tone,
                html_template=template_data.html_template,
                css_styles=template_data.css_styles,
                preview_image=template_data.preview_image,
                opening_template=template_data.opening_template,
                body_template=template_data.body_template,
                closing_template=template_data.closing_template,
                ai_prompt_template=template_data.ai_prompt_template,
                generation_settings=template_data.generation_settings,
                placeholders=template_data.placeholders,
                required_fields=template_data.required_fields,
                optional_fields=template_data.optional_fields,
                target_industries=template_data.target_industries,
                target_job_levels=template_data.target_job_levels,
                target_job_types=template_data.target_job_types,
                is_premium=template_data.is_premium,
                tags=template_data.tags,
                sort_order=template_data.sort_order,
                created_by=created_by
            )

            db.add(template)
            await db.commit()
            await db.refresh(template)

            # Clear template cache
            await self._clear_template_cache("cover_letter")

            return await self._cover_letter_template_to_response(template)

    async def get_cover_letter_templates(
            self,
            type: Optional[CoverLetterType] = None,
            tone: Optional[CoverLetterTone] = None,
            industry: Optional[str] = None,
            job_level: Optional[str] = None,
            is_premium: Optional[bool] = None,
            is_active: bool = True,
            tags: Optional[List[str]] = None,
            search: Optional[str] = None,
            page: int = 1,
            size: int = 20
    ) -> Dict[str, Any]:
        """Get cover letter templates with filtering and pagination."""
        cache_key = f"cover_letter_templates:{type}:{tone}:{industry}:{job_level}:{is_premium}:{is_active}:{page}:{size}:{search}"
        if tags:
            cache_key += f":{','.join(sorted(tags))}"

        # Try cache first
        cached_result = await cache_manager.get(cache_key) if cache_manager else None
        if cached_result:
            return cached_result

        async with get_db_context() as db:
            query = db.query(CoverLetterTemplate)

            # Apply filters
            if is_active is not None:
                query = query.filter(CoverLetterTemplate.is_active == is_active)

            if type:
                query = query.filter(CoverLetterTemplate.type == type)

            if tone:
                query = query.filter(CoverLetterTemplate.tone == tone)

            if industry:
                query = query.filter(
                    CoverLetterTemplate.target_industries.contains([industry])
                )

            if job_level:
                query = query.filter(
                    CoverLetterTemplate.target_job_levels.contains([job_level])
                )

            if is_premium is not None:
                query = query.filter(CoverLetterTemplate.is_premium == is_premium)

            if tags:
                tag_filters = []
                for tag in tags:
                    tag_filters.append(CoverLetterTemplate.tags.contains([tag]))
                query = query.filter(or_(*tag_filters))

            if search:
                search_filter = or_(
                    CoverLetterTemplate.name.ilike(f"%{search}%"),
                    CoverLetterTemplate.description.ilike(f"%{search}%")
                )
                query = query.filter(search_filter)

            # Order by sort_order and featured status
            query = query.order_by(
                CoverLetterTemplate.is_featured.desc(),
                CoverLetterTemplate.sort_order,
                CoverLetterTemplate.usage_count.desc(),
                CoverLetterTemplate.created_at
            )

            # Get total count
            total = await db.scalar(
                query.with_only_columns(func.count(CoverLetterTemplate.id))
            )

            # Apply pagination
            offset = (page - 1) * size
            templates = await db.scalars(query.offset(offset).limit(size))

            # Convert to response
            template_responses = []
            for template in templates:
                template_responses.append(await self._cover_letter_template_to_response(template))

            result = {
                "templates": template_responses,
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            }

            # Cache result
            if cache_manager:
                await cache_manager.set(cache_key, result, ttl=self.cache_ttl)

            return result

    async def get_cover_letter_template(self, template_id: uuid.UUID) -> CoverLetterTemplateResponse:
        """Get a specific cover letter template."""
        cache_key = f"cover_letter_template:{template_id}"
        cached_template = await cache_manager.get(cache_key) if cache_manager else None

        if cached_template:
            return CoverLetterTemplateResponse(**cached_template)

        async with get_db_context() as db:
            template = await db.get(CoverLetterTemplate, template_id)

            if not template:
                raise AppException(
                    message="Template not found",
                    status_code=404,
                    error_code="TEMPLATE_NOT_FOUND"
                )

            response = await self._cover_letter_template_to_response(template)

            # Cache template
            if cache_manager:
                await cache_manager.set(cache_key, response.model_dump(), ttl=self.cache_ttl)

            return response

    # Template Analytics Methods
    async def increment_template_usage(self, template_id: uuid.UUID, template_type: str):
        """Increment template usage counter."""
        async with get_db_context() as db:
            if template_type == "resume":
                template = await db.get(ResumeTemplate, template_id)
            else:
                template = await db.get(CoverLetterTemplate, template_id)

            if template:
                template.usage_count += 1
                await db.commit()

                # Clear relevant caches
                await self._clear_template_cache(template_type)

    async def get_template_analytics(self, template_type: str = None) -> Dict[str, Any]:
        """Get template usage analytics."""
        async with get_db_context() as db:
            analytics = {}

            if template_type in [None, "resume"]:
                # Resume template analytics
                resume_stats = await db.execute("""
                                                SELECT COUNT(*)         as total_templates,
                                                       COUNT(*)            FILTER (WHERE is_active = true) as active_templates, COUNT(*) FILTER (WHERE is_premium = true) as premium_templates, AVG(usage_count) as avg_usage,
                                                       MAX(usage_count) as max_usage
                                                FROM resume_templates
                                                """)
                resume_row = resume_stats.fetchone()

                analytics["resume_templates"] = {
                    "total": resume_row.total_templates,
                    "active": resume_row.active_templates,
                    "premium": resume_row.premium_templates,
                    "average_usage": float(resume_row.avg_usage or 0),
                    "max_usage": resume_row.max_usage or 0
                }

            if template_type in [None, "cover_letter"]:
                # Cover letter template analytics
                cover_letter_stats = await db.execute("""
                                                      SELECT COUNT(*)         as total_templates,
                                                             COUNT(*)            FILTER (WHERE is_active = true) as active_templates, COUNT(*) FILTER (WHERE is_premium = true) as premium_templates, AVG(usage_count) as avg_usage,
                                                             MAX(usage_count) as max_usage
                                                      FROM cover_letter_templates
                                                      """)
                cover_letter_row = cover_letter_stats.fetchone()

                analytics["cover_letter_templates"] = {
                    "total": cover_letter_row.total_templates,
                    "active": cover_letter_row.active_templates,
                    "premium": cover_letter_row.premium_templates,
                    "average_usage": float(cover_letter_row.avg_usage or 0),
                    "max_usage": cover_letter_row.max_usage or 0
                }

            return analytics

    async def get_popular_templates(
            self,
            template_type: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most popular templates by usage."""
        async with get_db_context() as db:
            if template_type == "resume":
                templates = await db.scalars(
                    db.query(ResumeTemplate)
                    .filter(ResumeTemplate.is_active == True)
                    .order_by(ResumeTemplate.usage_count.desc())
                    .limit(limit)
                )
                return [
                    {
                        "id": str(t.id),
                        "name": t.name,
                        "category": t.category.value,
                        "usage_count": t.usage_count,
                        "is_premium": t.is_premium
                    }
                    for t in templates
                ]
            else:
                templates = await db.scalars(
                    db.query(CoverLetterTemplate)
                    .filter(CoverLetterTemplate.is_active == True)
                    .order_by(CoverLetterTemplate.usage_count.desc())
                    .limit(limit)
                )
                return [
                    {
                        "id": str(t.id),
                        "name": t.name,
                        "type": t.type.value,
                        "tone": t.tone.value,
                        "usage_count": t.usage_count,
                        "is_premium": t.is_premium
                    }
                    for t in templates
                ]

    # Template Suggestion Methods
    async def suggest_resume_templates(
            self,
            user_preferences: Dict[str, Any],
            job_title: Optional[str] = None,
            industry: Optional[str] = None,
            experience_level: Optional[str] = None,
            limit: int = 5
    ) -> List[ResumeTemplateResponse]:
        """Suggest resume templates based on user preferences and job details."""
        async with get_db_context() as db:
            query = db.query(ResumeTemplate).filter(ResumeTemplate.is_active == True)

            # Score templates based on various factors
            suggestions = []
            templates = await db.scalars(query)

            for template in templates:
                score = await self._calculate_resume_template_score(
                    template, user_preferences, job_title, industry, experience_level
                )
                suggestions.append((template, score))

            # Sort by score and return top suggestions
            suggestions.sort(key=lambda x: x[1], reverse=True)
            top_suggestions = suggestions[:limit]

            return [
                await self._resume_template_to_response(template)
                for template, score in top_suggestions
            ]

    async def suggest_cover_letter_templates(
            self,
            job_description: str,
            job_title: str,
            company_name: str,
            industry: Optional[str] = None,
            experience_level: Optional[str] = None,
            preferred_tone: Optional[CoverLetterTone] = None,
            limit: int = 5
    ) -> List[CoverLetterTemplateResponse]:
        """Suggest cover letter templates based on job details."""
        async with get_db_context() as db:
            query = db.query(CoverLetterTemplate).filter(CoverLetterTemplate.is_active == True)

            suggestions = []
            templates = await db.scalars(query)

            for template in templates:
                score = await self._calculate_cover_letter_template_score(
                    template, job_description, job_title, company_name,
                    industry, experience_level, preferred_tone
                )
                suggestions.append((template, score))

            # Sort by score and return top suggestions
            suggestions.sort(key=lambda x: x[1], reverse=True)
            top_suggestions = suggestions[:limit]

            return [
                await self._cover_letter_template_to_response(template)
                for template, score in top_suggestions
            ]

    # Helper Methods
    async def _validate_template_content(
            self,
            html_template: str,
            css_styles: str,
            template_type: str
    ):
        """Validate template HTML and CSS content."""
        # Basic validation - check for dangerous content
        dangerous_patterns = [
            '<script', 'javascript:', 'vbscript:', 'data:', 'onload=',
            'onclick=', 'onerror=', 'onmouseover=', 'eval(', 'expression('
        ]

        content_to_check = html_template.lower() + css_styles.lower()

        for pattern in dangerous_patterns:
            if pattern in content_to_check:
                raise AppException(
                    message=f"Template contains potentially dangerous content: {pattern}",
                    status_code=400,
                    error_code="INVALID_TEMPLATE_CONTENT"
                )

        # Check for required placeholders based on template type
        if template_type == "resume":
            required_placeholders = ["{{personal_info.full_name}}", "{{personal_info.email}}"]
        else:
            required_placeholders = ["{{job_title}}", "{{company_name}}", "{{opening_paragraph}}"]

        missing_placeholders = []
        for placeholder in required_placeholders:
            if placeholder not in html_template:
                missing_placeholders.append(placeholder)

        if missing_placeholders:
            raise AppException(
                message=f"Template missing required placeholders: {', '.join(missing_placeholders)}",
                status_code=400,
                error_code="MISSING_TEMPLATE_PLACEHOLDERS"
            )

    async def _resume_template_to_response(self, template: ResumeTemplate) -> ResumeTemplateResponse:
        """Convert resume template model to response."""
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
            updated_at=template.updated_at
        )

    async def _cover_letter_template_to_response(self, template: CoverLetterTemplate) -> CoverLetterTemplateResponse:
        """Convert cover letter template model to response."""
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
            updated_at=template.updated_at
        )

    async def _calculate_resume_template_score(
            self,
            template: ResumeTemplate,
            user_preferences: Dict[str, Any],
            job_title: Optional[str],
            industry: Optional[str],
            experience_level: Optional[str]
    ) -> float:
        """Calculate template relevance score for resume templates."""
        score = 0.0

        # Base score from usage and rating
        score += min(template.usage_count / 100, 10)  # Max 10 points from usage
        if template.rating:
            score += template.rating * 2  # Max 10 points from rating

        # Category preferences
        preferred_category = user_preferences.get("preferred_category")
        if preferred_category and template.category.value == preferred_category:
            score += 15

        # Premium preference
        if user_preferences.get("premium_only") and template.is_premium:
            score += 5
        elif not user_preferences.get("premium_only") and not template.is_premium:
            score += 2

        # Industry matching
        if industry and template.tags:
            industry_lower = industry.lower()
            for tag in template.tags:
                if industry_lower in tag.lower():
                    score += 10
                    break

        # Featured templates get bonus
        if template.is_featured:
            score += 5

        return score

    async def _calculate_cover_letter_template_score(
            self,
            template: CoverLetterTemplate,
            job_description: str,
            job_title: str,
            company_name: str,
            industry: Optional[str],
            experience_level: Optional[str],
            preferred_tone: Optional[CoverLetterTone]
    ) -> float:
        """Calculate template relevance score for cover letter templates."""
        score = 0.0

        # Base score from usage and rating
        score += min(template.usage_count / 100, 10)
        if template.rating:
            score += template.rating * 2

        # Tone matching
        if preferred_tone and template.tone == preferred_tone:
            score += 15

        # Industry matching
        if industry and template.target_industries:
            if industry in template.target_industries:
                score += 20

        # Job level matching
        if experience_level and template.target_job_levels:
            if experience_level in template.target_job_levels:
                score += 15

        # Type matching based on job description keywords
        job_desc_lower = job_description.lower()
        if template.type == CoverLetterType.NETWORKING and any(
                word in job_desc_lower for word in ["referral", "network", "connection", "introduction"]
        ):
            score += 10
        elif template.type == CoverLetterType.FOLLOW_UP and any(
                word in job_desc_lower for word in ["follow", "interview", "meeting", "discussion"]
        ):
            score += 10

        # Featured templates get bonus
        if template.is_featured:
            score += 5

        return score

    async def _clear_template_cache(self, template_type: str):
        """Clear template cache for a specific type."""
        if cache_manager:
            pattern = f"{template_type}_template*"
            await cache_manager.delete_pattern(pattern)