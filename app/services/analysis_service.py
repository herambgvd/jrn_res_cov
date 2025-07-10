"""
Analysis service for AI-powered resume and cover letter analysis.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_context
from app.core.redis import cache_manager
from app.models.analysis import (
    AnalysisSession, AnalysisResult, Suggestion, KeywordAnalysis,
    ATSAnalysis, CompetitiveAnalysis, AnalysisType, SuggestionType,
    SuggestionPriority
)
from app.models.resume import Resume
from app.models.cover_letter import CoverLetter, JobDescription
from app.schemas.analysis import (
    AnalysisSessionCreate, AnalysisSessionResponse, AnalysisResultCreate,
    AnalysisResultResponse, SuggestionCreate, SuggestionResponse,
    KeywordAnalysisCreate, ATSAnalysisCreate, CompetitiveAnalysisCreate,
    AnalysisRequest, AnalysisComparisonResponse, RealTimeAnalysisRequest,
    RealTimeAnalysisResponse
)
from app.utils.exceptions import AppException
from app.services.ai_service import AIService
from app.tasks.analysis_tasks import process_analysis_session_task


class AnalysisService:
    """Service for comprehensive document analysis."""

    def __init__(self, ai_service: AIService = None):
        self.ai_service = ai_service or AIService()

    async def create_analysis_session(
        self,
        user_id: uuid.UUID,
        session_data: AnalysisSessionCreate
    ) -> AnalysisSessionResponse:
        """Create a new analysis session."""
        async with get_db_context() as db:
            # Validate target documents exist and belong to user
            await self._validate_target_documents(db, user_id, session_data)

            # Create analysis session
            session = AnalysisSession(
                user_id=user_id,
                session_name=session_data.session_name,
                analysis_types=session_data.analysis_types,
                target_type=session_data.target_type,
                target_ids=session_data.target_ids,
                job_description_id=session_data.job_description_id,
                job_title=session_data.job_title,
                company_name=session_data.company_name,
                total_items=len(session_data.target_ids),
                status="pending"
            )

            db.add(session)
            await db.commit()
            await db.refresh(session)

            # Queue background processing
            process_analysis_session_task.delay(str(session.id))

            return await self._session_to_response(session)

    async def get_analysis_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> AnalysisSessionResponse:
        """Get analysis session by ID."""
        async with get_db_context() as db:
            session = await db.get(AnalysisSession, session_id)

            if not session or session.user_id != user_id:
                raise AppException(
                    message="Analysis session not found",
                    status_code=404,
                    error_code="SESSION_NOT_FOUND"
                )

            return await self._session_to_response(session)

    async def analyze_document(
        self,
        user_id: uuid.UUID,
        analysis_request: AnalysisRequest
    ) -> List[AnalysisResultResponse]:
        """Analyze documents directly without session."""
        results = []

        for target_id in analysis_request.target_ids:
            for analysis_type in analysis_request.analysis_types:
                result = await self._perform_single_analysis(
                    user_id=user_id,
                    target_type=analysis_request.target_type,
                    target_id=target_id,
                    analysis_type=analysis_type,
                    job_description=analysis_request.job_description,
                    job_description_id=analysis_request.job_description_id,
                    include_suggestions=analysis_request.include_suggestions
                )
                results.append(result)

        return results

    async def get_analysis_results(
        self,
        user_id: uuid.UUID,
        target_type: Optional[str] = None,
        target_id: Optional[uuid.UUID] = None,
        analysis_type: Optional[AnalysisType] = None,
        page: int = 1,
        size: int = 20
    ) -> Dict[str, Any]:
        """Get analysis results with pagination."""
        async with get_db_context() as db:
            query = db.query(AnalysisResult).filter(
                AnalysisResult.user_id == user_id
            )

            if target_type:
                query = query.filter(AnalysisResult.target_type == target_type)

            if target_id:
                query = query.filter(AnalysisResult.target_id == target_id)

            if analysis_type:
                query = query.filter(AnalysisResult.analysis_type == analysis_type)

            query = query.order_by(desc(AnalysisResult.created_at))

            # Get total count
            total = await db.scalar(
                query.with_only_columns(func.count(AnalysisResult.id))
            )

            # Apply pagination
            offset = (page - 1) * size
            results = await db.scalars(
                query.offset(offset).limit(size)
            )

            result_responses = []
            for result in results:
                result_responses.append(await self._result_to_response(result))

            return {
                "results": result_responses,
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            }

    async def get_suggestions(
        self,
        user_id: uuid.UUID,
        analysis_result_id: Optional[uuid.UUID] = None,
        priority: Optional[SuggestionPriority] = None,
        type: Optional[SuggestionType] = None,
        is_applied: Optional[bool] = None,
        page: int = 1,
        size: int = 50
    ) -> Dict[str, Any]:
        """Get suggestions with filtering."""
        async with get_db_context() as db:
            # Build base query through analysis results to ensure user ownership
            query = db.query(Suggestion).join(AnalysisResult).filter(
                AnalysisResult.user_id == user_id
            )

            if analysis_result_id:
                query = query.filter(Suggestion.analysis_result_id == analysis_result_id)

            if priority:
                query = query.filter(Suggestion.priority == priority)

            if type:
                query = query.filter(Suggestion.type == type)

            if is_applied is not None:
                query = query.filter(Suggestion.is_applied == is_applied)

            query = query.order_by(
                Suggestion.priority.desc(),
                desc(Suggestion.impact_score),
                desc(Suggestion.created_at)
            )

            # Get total count
            total = await db.scalar(
                query.with_only_columns(func.count(Suggestion.id))
            )

            # Apply pagination
            offset = (page - 1) * size
            suggestions = await db.scalars(
                query.offset(offset).limit(size)
            )

            suggestion_responses = []
            for suggestion in suggestions:
                suggestion_responses.append(await self._suggestion_to_response(suggestion))

            return {
                "suggestions": suggestion_responses,
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            }

    async def apply_suggestion(
        self,
        suggestion_id: uuid.UUID,
        user_id: uuid.UUID,
        user_feedback: Optional[str] = None
    ) -> SuggestionResponse:
        """Apply a suggestion."""
        async with get_db_context() as db:
            # Get suggestion with user verification
            suggestion = await db.scalar(
                db.query(Suggestion)
                .join(AnalysisResult)
                .filter(
                    Suggestion.id == suggestion_id,
                    AnalysisResult.user_id == user_id
                )
            )

            if not suggestion:
                raise AppException(
                    message="Suggestion not found",
                    status_code=404,
                    error_code="SUGGESTION_NOT_FOUND"
                )

            if suggestion.is_applied:
                raise AppException(
                    message="Suggestion already applied",
                    status_code=400,
                    error_code="SUGGESTION_ALREADY_APPLIED"
                )

            # Mark as applied
            suggestion.is_applied = True
            suggestion.applied_at = datetime.utcnow()
            suggestion.user_feedback = user_feedback

            await db.commit()
            await db.refresh(suggestion)

            return await self._suggestion_to_response(suggestion)

    async def dismiss_suggestion(
        self,
        suggestion_id: uuid.UUID,
        user_id: uuid.UUID,
        user_feedback: Optional[str] = None
    ) -> SuggestionResponse:
        """Dismiss a suggestion."""
        async with get_db_context() as db:
            suggestion = await db.scalar(
                db.query(Suggestion)
                .join(AnalysisResult)
                .filter(
                    Suggestion.id == suggestion_id,
                    AnalysisResult.user_id == user_id
                )
            )

            if not suggestion:
                raise AppException(
                    message="Suggestion not found",
                    status_code=404,
                    error_code="SUGGESTION_NOT_FOUND"
                )

            suggestion.is_dismissed = True
            suggestion.dismissed_at = datetime.utcnow()
            suggestion.user_feedback = user_feedback

            await db.commit()
            await db.refresh(suggestion)

            return await self._suggestion_to_response(suggestion)

    async def compare_documents(
        self,
        user_id: uuid.UUID,
        target_ids: List[uuid.UUID],
        comparison_criteria: List[str] = None
    ) -> AnalysisComparisonResponse:
        """Compare multiple documents."""
        if len(target_ids) < 2:
            raise AppException(
                message="At least 2 documents required for comparison",
                status_code=400,
                error_code="INSUFFICIENT_DOCUMENTS"
            )

        criteria = comparison_criteria or ["overall_score", "ats_score", "keyword_score"]
        comparison_results = {}
        best_performers = {}

        async with get_db_context() as db:
            for target_id in target_ids:
                # Get latest analysis for each document
                latest_analysis = await db.scalar(
                    db.query(AnalysisResult)
                    .filter(
                        AnalysisResult.target_id == target_id,
                        AnalysisResult.user_id == user_id
                    )
                    .order_by(desc(AnalysisResult.created_at))
                    .limit(1)
                )

                if latest_analysis:
                    comparison_results[str(target_id)] = {
                        "overall_score": latest_analysis.overall_score,
                        "sub_scores": latest_analysis.sub_scores,
                        "analysis_data": latest_analysis.analysis_data,
                        "created_at": latest_analysis.created_at.isoformat()
                    }

        # Find best performers for each criterion
        for criterion in criteria:
            best_score = 0
            best_id = None

            for doc_id, data in comparison_results.items():
                score = data.get("sub_scores", {}).get(criterion, data.get("overall_score", 0))
                if score > best_score:
                    best_score = score
                    best_id = doc_id

            if best_id:
                best_performers[criterion] = uuid.UUID(best_id)

        # Generate improvement opportunities
        improvement_opportunities = await self._generate_improvement_opportunities(
            comparison_results
        )

        return AnalysisComparisonResponse(
            comparison_results=comparison_results,
            best_performers=best_performers,
            improvement_opportunities=improvement_opportunities,
            summary=await self._generate_comparison_summary(comparison_results),
            recommendations=await self._generate_comparison_recommendations(comparison_results)
        )

    async def real_time_analysis(
        self,
        user_id: uuid.UUID,
        request: RealTimeAnalysisRequest
    ) -> RealTimeAnalysisResponse:
        """Perform real-time analysis on content."""
        import time
        start_time = time.time()

        # Perform quick analysis using AI service
        analysis_result = await self.ai_service.analyze_content_real_time(
            content=request.content,
            content_type=request.content_type,
            analysis_types=request.analysis_types,
            job_context=request.job_context
        )

        processing_time = time.time() - start_time

        return RealTimeAnalysisResponse(
            content_score=analysis_result.get("content_score", 75.0),
            issues=analysis_result.get("issues", []),
            suggestions=analysis_result.get("suggestions", []),
            improvements=analysis_result.get("improvements", []),
            confidence=analysis_result.get("confidence", 0.85),
            processing_time=processing_time
        )

    async def get_keyword_analysis(
        self,
        user_id: uuid.UUID,
        target_id: uuid.UUID,
        job_description_id: Optional[uuid.UUID] = None
    ) -> Optional[Dict[str, Any]]:
        """Get keyword analysis for a document."""
        async with get_db_context() as db:
            query = db.query(KeywordAnalysis).filter(
                KeywordAnalysis.user_id == user_id,
                KeywordAnalysis.target_id == target_id
            )

            if job_description_id:
                query = query.filter(KeywordAnalysis.job_description_id == job_description_id)

            analysis = await db.scalar(
                query.order_by(desc(KeywordAnalysis.created_at)).limit(1)
            )

            if not analysis:
                return None

            return {
                "id": analysis.id,
                "extracted_keywords": analysis.extracted_keywords,
                "missing_keywords": analysis.missing_keywords,
                "matching_keywords": analysis.matching_keywords,
                "suggested_keywords": analysis.suggested_keywords,
                "keyword_density_score": analysis.keyword_density_score,
                "keyword_relevance_score": analysis.keyword_relevance_score,
                "keyword_diversity_score": analysis.keyword_diversity_score,
                "overall_keyword_score": analysis.overall_keyword_score,
                "keyword_frequency": analysis.keyword_frequency,
                "created_at": analysis.created_at
            }

    async def get_ats_analysis(
        self,
        user_id: uuid.UUID,
        target_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Get ATS analysis for a document."""
        async with get_db_context() as db:
            analysis = await db.scalar(
                db.query(ATSAnalysis)
                .filter(
                    ATSAnalysis.user_id == user_id,
                    ATSAnalysis.target_id == target_id
                )
                .order_by(desc(ATSAnalysis.created_at))
                .limit(1)
            )

            if not analysis:
                return None

            return {
                "id": analysis.id,
                "parsing_score": analysis.parsing_score,
                "formatting_score": analysis.formatting_score,
                "readability_score": analysis.readability_score,
                "overall_ats_score": analysis.overall_ats_score,
                "parseable_sections": analysis.parseable_sections,
                "problematic_sections": analysis.problematic_sections,
                "formatting_issues": analysis.formatting_issues,
                "text_extraction_quality": analysis.text_extraction_quality,
                "contact_extraction_confidence": analysis.contact_extraction_confidence,
                "created_at": analysis.created_at
            }

    async def get_competitive_analysis(
        self,
        user_id: uuid.UUID,
        target_id: uuid.UUID,
        industry: str,
        job_level: str
    ) -> Optional[Dict[str, Any]]:
        """Get competitive analysis for a document."""
        async with get_db_context() as db:
            analysis = await db.scalar(
                db.query(CompetitiveAnalysis)
                .filter(
                    CompetitiveAnalysis.user_id == user_id,
                    CompetitiveAnalysis.target_id == target_id,
                    CompetitiveAnalysis.industry == industry,
                    CompetitiveAnalysis.job_level == job_level
                )
                .order_by(desc(CompetitiveAnalysis.created_at))
                .limit(1)
            )

            if not analysis:
                return None

            return {
                "id": analysis.id,
                "content_competitiveness": analysis.content_competitiveness,
                "keyword_competitiveness": analysis.keyword_competitiveness,
                "format_competitiveness": analysis.format_competitiveness,
                "overall_competitiveness": analysis.overall_competitiveness,
                "content_percentile": analysis.content_percentile,
                "keyword_percentile": analysis.keyword_percentile,
                "format_percentile": analysis.format_percentile,
                "overall_percentile": analysis.overall_percentile,
                "industry_averages": analysis.industry_averages,
                "improvement_opportunities": analysis.improvement_opportunities,
                "created_at": analysis.created_at
            }

    # Helper methods
    async def _validate_target_documents(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        session_data: AnalysisSessionCreate
    ):
        """Validate that target documents exist and belong to user."""
        for target_id_str in session_data.target_ids:
            target_id = uuid.UUID(target_id_str)

            if session_data.target_type == "resume":
                document = await db.get(Resume, target_id)
                if not document or document.user_id != user_id:
                    raise AppException(
                        message=f"Resume {target_id} not found",
                        status_code=404,
                        error_code="DOCUMENT_NOT_FOUND"
                    )

            elif session_data.target_type == "cover_letter":
                document = await db.get(CoverLetter, target_id)
                if not document or document.user_id != user_id:
                    raise AppException(
                        message=f"Cover letter {target_id} not found",
                        status_code=404,
                        error_code="DOCUMENT_NOT_FOUND"
                    )

    async def _perform_single_analysis(
        self,
        user_id: uuid.UUID,
        target_type: str,
        target_id: uuid.UUID,
        analysis_type: AnalysisType,
        job_description: Optional[str] = None,
        job_description_id: Optional[uuid.UUID] = None,
        include_suggestions: bool = True
    ) -> AnalysisResultResponse:
        """Perform a single analysis on a document."""
        async with get_db_context() as db:
            # Get document content
            document_content = await self._get_document_content(
                db, target_type, target_id, user_id
            )

            # Get job description if provided
            job_desc_content = job_description
            if job_description_id:
                job_desc = await db.get(JobDescription, job_description_id)
                if job_desc and job_desc.user_id == user_id:
                    job_desc_content = job_desc.description

            # Perform analysis using AI service
            analysis_data = await self.ai_service.analyze_document(
                content=document_content,
                document_type=target_type,
                analysis_type=analysis_type.value,
                job_description=job_desc_content
            )

            # Create analysis result
            result = AnalysisResult(
                target_type=target_type,
                target_id=target_id,
                user_id=user_id,
                analysis_type=analysis_type,
                overall_score=analysis_data["overall_score"],
                sub_scores=analysis_data.get("sub_scores", {}),
                analysis_data=analysis_data,
                processing_time=analysis_data.get("processing_time"),
                ai_model_used=analysis_data.get("ai_model_used"),
                confidence_score=analysis_data.get("confidence_score")
            )

            db.add(result)
            await db.commit()
            await db.refresh(result)

            # Create suggestions if requested
            if include_suggestions and analysis_data.get("suggestions"):
                await self._create_suggestions(db, result, analysis_data["suggestions"])

            return await self._result_to_response(result)

    async def _get_document_content(
        self,
        db: AsyncSession,
        target_type: str,
        target_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Get document content for analysis."""
        if target_type == "resume":
            document = await db.get(Resume, target_id)
            if not document or document.user_id != user_id:
                raise AppException(
                    message="Document not found",
                    status_code=404,
                    error_code="DOCUMENT_NOT_FOUND"
                )

            return {
                "personal_info": document.personal_info,
                "work_experience": document.work_experience,
                "education": document.education,
                "skills": document.skills,
                "projects": document.projects,
                "certifications": document.certifications,
                "languages": document.languages,
                "awards": document.awards,
                "custom_sections": document.custom_sections,
            }

        elif target_type == "cover_letter":
            document = await db.get(CoverLetter, target_id)
            if not document or document.user_id != user_id:
                raise AppException(
                    message="Document not found",
                    status_code=404,
                    error_code="DOCUMENT_NOT_FOUND"
                )

            return {
                "content": document.content,
                "opening_paragraph": document.opening_paragraph,
                "body_paragraphs": document.body_paragraphs,
                "closing_paragraph": document.closing_paragraph,
                "job_title": document.job_title,
                "company_name": document.company_name,
            }

        else:
            raise AppException(
                message="Unsupported document type",
                status_code=400,
                error_code="UNSUPPORTED_DOCUMENT_TYPE"
            )

    async def _create_suggestions(
        self,
        db: AsyncSession,
        result: AnalysisResult,
        suggestions_data: List[Dict[str, Any]]
    ):
        """Create suggestion records from analysis data."""
        for suggestion_data in suggestions_data:
            suggestion = Suggestion(
                analysis_result_id=result.id,
                type=SuggestionType(suggestion_data.get("type", "content_improvement")),
                priority=SuggestionPriority(suggestion_data.get("priority", "medium")),
                title=suggestion_data["title"],
                description=suggestion_data["description"],
                before_text=suggestion_data.get("before_text"),
                suggested_text=suggestion_data.get("suggested_text"),
                section=suggestion_data.get("section"),
                field_path=suggestion_data.get("field_path"),
                impact_score=suggestion_data.get("impact_score"),
                reasoning=suggestion_data.get("reasoning"),
                confidence=suggestion_data.get("confidence"),
                implementation_effort=suggestion_data.get("implementation_effort"),
                estimated_time_minutes=suggestion_data.get("estimated_time_minutes"),
                requires_manual_review=suggestion_data.get("requires_manual_review", False)
            )

            db.add(suggestion)

        await db.commit()

    async def _session_to_response(self, session: AnalysisSession) -> AnalysisSessionResponse:
        """Convert session model to response."""
        return AnalysisSessionResponse(
            id=session.id,
            user_id=session.user_id,
            session_name=session.session_name,
            analysis_types=session.analysis_types,
            target_type=session.target_type,
            target_ids=session.target_ids,
            job_description_id=session.job_description_id,
            job_title=session.job_title,
            company_name=session.company_name,
            status=session.status,
            progress=session.progress,
            total_items=session.total_items,
            completed_items=session.completed_items,
            failed_items=session.failed_items,
            overall_score=session.overall_score,
            score_breakdown=session.score_breakdown,
            total_suggestions=session.total_suggestions,
            critical_issues=session.critical_issues,
            processing_time=session.processing_time,
            ai_model_used=session.ai_model_used,
            error_details=session.error_details,
            created_at=session.created_at,
            started_at=session.started_at,
            completed_at=session.completed_at
        )

    async def _result_to_response(self, result: AnalysisResult) -> AnalysisResultResponse:
        """Convert result model to response."""
        async with get_db_context() as db:
            # Get suggestions for this result
            suggestions = await db.scalars(
                db.query(Suggestion).filter(
                    Suggestion.analysis_result_id == result.id
                ).order_by(Suggestion.priority.desc(), desc(Suggestion.impact_score))
            )

            suggestion_responses = []
            for suggestion in suggestions:
                suggestion_responses.append(await self._suggestion_to_response(suggestion))

        return AnalysisResultResponse(
            id=result.id,
            session_id=result.session_id,
            target_type=result.target_type,
            target_id=result.target_id,
            user_id=result.user_id,
            analysis_type=result.analysis_type,
            analysis_version=result.analysis_version,
            overall_score=result.overall_score,
            sub_scores=result.sub_scores,
            analysis_data=result.analysis_data,
            raw_results=result.raw_results,
            processing_time=result.processing_time,
            ai_model_used=result.ai_model_used,
            confidence_score=result.confidence_score,
            created_at=result.created_at,
            suggestions=suggestion_responses
        )

    async def _suggestion_to_response(self, suggestion: Suggestion) -> SuggestionResponse:
        """Convert suggestion model to response."""
        return SuggestionResponse(
            id=suggestion.id,
            analysis_result_id=suggestion.analysis_result_id,
            type=suggestion.type,
            priority=suggestion.priority,
            title=suggestion.title,
            description=suggestion.description,
            before_text=suggestion.before_text,
            suggested_text=suggestion.suggested_text,
            section=suggestion.section,
            field_path=suggestion.field_path,
            line_number=suggestion.line_number,
            character_position=suggestion.character_position,
            impact_score=suggestion.impact_score,
            reasoning=suggestion.reasoning,
            confidence=suggestion.confidence,
            implementation_effort=suggestion.implementation_effort,
            estimated_time_minutes=suggestion.estimated_time_minutes,
            requires_manual_review=suggestion.requires_manual_review,
            is_applied=suggestion.is_applied,
            is_dismissed=suggestion.is_dismissed,
            user_feedback=suggestion.user_feedback,
            applied_at=suggestion.applied_at,
            dismissed_at=suggestion.dismissed_at,
            created_at=suggestion.created_at
        )

    async def _generate_improvement_opportunities(
        self,
        comparison_results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate improvement opportunities from comparison."""
        opportunities = []

        # Find areas where all documents score poorly
        for doc_id, data in comparison_results.items():
            sub_scores = data.get("sub_scores", {})

            for metric, score in sub_scores.items():
                if score < 70:  # Poor score threshold
                    opportunities.append({
                        "document_id": doc_id,
                        "metric": metric,
                        "current_score": score,
                        "improvement_potential": min(100 - score, 30),
                        "priority": "high" if score < 50 else "medium"
                    })

        return opportunities

    async def _generate_comparison_summary(
        self,
        comparison_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate a summary of the comparison."""
        if not comparison_results:
            return "No documents to compare."

        scores = [data.get("overall_score", 0) for data in comparison_results.values()]
        avg_score = sum(scores) / len(scores)
        best_score = max(scores)
        worst_score = min(scores)

        return (
            f"Compared {len(comparison_results)} documents. "
            f"Average score: {avg_score:.1f}, "
            f"Best: {best_score:.1f}, "
            f"Worst: {worst_score:.1f}. "
            f"Score range: {best_score - worst_score:.1f} points."
        )

    async def _generate_comparison_recommendations(
        self,
        comparison_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []

        if len(comparison_results) < 2:
            return recommendations

        scores = [(doc_id, data.get("overall_score", 0)) for doc_id, data in comparison_results.items()]
        scores.sort(key=lambda x: x[1], reverse=True)

        best_doc = scores[0][0]
        worst_doc = scores[-1][0]

        recommendations.append(f"Use document {best_doc} as your primary template")

        if scores[0][1] - scores[-1][1] > 20:
            recommendations.append(f"Document {worst_doc} needs significant improvement")

        # Add specific metric recommendations
        best_data = comparison_results[best_doc]
        best_sub_scores = best_data.get("sub_scores", {})

        if best_sub_scores.get("ats_score", 0) > 85:
            recommendations.append("Leverage the ATS-optimized format from your best document")

        if best_sub_scores.get("keyword_score", 0) > 85:
            recommendations.append("Apply the keyword strategy from your top-performing document")

        return recommendations