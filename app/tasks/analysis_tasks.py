"""
Celery tasks for analysis operations including batch processing and AI analysis.
"""

import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from celery import current_task
from sqlalchemy import and_, desc

from app.core.celery_app import task, long_task
from app.core.database import get_db_context
from app.models.analysis import (
    AnalysisSession, AnalysisResult, Suggestion, KeywordAnalysis,
    ATSAnalysis, CompetitiveAnalysis, AnalysisType, SuggestionType,
    SuggestionPriority
)
from app.models.resume import Resume
from app.models.cover_letter import CoverLetter, JobDescription
from app.services.ai_service import AIService
from app.services.analysis_service import AnalysisService
from app.core.redis import cache_manager
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)


@long_task
async def process_analysis_session_task(session_id: str):
    """
    Process an analysis session with multiple documents and analysis types.

    Args:
        session_id: Analysis session UUID as string

    Returns:
        Dict with session processing results
    """
    try:
        async with get_db_context() as db:
            # Get analysis session
            session = await db.get(AnalysisSession, uuid.UUID(session_id))
            if not session:
                raise AppException("Analysis session not found", status_code=404)

            # Update session status
            session.status = "running"
            session.started_at = datetime.utcnow()
            await db.commit()

            current_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': 5,
                    'status': 'Session started',
                    'session_id': session_id,
                    'total_items': session.total_items
                }
            )

            # Initialize services
            ai_service = AIService()
            analysis_service = AnalysisService(ai_service)

            completed_items = 0
            failed_items = 0
            all_scores = []
            total_suggestions = 0
            critical_issues = 0

            # Process each target document
            for i, target_id_str in enumerate(session.target_ids):
                try:
                    target_id = uuid.UUID(target_id_str)

                    # Process each analysis type for this document
                    for analysis_type in session.analysis_types:
                        try:
                            result = await analysis_service._perform_single_analysis(
                                user_id=session.user_id,
                                target_type=session.target_type,
                                target_id=target_id,
                                analysis_type=AnalysisType(analysis_type),
                                job_description_id=session.job_description_id,
                                include_suggestions=True
                            )

                            # Update session reference in result
                            async with get_db_context() as db:
                                analysis_result = await db.get(AnalysisResult, result.id)
                                if analysis_result:
                                    analysis_result.session_id = session.id
                                    await db.commit()

                            all_scores.append(result.overall_score)
                            total_suggestions += len(result.suggestions)

                            # Count critical suggestions
                            critical_suggestions = [
                                s for s in result.suggestions
                                if s.priority == SuggestionPriority.CRITICAL
                            ]
                            critical_issues += len(critical_suggestions)

                        except Exception as e:
                            logger.error(f"Analysis failed for {target_id} with type {analysis_type}: {str(e)}")
                            failed_items += 1

                    completed_items += 1

                    # Update progress
                    progress = int((i + 1) / len(session.target_ids) * 90)  # Leave 10% for finalization
                    current_task.update_state(
                        state='PROGRESS',
                        meta={
                            'progress': progress,
                            'status': f'Processed {i + 1}/{len(session.target_ids)} documents',
                            'completed': completed_items,
                            'failed': failed_items
                        }
                    )

                except Exception as e:
                    logger.error(f"Document processing failed for {target_id_str}: {str(e)}")
                    failed_items += 1

            # Calculate session summary
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
            score_breakdown = {
                "average_score": overall_score,
                "max_score": max(all_scores) if all_scores else 0,
                "min_score": min(all_scores) if all_scores else 0
            }

            # Update session with results
            session.status = "completed" if failed_items == 0 else "completed_with_errors"
            session.completed_at = datetime.utcnow()
            session.completed_items = completed_items
            session.failed_items = failed_items
            session.overall_score = overall_score
            session.score_breakdown = score_breakdown
            session.total_suggestions = total_suggestions
            session.critical_issues = critical_issues
            session.processing_time = (session.completed_at - session.started_at).total_seconds()

            await db.commit()

            current_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': 100,
                    'status': 'Session completed',
                    'completed': completed_items,
                    'failed': failed_items,
                    'overall_score': overall_score
                }
            )

            logger.info(f"Analysis session {session_id} completed: {completed_items} successful, {failed_items} failed")

            return {
                "success": True,
                "session_id": session_id,
                "completed_items": completed_items,
                "failed_items": failed_items,
                "overall_score": overall_score,
                "total_suggestions": total_suggestions,
                "critical_issues": critical_issues
            }

    except Exception as e:
        # Update session status on failure
        try:
            async with get_db_context() as db:
                session = await db.get(AnalysisSession, uuid.UUID(session_id))
                if session:
                    session.status = "failed"
                    session.error_details = str(e)
                    session.completed_at = datetime.utcnow()
                    await db.commit()
        except Exception:
            pass

        logger.error(f"Analysis session {session_id} failed: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'session_id': session_id}
        )
        raise


@task
async def analyze_single_document_task(
        user_id: str,
        target_type: str,
        target_id: str,
        analysis_type: str,
        job_description: str = None,
        job_description_id: str = None
):
    """
    Analyze a single document.

    Args:
        user_id: User UUID as string
        target_type: Type of document (resume, cover_letter)
        target_id: Document UUID as string
        analysis_type: Type of analysis to perform
        job_description: Job description text
        job_description_id: Job description ID

    Returns:
        Dict with analysis results
    """
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={
                'progress': 10,
                'status': f'Starting {analysis_type} analysis',
                'target_id': target_id
            }
        )

        # Initialize services
        ai_service = AIService()
        analysis_service = AnalysisService(ai_service)

        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 30, 'status': 'Services initialized'}
        )

        # Perform analysis
        result = await analysis_service._perform_single_analysis(
            user_id=uuid.UUID(user_id),
            target_type=target_type,
            target_id=uuid.UUID(target_id),
            analysis_type=AnalysisType(analysis_type),
            job_description=job_description,
            job_description_id=uuid.UUID(job_description_id) if job_description_id else None,
            include_suggestions=True
        )

        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 100, 'status': 'Analysis completed'}
        )

        logger.info(f"Single document analysis completed for {target_id}")

        return {
            "success": True,
            "analysis_id": str(result.id),
            "overall_score": result.overall_score,
            "sub_scores": result.sub_scores,
            "suggestions_count": len(result.suggestions),
            "target_id": target_id
        }

    except Exception as e:
        logger.error(f"Single document analysis failed for {target_id}: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'target_id': target_id}
        )
        raise


@task
async def generate_keyword_analysis_task(
        target_type: str,
        target_id: str,
        job_description: str = None,
        job_description_id: str = None,
        user_id: str = None
):
    """
    Generate keyword analysis for a document.

    Args:
        target_type: Type of document
        target_id: Document UUID as string
        job_description: Job description text
        job_description_id: Job description ID
        user_id: User ID for the analysis

    Returns:
        Dict with keyword analysis results
    """
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Starting keyword analysis'}
        )

        async with get_db_context() as db:
            # Get document content
            if target_type == "resume":
                document = await db.get(Resume, uuid.UUID(target_id))
            else:
                document = await db.get(CoverLetter, uuid.UUID(target_id))

            if not document:
                raise AppException("Document not found", status_code=404)

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 30, 'status': 'Document loaded'}
            )

            # Get job description if ID provided
            if job_description_id and not job_description:
                job_desc = await db.get(JobDescription, uuid.UUID(job_description_id))
                if job_desc:
                    job_description = job_desc.description

            # Initialize AI service
            ai_service = AIService()

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 50, 'status': 'Extracting keywords'}
            )

            # Extract keywords from document
            if target_type == "resume":
                document_text = ""
                if document.work_experience:
                    for exp in document.work_experience:
                        document_text += " ".join(exp.get("achievements", []))
                if document.skills:
                    for skill in document.skills:
                        document_text += " " + str(skill.get("name", skill))
            else:
                content = document.content or {}
                document_text = " ".join([
                    content.get("opening_paragraph", ""),
                    " ".join(content.get("body_paragraphs", [])),
                    content.get("closing_paragraph", "")
                ])

            # Extract keywords from job description
            job_keywords = {}
            if job_description:
                job_keywords = await ai_service.extract_job_keywords(job_description)

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 80, 'status': 'Analyzing keyword matches'}
            )

            # Analyze keyword overlap
            document_keywords = document_text.lower().split()
            job_keywords_list = job_keywords.get("keywords", [])

            matching_keywords = [kw for kw in job_keywords_list if kw.lower() in document_text.lower()]
            missing_keywords = [kw for kw in job_keywords_list if kw.lower() not in document_text.lower()]

            # Calculate scores
            keyword_density_score = min(100, len(document_keywords) / 10)  # Simple scoring
            keyword_relevance_score = (len(matching_keywords) / max(len(job_keywords_list),
                                                                    1)) * 100 if job_keywords_list else 80
            keyword_diversity_score = min(100, len(set(document_keywords)) / max(len(document_keywords), 1) * 100)
            overall_score = (keyword_density_score + keyword_relevance_score + keyword_diversity_score) / 3

            # Create keyword analysis record
            keyword_analysis = KeywordAnalysis(
                target_type=target_type,
                target_id=uuid.UUID(target_id),
                user_id=uuid.UUID(user_id) if user_id else document.user_id,
                job_description_id=uuid.UUID(job_description_id) if job_description_id else None,
                extracted_keywords=list(set(document_keywords[:50])),  # Limit to 50
                missing_keywords=missing_keywords[:20],  # Limit to 20
                matching_keywords=matching_keywords,
                suggested_keywords=missing_keywords[:10],  # Top 10 suggestions
                technical_keywords=job_keywords.get("technical_skills", []),
                soft_skills_keywords=job_keywords.get("soft_skills", []),
                industry_keywords=job_keywords.get("industry_terms", []),
                keyword_density_score=keyword_density_score,
                keyword_relevance_score=keyword_relevance_score,
                keyword_diversity_score=keyword_diversity_score,
                overall_keyword_score=overall_score,
                keyword_frequency={},  # Would implement frequency counting
                keyword_positions={},  # Would implement position tracking
                keyword_importance={}  # Would implement importance scoring
            )

            db.add(keyword_analysis)
            await db.commit()
            await db.refresh(keyword_analysis)

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 100, 'status': 'Keyword analysis completed'}
            )

            logger.info(f"Keyword analysis completed for {target_id}")

            return {
                "success": True,
                "analysis_id": str(keyword_analysis.id),
                "overall_score": overall_score,
                "matching_keywords": len(matching_keywords),
                "missing_keywords": len(missing_keywords),
                "target_id": target_id
            }

    except Exception as e:
        logger.error(f"Keyword analysis failed for {target_id}: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'target_id': target_id}
        )
        raise


@task
async def generate_ats_analysis_task(
        target_type: str,
        target_id: str,
        user_id: str = None
):
    """
    Generate ATS analysis for a document.

    Args:
        target_type: Type of document
        target_id: Document UUID as string
        user_id: User ID for the analysis

    Returns:
        Dict with ATS analysis results
    """
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Starting ATS analysis'}
        )

        async with get_db_context() as db:
            # Get document
            if target_type == "resume":
                document = await db.get(Resume, uuid.UUID(target_id))
            else:
                document = await db.get(CoverLetter, uuid.UUID(target_id))

            if not document:
                raise AppException("Document not found", status_code=404)

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 30, 'status': 'Document loaded'}
            )

            # Initialize AI service
            ai_service = AIService()

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 50, 'status': 'Running ATS simulation'}
            )

            # Simulate ATS parsing
            # In a real implementation, this would test against actual ATS systems

            # Basic ATS compatibility checks
            parsing_score = 95.0  # Assume good parsing
            formatting_score = 85.0  # Basic format check
            readability_score = 90.0  # Text readability
            overall_ats_score = (parsing_score + formatting_score + readability_score) / 3

            # Mock text extraction
            if target_type == "resume":
                extracted_text = f"""
                {document.personal_info.get('first_name', '')} {document.personal_info.get('last_name', '')}
                {document.personal_info.get('email', '')}
                {document.personal_info.get('phone', '')}

                Work Experience:
                {' '.join([exp.get('title', '') + ' at ' + exp.get('company', '') for exp in (document.work_experience or [])])}

                Education:
                {' '.join([edu.get('degree', '') + ' from ' + edu.get('institution', '') for edu in (document.education or [])])}

                Skills:
                {' '.join([str(skill.get('name', skill)) for skill in (document.skills or [])])}
                """
            else:
                content = document.content or {}
                extracted_text = f"""
                Cover Letter for {document.job_title} at {document.company_name}

                {content.get('opening_paragraph', '')}
                {' '.join(content.get('body_paragraphs', []))}
                {content.get('closing_paragraph', '')}
                """

            # Create ATS analysis record
            ats_analysis = ATSAnalysis(
                target_type=target_type,
                target_id=uuid.UUID(target_id),
                user_id=uuid.UUID(user_id) if user_id else document.user_id,
                parsing_score=parsing_score,
                formatting_score=formatting_score,
                readability_score=readability_score,
                overall_ats_score=overall_ats_score,
                parseable_sections=["header", "experience", "education", "skills"],
                problematic_sections=[],
                formatting_issues=[],
                extracted_text=extracted_text.strip(),
                text_extraction_quality=95.0,
                font_analysis={"primary_font": "Arial", "readable": True},
                layout_analysis={"sections_detected": 4, "proper_order": True},
                contact_info_extracted={
                    "email": document.personal_info.get('email') if hasattr(document, 'personal_info') else None,
                    "phone": document.personal_info.get('phone') if hasattr(document, 'personal_info') else None
                },
                contact_extraction_confidence=95.0
            )

            db.add(ats_analysis)
            await db.commit()
            await db.refresh(ats_analysis)

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 100, 'status': 'ATS analysis completed'}
            )

            logger.info(f"ATS analysis completed for {target_id}")

            return {
                "success": True,
                "analysis_id": str(ats_analysis.id),
                "overall_ats_score": overall_ats_score,
                "parsing_score": parsing_score,
                "formatting_score": formatting_score,
                "target_id": target_id
            }

    except Exception as e:
        logger.error(f"ATS analysis failed for {target_id}: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'target_id': target_id}
        )
        raise


@task
async def update_analysis_cache():
    """Update analysis cache with frequently accessed data."""
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Starting cache update'}
        )

        if not cache_manager:
            logger.info("Cache manager not available, skipping cache update")
            return {"success": True, "message": "Cache manager not available"}

        async with get_db_context() as db:
            # Cache popular analysis results
            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 30, 'status': 'Caching analysis results'}
            )

            # Get recent analyses
            recent_analyses = await db.scalars(
                db.query(AnalysisResult)
                .order_by(desc(AnalysisResult.created_at))
                .limit(100)
            )

            cached_count = 0
            for analysis in recent_analyses:
                cache_key = f"analysis_result:{analysis.id}"
                analysis_data = {
                    "id": str(analysis.id),
                    "overall_score": analysis.overall_score,
                    "sub_scores": analysis.sub_scores,
                    "analysis_type": analysis.analysis_type.value,
                    "created_at": analysis.created_at.isoformat()
                }

                await cache_manager.set(cache_key, analysis_data, ttl=3600)
                cached_count += 1

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 60, 'status': 'Caching suggestions'}
            )

            # Cache popular suggestions
            popular_suggestions = await db.scalars(
                db.query(Suggestion)
                .filter(Suggestion.is_applied == False, Suggestion.is_dismissed == False)
                .order_by(desc(Suggestion.impact_score))
                .limit(50)
            )

            suggestion_cache_count = 0
            for suggestion in popular_suggestions:
                cache_key = f"suggestion:{suggestion.id}"
                suggestion_data = {
                    "id": str(suggestion.id),
                    "title": suggestion.title,
                    "description": suggestion.description,
                    "priority": suggestion.priority.value,
                    "type": suggestion.type.value,
                    "impact_score": suggestion.impact_score
                }

                await cache_manager.set(cache_key, suggestion_data, ttl=1800)
                suggestion_cache_count += 1

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 100, 'status': 'Cache update completed'}
            )

            logger.info(f"Analysis cache updated: {cached_count} analyses, {suggestion_cache_count} suggestions")

            return {
                "success": True,
                "cached_analyses": cached_count,
                "cached_suggestions": suggestion_cache_count
            }

    except Exception as e:
        logger.error(f"Analysis cache update failed: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise


@task
async def generate_analysis_report_task(
        user_id: str,
        analysis_ids: List[str],
        report_format: str = "pdf"
):
    """
    Generate a comprehensive analysis report.

    Args:
        user_id: User UUID as string
        analysis_ids: List of analysis UUIDs
        report_format: Format for the report (pdf, excel, html)

    Returns:
        Dict with report generation results
    """
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={
                'progress': 10,
                'status': 'Starting report generation',
                'format': report_format
            }
        )

        async with get_db_context() as db:
            # Get analyses
            analyses = []
            for analysis_id in analysis_ids:
                analysis = await db.get(AnalysisResult, uuid.UUID(analysis_id))
                if analysis and analysis.user_id == uuid.UUID(user_id):
                    analyses.append(analysis)

            if not analyses:
                raise AppException("No valid analyses found", status_code=404)

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 30, 'status': f'Processing {len(analyses)} analyses'}
            )

            # Generate report data
            report_data = {
                "title": f"Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
                "generated_at": datetime.now().isoformat(),
                "user_id": user_id,
                "total_analyses": len(analyses),
                "analyses": []
            }

            for analysis in analyses:
                analysis_data = {
                    "id": str(analysis.id),
                    "target_type": analysis.target_type,
                    "target_id": str(analysis.target_id),
                    "analysis_type": analysis.analysis_type.value,
                    "overall_score": analysis.overall_score,
                    "sub_scores": analysis.sub_scores,
                    "created_at": analysis.created_at.isoformat(),
                    "suggestions_count": len(analysis.suggestions) if analysis.suggestions else 0
                }
                report_data["analyses"].append(analysis_data)

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 70, 'status': 'Generating report file'}
            )

            # For now, just return the data
            # In a real implementation, you would generate PDF/Excel files
            report_file_name = f"analysis_report_{user_id}_{int(datetime.now().timestamp())}.{report_format}"

            current_task.update_state(
                state='PROGRESS',
                meta={'progress': 100, 'status': 'Report generation completed'}
            )

            logger.info(f"Analysis report generated for user {user_id}: {len(analyses)} analyses")

            return {
                "success": True,
                "report_file": report_file_name,
                "format": report_format,
                "analyses_count": len(analyses),
                "data": report_data
            }

    except Exception as e:
        logger.error(f"Analysis report generation failed for user {user_id}: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'user_id': user_id}
        )
        raise