"""
Celery tasks for resume operations including PDF generation, analysis, and file management.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from celery import current_task
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import task, long_task, priority_task
from app.core.database import get_db_context
from app.models.resume import Resume, ResumeAnalysis
from app.models.cover_letter import CoverLetter, CoverLetterAnalysis
from app.services.ai_service import AIService
from app.services.pdf_service import PDFService
from app.services.analysis_service import AnalysisService
from app.core.redis import cache_manager
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)


@task
async def generate_resume_pdf_task(resume_id: str, template_id: str = None, custom_settings: Dict[str, Any] = None):
    """
    Generate PDF for a resume.

    Args:
        resume_id: Resume UUID as string
        template_id: Optional template ID
        custom_settings: Custom PDF generation settings

    Returns:
        Dict with file information or error details
    """
    try:
        # Update task progress
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting PDF generation'})

        async with get_db_context() as db:
            # Get resume
            resume = await db.get(Resume, uuid.UUID(resume_id))
            if not resume:
                raise AppException("Resume not found", status_code=404)

            current_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Resume loaded'})

            # Initialize PDF service
            pdf_service = PDFService()

            current_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Generating PDF'})

            # Generate PDF
            result = await pdf_service.generate_resume_pdf(
                resume=resume,
                template_id=uuid.UUID(template_id) if template_id else None,
                custom_settings=custom_settings
            )

            current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'PDF generated'})

            # Update resume with PDF path
            resume.pdf_path = result["file_path"]
            resume.file_size = result["file_size"]
            await db.commit()

            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Completed'})

            logger.info(f"PDF generated for resume {resume_id}: {result['file_name']}")

            return {
                "success": True,
                "file_info": result,
                "resume_id": resume_id
            }

    except Exception as e:
        logger.error(f"PDF generation failed for resume {resume_id}: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'resume_id': resume_id}
        )
        raise


@task
async def generate_cover_letter_pdf_task(cover_letter_id: str, template_id: str = None,
                                         custom_settings: Dict[str, Any] = None):
    """
    Generate PDF for a cover letter.

    Args:
        cover_letter_id: Cover letter UUID as string
        template_id: Optional template ID
        custom_settings: Custom PDF generation settings

    Returns:
        Dict with file information or error details
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting PDF generation'})

        async with get_db_context() as db:
            # Get cover letter
            cover_letter = await db.get(CoverLetter, uuid.UUID(cover_letter_id))
            if not cover_letter:
                raise AppException("Cover letter not found", status_code=404)

            current_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Cover letter loaded'})

            # Initialize PDF service
            pdf_service = PDFService()

            current_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Generating PDF'})

            # Generate PDF
            result = await pdf_service.generate_cover_letter_pdf(
                cover_letter=cover_letter,
                template_id=uuid.UUID(template_id) if template_id else None,
                custom_settings=custom_settings
            )

            current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'PDF generated'})

            # Update cover letter with PDF path
            cover_letter.pdf_path = result["file_path"]
            cover_letter.file_size = result["file_size"]
            await db.commit()

            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Completed'})

            logger.info(f"PDF generated for cover letter {cover_letter_id}: {result['file_name']}")

            return {
                "success": True,
                "file_info": result,
                "cover_letter_id": cover_letter_id
            }

    except Exception as e:
        logger.error(f"PDF generation failed for cover letter {cover_letter_id}: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'cover_letter_id': cover_letter_id}
        )
        raise


@task
async def analyze_resume_task(
        resume_id: str,
        analysis_types: List[str] = None,
        job_description: str = None,
        job_description_id: str = None
):
    """
    Analyze a resume using AI.

    Args:
        resume_id: Resume UUID as string
        analysis_types: Types of analysis to perform
        job_description: Job description text for matching
        job_description_id: Job description ID for matching

    Returns:
        Dict with analysis results
    """
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting analysis'})

        analysis_types = analysis_types or ["comprehensive"]

        async with get_db_context() as db:
            # Get resume
            resume = await db.get(Resume, uuid.UUID(resume_id))
            if not resume:
                raise AppException("Resume not found", status_code=404)

            current_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Resume loaded'})

            # Get job description if ID provided
            if job_description_id and not job_description:
                from app.models.cover_letter import JobDescription
                job_desc = await db.get(JobDescription, uuid.UUID(job_description_id))
                if job_desc:
                    job_description = job_desc.description

            current_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Running AI analysis'})

            # Initialize AI service
            ai_service = AIService()

            # Prepare resume data
            resume_data = {
                "personal_info": resume.personal_info,
                "work_experience": resume.work_experience,
                "education": resume.education,
                "skills": resume.skills,
                "projects": resume.projects,
                "certifications": resume.certifications,
                "languages": resume.languages,
                "awards": resume.awards,
                "custom_sections": resume.custom_sections
            }

            # Perform analysis
            analysis_result = await ai_service.analyze_resume(
                resume_data=resume_data,
                job_description=job_description,
                analysis_types=analysis_types
            )

            current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Saving analysis results'})

            # Save analysis to database
            analysis = ResumeAnalysis(
                resume_id=uuid.UUID(resume_id),
                analysis_type="comprehensive",
                analysis_version="1.0",
                ats_score=analysis_result.get("sub_scores", {}).get("ats_score", 0),
                content_score=analysis_result.get("sub_scores", {}).get("content_score", 0),
                format_score=analysis_result.get("sub_scores", {}).get("format_score", 0),
                overall_score=analysis_result.get("overall_score", 0),
                keyword_analysis=analysis_result.get("keyword_analysis", {}),
                content_analysis=analysis_result.get("content_analysis", {}),
                format_analysis=analysis_result.get("format_analysis", {}),
                suggestions=analysis_result.get("suggestions", []),
                processing_time=analysis_result.get("processing_time"),
                ai_model_used=analysis_result.get("ai_model_used"),
                job_description_id=uuid.UUID(job_description_id) if job_description_id else None
            )

            db.add(analysis)

            # Update resume scores
            resume.ats_score = analysis_result.get("sub_scores", {}).get("ats_score")
            resume.content_score = analysis_result.get("sub_scores", {}).get("content_score")
            resume.format_score = analysis_result.get("sub_scores", {}).get("format_score")
            resume.overall_score = analysis_result.get("overall_score")
            resume.last_analyzed_at = datetime.utcnow()

            await db.commit()
            await db.refresh(analysis)

            current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Analysis completed'})

            logger.info(f"Analysis completed for resume {resume_id}")

            return {
                "success": True,
                "analysis_id": str(analysis.id),
                "overall_score": analysis_result.get("overall_score"),
                "sub_scores": analysis_result.get("sub_scores", {}),
                "suggestions_count": len(analysis_result.get("suggestions", [])),
                "resume_id": resume_id
            }

    except Exception as e:
        logger.error(f"Analysis failed for resume {resume_id}: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'resume_id': resume_id}
        )
        raise


@long_task
async def bulk_analyze_resumes_task(
        resume_ids: List[str],
        analysis_types: List[str] = None,
        job_description: str = None,
        user_id: str = None
):
    """
    Analyze multiple resumes in bulk.

    Args:
        resume_ids: List of resume UUIDs as strings
        analysis_types: Types of analysis to perform
        job_description: Job description for matching
        user_id: User ID for notifications

    Returns:
        Dict with bulk analysis results
    """
    try:
        total_resumes = len(resume_ids)
        completed = 0
        failed = 0
        results = []

        current_task.update_state(
            state='PROGRESS',
            meta={
                'progress': 0,
                'status': f'Starting bulk analysis of {total_resumes} resumes',
                'completed': completed,
                'failed': failed,
                'total': total_resumes
            }
        )

        for i, resume_id in enumerate(resume_ids):
            try:
                # Analyze individual resume
                result = await analyze_resume_task.apply_async(
                    args=[resume_id, analysis_types, job_description]
                ).get()

                results.append({
                    "resume_id": resume_id,
                    "success": True,
                    "analysis_id": result.get("analysis_id"),
                    "overall_score": result.get("overall_score")
                })
                completed += 1

            except Exception as e:
                logger.error(f"Bulk analysis failed for resume {resume_id}: {str(e)}")
                results.append({
                    "resume_id": resume_id,
                    "success": False,
                    "error": str(e)
                })
                failed += 1

            # Update progress
            progress = int((i + 1) / total_resumes * 100)
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'status': f'Analyzed {i + 1}/{total_resumes} resumes',
                    'completed': completed,
                    'failed': failed,
                    'total': total_resumes
                }
            )

        # Send notification if user_id provided
        if user_id:
            await send_analysis_completion_notification.delay(
                user_id, total_resumes, completed, failed
            )

        logger.info(f"Bulk analysis completed: {completed} successful, {failed} failed")

        return {
            "success": True,
            "total_resumes": total_resumes,
            "completed": completed,
            "failed": failed,
            "results": results
        }

    except Exception as e:
        logger.error(f"Bulk analysis task failed: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'resume_ids': resume_ids}
        )
        raise


@task
async def cleanup_expired_files():
    """Clean up expired PDF and temporary files."""
    try:
        current_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting cleanup'})

        # Initialize PDF service for cleanup
        pdf_service = PDFService()

        current_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Cleaning expired files'})

        # Clean up expired files
        await pdf_service.cleanup_expired_files()

        current_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Cleaning cache'})

        # Clean up old cache entries if Redis is available
        if cache_manager:
            # Clean up old resume cache entries
            pattern = "resume:*"
            await cache_manager.delete_pattern(pattern)

            # Clean up old analysis cache entries
            pattern = "analysis:*"
            await cache_manager.delete_pattern(pattern)

        current_task.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Cleanup completed'})

        logger.info("File cleanup completed successfully")

        return {"success": True, "message": "Cleanup completed"}

    except Exception as e:
        logger.error(f"File cleanup failed: {str(e)}")
        current_task.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@task
async def cleanup_temp_files():
    """Clean up temporary files created during processing."""
    try:
        import tempfile
        import os
        from pathlib import Path

        temp_dir = Path(tempfile.gettempdir())
        cutoff_time = datetime.now() - timedelta(hours=2)  # Remove files older than 2 hours

        cleaned_count = 0

        # Clean up temporary PDF files
        for temp_file in temp_dir.glob("tmp*.pdf"):
            try:
                if temp_file.stat().st_mtime < cutoff_time.timestamp():
                    temp_file.unlink()
                    cleaned_count += 1
            except (OSError, PermissionError):
                continue

        # Clean up temporary HTML files
        for temp_file in temp_dir.glob("tmp*.html"):
            try:
                if temp_file.stat().st_mtime < cutoff_time.timestamp():
                    temp_file.unlink()
                    cleaned_count += 1
            except (OSError, PermissionError):
                continue

        logger.info(f"Cleaned up {cleaned_count} temporary files")

        return {"success": True, "cleaned_files": cleaned_count}

    except Exception as e:
        logger.error(f"Temp file cleanup failed: {str(e)}")
        raise


@task
async def update_resume_statistics():
    """Update resume statistics and analytics."""
    try:
        async with get_db_context() as db:
            from sqlalchemy import func

            # Calculate various statistics
            stats = {}

            # Total resumes
            total_resumes = await db.scalar(
                func.count(Resume.id).filter(Resume.deleted_at.is_(None))
            )
            stats["total_resumes"] = total_resumes

            # Active resumes
            active_resumes = await db.scalar(
                func.count(Resume.id).filter(
                    Resume.status == "active",
                    Resume.deleted_at.is_(None)
                )
            )
            stats["active_resumes"] = active_resumes

            # Average scores
            avg_overall_score = await db.scalar(
                func.avg(Resume.overall_score).filter(
                    Resume.overall_score.isnot(None),
                    Resume.deleted_at.is_(None)
                )
            )
            stats["average_overall_score"] = float(avg_overall_score) if avg_overall_score else 0.0

            # Store stats in cache
            if cache_manager:
                await cache_manager.set("resume_statistics", stats, ttl=3600)  # 1 hour

            logger.info("Resume statistics updated successfully")

            return {"success": True, "statistics": stats}

    except Exception as e:
        logger.error(f"Statistics update failed: {str(e)}")
        raise


@priority_task
async def send_analysis_completion_notification(
        user_id: str,
        total_items: int,
        completed: int,
        failed: int
):
    """
    Send notification when bulk analysis is completed.

    Args:
        user_id: User ID to notify
        total_items: Total number of items processed
        completed: Number of successfully completed items
        failed: Number of failed items
    """
    try:
        # This would integrate with your notification system
        # For now, we'll just log the notification

        message = f"Analysis completed: {completed}/{total_items} successful"
        if failed > 0:
            message += f", {failed} failed"

        logger.info(f"Notification for user {user_id}: {message}")

        # In a real implementation, you would:
        # 1. Send email notification
        # 2. Send push notification
        # 3. Create in-app notification
        # 4. Send WebSocket message for real-time updates

        return {"success": True, "message": message, "user_id": user_id}

    except Exception as e:
        logger.error(f"Notification sending failed for user {user_id}: {str(e)}")
        raise


@task
async def generate_resume_variants_task(
        resume_id: str,
        job_descriptions: List[Dict[str, Any]],
        customization_level: int = 3
):
    """
    Generate multiple resume variants optimized for different jobs.

    Args:
        resume_id: Base resume UUID as string
        job_descriptions: List of job descriptions with details
        customization_level: Level of customization (1-5)

    Returns:
        Dict with variant generation results
    """
    try:
        total_jobs = len(job_descriptions)
        completed = 0
        failed = 0
        variants = []

        current_task.update_state(
            state='PROGRESS',
            meta={
                'progress': 0,
                'status': f'Starting generation of {total_jobs} resume variants',
                'completed': completed,
                'failed': failed,
                'total': total_jobs
            }
        )

        async with get_db_context() as db:
            # Get base resume
            base_resume = await db.get(Resume, uuid.UUID(resume_id))
            if not base_resume:
                raise AppException("Base resume not found", status_code=404)

            # Initialize AI service
            ai_service = AIService()

            for i, job_desc in enumerate(job_descriptions):
                try:
                    # Create optimized variant
                    # This would use AI to customize the resume for each job
                    variant_title = f"{base_resume.title} - {job_desc.get('company', 'Company')} Variant"

                    # Create new resume variant
                    variant = Resume(
                        user_id=base_resume.user_id,
                        title=variant_title,
                        description=f"Optimized for {job_desc.get('job_title', 'position')} at {job_desc.get('company', 'company')}",
                        template_id=base_resume.template_id,
                        personal_info=base_resume.personal_info,
                        work_experience=base_resume.work_experience,
                        education=base_resume.education,
                        skills=base_resume.skills,
                        projects=base_resume.projects,
                        certifications=base_resume.certifications,
                        languages=base_resume.languages,
                        awards=base_resume.awards,
                        references=base_resume.references,
                        custom_sections=base_resume.custom_sections,
                        target_job_title=job_desc.get('job_title'),
                        target_industry=job_desc.get('industry'),
                        keywords=job_desc.get('keywords', []),
                        status="draft"
                    )

                    db.add(variant)
                    await db.flush()  # Get ID without committing

                    variants.append({
                        "resume_id": str(variant.id),
                        "job_title": job_desc.get('job_title'),
                        "company": job_desc.get('company'),
                        "success": True
                    })
                    completed += 1

                except Exception as e:
                    logger.error(f"Variant generation failed for job {i}: {str(e)}")
                    variants.append({
                        "job_title": job_desc.get('job_title'),
                        "company": job_desc.get('company'),
                        "success": False,
                        "error": str(e)
                    })
                    failed += 1

                # Update progress
                progress = int((i + 1) / total_jobs * 100)
                current_task.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'status': f'Generated {i + 1}/{total_jobs} variants',
                        'completed': completed,
                        'failed': failed,
                        'total': total_jobs
                    }
                )

            # Commit all successful variants
            await db.commit()

        logger.info(f"Resume variant generation completed: {completed} successful, {failed} failed")

        return {
            "success": True,
            "base_resume_id": resume_id,
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": failed,
            "variants": variants
        }

    except Exception as e:
        logger.error(f"Resume variant generation failed: {str(e)}")
        current_task.update_state(
            state='FAILURE',
            meta={'error': str(e), 'resume_id': resume_id}
        )
        raise


# Periodic tasks configuration for Celery Beat
@task
async def daily_maintenance_task():
    """Daily maintenance task that runs cleanup and updates."""
    try:
        logger.info("Starting daily maintenance")

        # Run cleanup tasks
        await cleanup_expired_files.delay()
        await cleanup_temp_files.delay()

        # Update statistics
        await update_resume_statistics.delay()

        logger.info("Daily maintenance completed")

        return {"success": True, "message": "Daily maintenance completed"}

    except Exception as e:
        logger.error(f"Daily maintenance failed: {str(e)}")
        raise