"""
Analysis API endpoints for resume and cover letter analysis.
"""

import uuid
# Import required modules
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks

from app.api.deps import get_current_user, get_analysis_service
from app.models.analysis import AnalysisType, SuggestionType, SuggestionPriority
from app.schemas.analysis import (
    AnalysisSessionCreate, AnalysisSessionResponse, AnalysisResultResponse,
    AnalysisResultListResponse, SuggestionResponse, SuggestionListResponse,
    AnalysisRequest, AnalysisComparisonRequest,
    AnalysisComparisonResponse, RealTimeAnalysisRequest, RealTimeAnalysisResponse,
    AnalysisExportRequest, AnalysisExportResponse, AnalysisStatsResponse,
    AnalysisHistoryResponse, AnalysisFeedback, AnalysisFeedbackResponse,
    BatchAnalysisRequest
)
from app.schemas.common import ResponseWrapper, TaskStatus
from app.schemas.common import UserContext
from app.services.analysis_service import AnalysisService
from app.utils.exceptions import AppException

router = APIRouter()


@router.post("/sessions", response_model=ResponseWrapper[AnalysisSessionResponse], status_code=201)
async def create_analysis_session(
        session_data: AnalysisSessionCreate,
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Create a new analysis session for batch processing.

    - **session_name**: Optional name for the session
    - **analysis_types**: Types of analysis to perform
    - **target_type**: Type of documents (resume, cover_letter)
    - **target_ids**: List of document IDs to analyze
    - **job_description_id**: Optional job description for context
    """
    try:
        session = await analysis_service.create_analysis_session(
            current_user.user_id, session_data
        )
        return ResponseWrapper(
            data=session,
            message="Analysis session created successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sessions/{session_id}", response_model=ResponseWrapper[AnalysisSessionResponse])
async def get_analysis_session(
        session_id: uuid.UUID = Path(..., description="Analysis session ID"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get analysis session status and results.

    - **session_id**: UUID of the analysis session
    """
    try:
        session = await analysis_service.get_analysis_session(session_id, current_user.user_id)
        return ResponseWrapper(
            data=session,
            message="Analysis session retrieved successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=ResponseWrapper[List[AnalysisResultResponse]])
async def analyze_documents(
        analysis_request: AnalysisRequest,
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Analyze documents directly without creating a session.

    - **target_type**: Type of documents (resume, cover_letter)
    - **target_ids**: List of document IDs to analyze
    - **analysis_types**: Types of analysis to perform
    - **job_description**: Optional job description for context
    - **include_suggestions**: Whether to include improvement suggestions
    """
    try:
        results = await analysis_service.analyze_document(current_user.user_id, analysis_request)
        return ResponseWrapper(
            data=results,
            message="Documents analyzed successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/batch", response_model=ResponseWrapper[TaskStatus])
async def batch_analyze_documents(
        batch_request: BatchAnalysisRequest,
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Analyze multiple document sets in batch.

    - **analysis_requests**: List of analysis requests
    - **session_name**: Optional batch session name
    - **notify_on_completion**: Send notification when complete
    """
    try:
        if len(batch_request.analysis_requests) > 50:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 analysis requests allowed in batch operation"
            )

        # This would queue a background task for batch analysis
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


@router.get("/results", response_model=ResponseWrapper[AnalysisResultListResponse])
async def get_analysis_results(
        target_type: Optional[str] = Query(None, description="Filter by document type"),
        target_id: Optional[uuid.UUID] = Query(None, description="Filter by document ID"),
        analysis_type: Optional[AnalysisType] = Query(None, description="Filter by analysis type"),
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Page size"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get analysis results with pagination and filtering.

    - **target_type**: Filter by document type (resume, cover_letter)
    - **target_id**: Filter by specific document ID
    - **analysis_type**: Filter by analysis type
    - **page**: Page number
    - **size**: Items per page
    """
    try:
        results = await analysis_service.get_analysis_results(
            user_id=current_user.user_id,
            target_type=target_type,
            target_id=target_id,
            analysis_type=analysis_type,
            page=page,
            size=size
        )
        return ResponseWrapper(
            data=results,
            message="Analysis results retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}", response_model=ResponseWrapper[AnalysisResultResponse])
async def get_analysis_result(
        result_id: uuid.UUID = Path(..., description="Analysis result ID"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get a specific analysis result by ID.

    - **result_id**: UUID of the analysis result
    """
    try:
        # This would be implemented in the service
        raise HTTPException(status_code=501, detail="Not implemented yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ResponseWrapper[AnalysisComparisonResponse])
async def compare_documents(
        comparison_request: AnalysisComparisonRequest,
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Compare analysis results of multiple documents.

    - **target_ids**: List of document IDs to compare (2-10 documents)
    - **comparison_criteria**: Criteria to compare
    - **job_context**: Optional job context for comparison
    - **include_suggestions**: Include comparison suggestions
    """
    try:
        if len(comparison_request.target_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 documents required for comparison"
            )

        if len(comparison_request.target_ids) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 documents allowed for comparison"
            )

        result = await analysis_service.compare_documents(
            user_id=current_user.user_id,
            target_ids=comparison_request.target_ids,
            comparison_criteria=comparison_request.comparison_criteria
        )

        return ResponseWrapper(
            data=result,
            message="Document comparison completed successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/real-time", response_model=ResponseWrapper[RealTimeAnalysisResponse])
async def real_time_analysis(
        analysis_request: RealTimeAnalysisRequest,
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform real-time analysis on content.

    - **content**: Content to analyze
    - **content_type**: Type of content (resume_section, cover_letter_paragraph)
    - **analysis_types**: Types of real-time analysis
    - **job_context**: Optional job context
    """
    try:
        result = await analysis_service.real_time_analysis(current_user.user_id, analysis_request)
        return ResponseWrapper(
            data=result,
            message="Real-time analysis completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Suggestions endpoints
@router.get("/suggestions", response_model=ResponseWrapper[SuggestionListResponse])
async def get_suggestions(
        analysis_result_id: Optional[uuid.UUID] = Query(None, description="Filter by analysis result"),
        priority: Optional[SuggestionPriority] = Query(None, description="Filter by priority"),
        type: Optional[SuggestionType] = Query(None, description="Filter by suggestion type"),
        is_applied: Optional[bool] = Query(None, description="Filter by applied status"),
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(50, ge=1, le=100, description="Page size"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get suggestions with filtering and pagination.

    - **analysis_result_id**: Filter by specific analysis result
    - **priority**: Filter by suggestion priority
    - **type**: Filter by suggestion type
    - **is_applied**: Filter by applied status
    - **page**: Page number
    - **size**: Items per page
    """
    try:
        suggestions = await analysis_service.get_suggestions(
            user_id=current_user.user_id,
            analysis_result_id=analysis_result_id,
            priority=priority,
            type=type,
            is_applied=is_applied,
            page=page,
            size=size
        )
        return ResponseWrapper(
            data=suggestions,
            message="Suggestions retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggestions/{suggestion_id}/apply", response_model=ResponseWrapper[SuggestionResponse])
async def apply_suggestion(
        suggestion_id: uuid.UUID = Path(..., description="Suggestion ID"),
        user_feedback: Optional[str] = Query(None, description="Optional user feedback"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Apply a suggestion.

    - **suggestion_id**: UUID of the suggestion to apply
    - **user_feedback**: Optional feedback about the suggestion
    """
    try:
        suggestion = await analysis_service.apply_suggestion(
            suggestion_id, current_user.user_id, user_feedback
        )
        return ResponseWrapper(
            data=suggestion,
            message="Suggestion applied successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/suggestions/{suggestion_id}/dismiss", response_model=ResponseWrapper[SuggestionResponse])
async def dismiss_suggestion(
        suggestion_id: uuid.UUID = Path(..., description="Suggestion ID"),
        user_feedback: Optional[str] = Query(None, description="Optional dismissal reason"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Dismiss a suggestion.

    - **suggestion_id**: UUID of the suggestion to dismiss
    - **user_feedback**: Optional reason for dismissal
    """
    try:
        suggestion = await analysis_service.dismiss_suggestion(
            suggestion_id, current_user.user_id, user_feedback
        )
        return ResponseWrapper(
            data=suggestion,
            message="Suggestion dismissed successfully"
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Specialized analysis endpoints
@router.get("/keyword/{target_id}", response_model=ResponseWrapper[Dict[str, Any]])
async def get_keyword_analysis(
        target_id: uuid.UUID = Path(..., description="Document ID"),
        job_description_id: Optional[uuid.UUID] = Query(None, description="Job description for context"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get keyword analysis for a document.

    - **target_id**: UUID of the document
    - **job_description_id**: Optional job description for keyword matching
    """
    try:
        analysis = await analysis_service.get_keyword_analysis(
            current_user.user_id, target_id, job_description_id
        )

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="No keyword analysis found for this document"
            )

        return ResponseWrapper(
            data=analysis,
            message="Keyword analysis retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ats/{target_id}", response_model=ResponseWrapper[Dict[str, Any]])
async def get_ats_analysis(
        target_id: uuid.UUID = Path(..., description="Document ID"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get ATS (Applicant Tracking System) analysis for a document.

    - **target_id**: UUID of the document
    """
    try:
        analysis = await analysis_service.get_ats_analysis(current_user.user_id, target_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="No ATS analysis found for this document"
            )

        return ResponseWrapper(
            data=analysis,
            message="ATS analysis retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitive/{target_id}", response_model=ResponseWrapper[Dict[str, Any]])
async def get_competitive_analysis(
        target_id: uuid.UUID = Path(..., description="Document ID"),
        industry: str = Query(..., description="Industry for benchmarking"),
        job_level: str = Query(..., description="Job level for benchmarking"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get competitive analysis for a document.

    - **target_id**: UUID of the document
    - **industry**: Industry for benchmarking
    - **job_level**: Job level for benchmarking
    """
    try:
        analysis = await analysis_service.get_competitive_analysis(
            current_user.user_id, target_id, industry, job_level
        )

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="No competitive analysis found for this document"
            )

        return ResponseWrapper(
            data=analysis,
            message="Competitive analysis retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export and reporting
@router.post("/export", response_model=ResponseWrapper[AnalysisExportResponse])
async def export_analysis_results(
        export_request: AnalysisExportRequest,
        background_tasks: BackgroundTasks,
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Export analysis results to various formats.

    - **analysis_ids**: List of analysis IDs to export
    - **export_format**: Export format (pdf, excel, csv)
    - **include_suggestions**: Include suggestions in export
    - **include_charts**: Include charts and visualizations
    """
    try:
        # This would be implemented to generate exports
        export_result = {
            "export_url": f"/downloads/analysis-export-{uuid.uuid4()}.{export_request.export_format}",
            "file_name": f"analysis-export-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{export_request.export_format}",
            "file_size": 2048000,  # Placeholder
            "export_format": export_request.export_format,
            "expires_at": datetime.utcnow() + timedelta(hours=24),
            "included_analyses": export_request.analysis_ids
        }

        return ResponseWrapper(
            data=export_result,
            message="Analysis export initiated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Statistics and history
@router.get("/stats/overview", response_model=ResponseWrapper[AnalysisStatsResponse])
async def get_analysis_statistics(
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get user's analysis statistics and overview.
    """
    try:
        stats = {
            "total_analyses": 0,
            "analyses_by_type": {},
            "average_scores": {},
            "total_suggestions": 0,
            "applied_suggestions": 0,
            "dismissed_suggestions": 0,
            "suggestions_by_type": {},
            "suggestions_by_priority": {},
            "processing_time_stats": {},
            "recent_analyses": []
        }

        return ResponseWrapper(
            data=stats,
            message="Analysis statistics retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=ResponseWrapper[AnalysisHistoryResponse])
async def get_analysis_history(
        target_id: Optional[uuid.UUID] = Query(None, description="Filter by document ID"),
        days: int = Query(30, ge=1, le=365, description="Number of days of history"),
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get analysis history with trends and improvements over time.

    - **target_id**: Optional filter by specific document
    - **days**: Number of days of history to retrieve
    """
    try:
        # This would be implemented to show analysis trends
        history = {
            "analyses": [],
            "total": 0,
            "date_range": {
                "start": datetime.utcnow() - timedelta(days=days),
                "end": datetime.utcnow()
            },
            "score_trends": {},
            "improvement_over_time": {}
        }

        return ResponseWrapper(
            data=history,
            message="Analysis history retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Feedback endpoints
@router.post("/feedback", response_model=ResponseWrapper[AnalysisFeedbackResponse])
async def submit_analysis_feedback(
        feedback: AnalysisFeedback,
        current_user: UserContext = Depends(get_current_user),
        analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Submit feedback on analysis quality.

    - **analysis_id**: Analysis result ID
    - **accuracy_rating**: Accuracy rating (1-5)
    - **usefulness_rating**: Usefulness rating (1-5)
    - **suggestion_quality_rating**: Suggestion quality rating (1-5)
    - **feedback_text**: Detailed feedback
    """
    try:
        # This would be implemented to store and process feedback
        feedback_response = {
            "id": uuid.uuid4(),
            "analysis_id": feedback.analysis_id,
            "user_id": current_user.user_id,
            "accuracy_rating": feedback.accuracy_rating,
            "usefulness_rating": feedback.usefulness_rating,
            "suggestion_quality_rating": feedback.suggestion_quality_rating,
            "feedback_text": feedback.feedback_text,
            "created_at": datetime.utcnow()
        }

        return ResponseWrapper(
            data=feedback_response,
            message="Feedback submitted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
