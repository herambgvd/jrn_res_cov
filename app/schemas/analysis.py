"""
Pydantic schemas for Analysis API endpoints.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator, ConfigDict

from app.models.analysis import AnalysisType, SuggestionType, SuggestionPriority


# Base analysis schemas
class AnalysisSessionBase(BaseModel):
    """Base analysis session schema."""
    session_name: Optional[str] = Field(None, max_length=200, description="Session name")
    analysis_types: List[str] = Field(..., min_items=1, description="Types of analysis to perform")
    target_type: str = Field(..., description="Target document type (resume, cover_letter)")
    target_ids: List[str] = Field(..., min_items=1, description="Target document IDs")


class AnalysisSessionCreate(AnalysisSessionBase):
    """Schema for creating an analysis session."""
    job_description_id: Optional[uuid.UUID] = Field(None, description="Job description for context")
    job_title: Optional[str] = Field(None, max_length=200, description="Job title for context")
    company_name: Optional[str] = Field(None, max_length=200, description="Company name for context")

    @validator("analysis_types")
    def validate_analysis_types(cls, v):
        """Validate analysis types are valid."""
        valid_types = [at.value for at in AnalysisType]
        for analysis_type in v:
            if analysis_type not in valid_types:
                raise ValueError(f"Invalid analysis type: {analysis_type}")
        return v


class AnalysisSessionResponse(AnalysisSessionBase):
    """Schema for analysis session response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    job_description_id: Optional[uuid.UUID] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    status: str = "pending"
    progress: int = 0
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    overall_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None
    total_suggestions: int = 0
    critical_issues: int = 0
    processing_time: Optional[float] = None
    ai_model_used: Optional[str] = None
    error_details: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# Analysis result schemas
class AnalysisResultBase(BaseModel):
    """Base analysis result schema."""
    target_type: str = Field(..., description="Target document type")
    target_id: uuid.UUID = Field(..., description="Target document ID")
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    overall_score: float = Field(..., ge=0, le=100, description="Overall analysis score")
    sub_scores: Dict[str, float] = Field(default_factory=dict, description="Sub-category scores")
    analysis_data: Dict[str, Any] = Field(..., description="Detailed analysis data")


class AnalysisResultCreate(AnalysisResultBase):
    """Schema for creating an analysis result."""
    session_id: Optional[uuid.UUID] = Field(None, description="Associated session ID")
    analysis_version: str = Field("1.0", description="Analysis version")
    raw_results: Optional[Dict[str, Any]] = Field(None, description="Raw analysis results")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    ai_model_used: Optional[str] = Field(None, description="AI model used for analysis")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence in results")


class AnalysisResultResponse(AnalysisResultBase):
    """Schema for analysis result response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: Optional[uuid.UUID] = None
    user_id: uuid.UUID
    analysis_version: str = "1.0"
    raw_results: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    ai_model_used: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: datetime
    suggestions: List["SuggestionResponse"] = Field(default_factory=list)


class AnalysisResultListResponse(BaseModel):
    """Schema for analysis result list response."""
    results: List[AnalysisResultResponse]
    total: int
    page: int
    size: int
    pages: int


# Suggestion schemas
class SuggestionBase(BaseModel):
    """Base suggestion schema."""
    type: SuggestionType = Field(..., description="Type of suggestion")
    priority: SuggestionPriority = Field(..., description="Suggestion priority")
    title: str = Field(..., min_length=1, max_length=200, description="Suggestion title")
    description: str = Field(..., min_length=1, description="Detailed description")
    before_text: Optional[str] = Field(None, description="Text before improvement")
    suggested_text: Optional[str] = Field(None, description="Suggested improvement text")


class SuggestionCreate(SuggestionBase):
    """Schema for creating a suggestion."""
    section: Optional[str] = Field(None, max_length=100, description="Document section")
    field_path: Optional[str] = Field(None, max_length=200, description="Field path in document")
    line_number: Optional[int] = Field(None, ge=1, description="Line number")
    character_position: Optional[int] = Field(None, ge=0, description="Character position")
    impact_score: Optional[float] = Field(None, ge=0, le=1, description="Impact score")
    reasoning: Optional[str] = Field(None, description="Reasoning for suggestion")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in suggestion")
    implementation_effort: Optional[str] = Field(None, description="Implementation effort level")
    estimated_time_minutes: Optional[int] = Field(None, ge=1, description="Estimated time to implement")
    requires_manual_review: bool = Field(False, description="Requires manual review")


class SuggestionUpdate(BaseModel):
    """Schema for updating a suggestion."""
    is_applied: Optional[bool] = None
    is_dismissed: Optional[bool] = None
    user_feedback: Optional[str] = None


class SuggestionResponse(SuggestionBase):
    """Schema for suggestion response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    analysis_result_id: uuid.UUID
    section: Optional[str] = None
    field_path: Optional[str] = None
    line_number: Optional[int] = None
    character_position: Optional[int] = None
    impact_score: Optional[float] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    implementation_effort: Optional[str] = None
    estimated_time_minutes: Optional[int] = None
    requires_manual_review: bool = False
    is_applied: bool = False
    is_dismissed: bool = False
    user_feedback: Optional[str] = None
    applied_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None
    created_at: datetime


class SuggestionListResponse(BaseModel):
    """Schema for suggestion list response."""
    suggestions: List[SuggestionResponse]
    total: int
    page: int
    size: int
    pages: int


# Keyword analysis schemas
class KeywordAnalysisBase(BaseModel):
    """Base keyword analysis schema."""
    target_type: str = Field(..., description="Target document type")
    target_id: uuid.UUID = Field(..., description="Target document ID")
    extracted_keywords: List[str] = Field(..., description="Keywords found in document")
    missing_keywords: List[str] = Field(..., description="Keywords missing from document")
    matching_keywords: List[str] = Field(..., description="Keywords that match job requirements")
    suggested_keywords: List[str] = Field(..., description="Suggested keywords to add")


class KeywordAnalysisCreate(KeywordAnalysisBase):
    """Schema for creating keyword analysis."""
    job_description_id: Optional[uuid.UUID] = Field(None, description="Job description for context")
    job_title: Optional[str] = Field(None, max_length=200, description="Job title")
    industry: Optional[str] = Field(None, max_length=100, description="Industry context")
    technical_keywords: Optional[List[str]] = Field(default_factory=list)
    soft_skills_keywords: Optional[List[str]] = Field(default_factory=list)
    industry_keywords: Optional[List[str]] = Field(default_factory=list)
    action_keywords: Optional[List[str]] = Field(default_factory=list)
    keyword_frequency: Optional[Dict[str, int]] = Field(default_factory=dict)
    keyword_positions: Optional[Dict[str, List[int]]] = Field(default_factory=dict)
    keyword_importance: Optional[Dict[str, float]] = Field(default_factory=dict)


class KeywordAnalysisResponse(KeywordAnalysisBase):
    """Schema for keyword analysis response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    job_description_id: Optional[uuid.UUID] = None
    job_title: Optional[str] = None
    industry: Optional[str] = None
    technical_keywords: Optional[List[str]] = None
    soft_skills_keywords: Optional[List[str]] = None
    industry_keywords: Optional[List[str]] = None
    action_keywords: Optional[List[str]] = None
    keyword_density_score: float
    keyword_relevance_score: float
    keyword_diversity_score: float
    overall_keyword_score: float
    keyword_frequency: Optional[Dict[str, int]] = None
    keyword_positions: Optional[Dict[str, List[int]]] = None
    keyword_importance: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    ai_model_used: Optional[str] = None
    created_at: datetime


# ATS analysis schemas
class ATSAnalysisBase(BaseModel):
    """Base ATS analysis schema."""
    target_type: str = Field(..., description="Target document type")
    target_id: uuid.UUID = Field(..., description="Target document ID")
    parsing_score: float = Field(..., ge=0, le=100, description="ATS parsing score")
    formatting_score: float = Field(..., ge=0, le=100, description="Formatting score")
    readability_score: float = Field(..., ge=0, le=100, description="Readability score")
    overall_ats_score: float = Field(..., ge=0, le=100, description="Overall ATS score")


class ATSAnalysisCreate(ATSAnalysisBase):
    """Schema for creating ATS analysis."""
    parseable_sections: List[str] = Field(default_factory=list)
    problematic_sections: List[str] = Field(default_factory=list)
    formatting_issues: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_text: str = Field(..., description="Text extracted by ATS")
    text_extraction_quality: float = Field(..., ge=0, le=100)
    font_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict)
    layout_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict)
    image_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict)
    contact_info_extracted: Optional[Dict[str, Any]] = Field(default_factory=dict)
    contact_extraction_confidence: float = Field(..., ge=0, le=100)


class ATSAnalysisResponse(ATSAnalysisBase):
    """Schema for ATS analysis response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    parseable_sections: List[str]
    problematic_sections: List[str]
    formatting_issues: List[Dict[str, Any]]
    extracted_text: str
    text_extraction_quality: float
    font_analysis: Optional[Dict[str, Any]] = None
    layout_analysis: Optional[Dict[str, Any]] = None
    image_analysis: Optional[Dict[str, Any]] = None
    contact_info_extracted: Optional[Dict[str, Any]] = None
    contact_extraction_confidence: float
    processing_time: Optional[float] = None
    ats_simulator_used: Optional[str] = None
    created_at: datetime


# Competitive analysis schemas
class CompetitiveAnalysisBase(BaseModel):
    """Base competitive analysis schema."""
    target_type: str = Field(..., description="Target document type")
    target_id: uuid.UUID = Field(..., description="Target document ID")
    industry: str = Field(..., max_length=100, description="Industry for benchmarking")
    job_level: str = Field(..., max_length=50, description="Job level for benchmarking")
    content_competitiveness: float = Field(..., ge=0, le=100)
    keyword_competitiveness: float = Field(..., ge=0, le=100)
    format_competitiveness: float = Field(..., ge=0, le=100)
    overall_competitiveness: float = Field(..., ge=0, le=100)


class CompetitiveAnalysisCreate(CompetitiveAnalysisBase):
    """Schema for creating competitive analysis."""
    experience_years: Optional[int] = Field(None, ge=0, description="Years of experience")
    content_percentile: float = Field(..., ge=0, le=100)
    keyword_percentile: float = Field(..., ge=0, le=100)
    format_percentile: float = Field(..., ge=0, le=100)
    overall_percentile: float = Field(..., ge=0, le=100)
    industry_averages: Dict[str, float] = Field(default_factory=dict)
    top_performer_examples: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    benchmark_sample_size: int = Field(..., ge=1)
    confidence_interval: float = Field(..., ge=0, le=1)


class CompetitiveAnalysisResponse(CompetitiveAnalysisBase):
    """Schema for competitive analysis response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    experience_years: Optional[int] = None
    content_percentile: float
    keyword_percentile: float
    format_percentile: float
    overall_percentile: float
    industry_averages: Dict[str, float]
    top_performer_examples: List[Dict[str, Any]]
    improvement_opportunities: List[Dict[str, Any]]
    benchmark_sample_size: int
    confidence_interval: float
    processing_time: Optional[float] = None
    benchmark_data_version: str = "1.0"
    created_at: datetime


# Analysis request schemas
class AnalysisRequest(BaseModel):
    """Schema for general analysis request."""
    target_type: str = Field(..., description="Target document type (resume, cover_letter)")
    target_ids: List[uuid.UUID] = Field(..., min_items=1, description="Document IDs to analyze")
    analysis_types: List[AnalysisType] = Field(..., min_items=1, description="Types of analysis to perform")
    job_description: Optional[str] = Field(None, description="Job description for context")
    job_description_id: Optional[uuid.UUID] = Field(None, description="Job description ID")
    job_title: Optional[str] = Field(None, max_length=200, description="Job title")
    company_name: Optional[str] = Field(None, max_length=200, description="Company name")
    industry: Optional[str] = Field(None, max_length=100, description="Industry")
    include_suggestions: bool = Field(True, description="Include improvement suggestions")
    include_competitive_analysis: bool = Field(False, description="Include competitive benchmarking")
    priority: str = Field("normal", description="Analysis priority (low, normal, high)")


class BatchAnalysisRequest(BaseModel):
    """Schema for batch analysis request."""
    analysis_requests: List[AnalysisRequest] = Field(..., min_items=1, max_items=50)
    session_name: Optional[str] = Field(None, max_length=200, description="Batch session name")
    notify_on_completion: bool = Field(True, description="Send notification when complete")
    email_notification: Optional[str] = Field(None, description="Email for notifications")


class AnalysisComparisonRequest(BaseModel):
    """Schema for comparing analysis results."""
    target_ids: List[uuid.UUID] = Field(..., min_items=2, max_items=10, description="Documents to compare")
    comparison_criteria: List[str] = Field(default=["overall_score", "ats_score", "keyword_score"],
                                           description="Criteria to compare")
    job_context: Optional[Dict[str, Any]] = Field(None, description="Job context for comparison")
    include_suggestions: bool = Field(True, description="Include comparison suggestions")


class AnalysisComparisonResponse(BaseModel):
    """Schema for analysis comparison response."""
    comparison_results: Dict[str, Dict[str, Any]] = Field(..., description="Comparison results by document ID")
    best_performers: Dict[str, uuid.UUID] = Field(..., description="Best performers by criteria")
    improvement_opportunities: List[Dict[str, Any]] = Field(..., description="Improvement opportunities")
    summary: str = Field(..., description="Comparison summary")
    recommendations: List[str] = Field(..., description="Actionable recommendations")


# Analysis statistics schemas
class AnalysisStatsResponse(BaseModel):
    """Schema for analysis statistics response."""
    total_analyses: int
    analyses_by_type: Dict[str, int]
    average_scores: Dict[str, float]
    total_suggestions: int
    applied_suggestions: int
    dismissed_suggestions: int
    suggestions_by_type: Dict[str, int]
    suggestions_by_priority: Dict[str, int]
    processing_time_stats: Dict[str, float]
    recent_analyses: List[Dict[str, Any]] = Field(default_factory=list)


# Analysis history schemas
class AnalysisHistoryResponse(BaseModel):
    """Schema for analysis history response."""
    analyses: List[AnalysisResultResponse]
    total: int
    date_range: Dict[str, datetime]
    score_trends: Dict[str, List[Dict[str, Any]]]
    improvement_over_time: Dict[str, float]


# Analysis export schemas
class AnalysisExportRequest(BaseModel):
    """Schema for exporting analysis results."""
    analysis_ids: List[uuid.UUID] = Field(..., min_items=1, description="Analysis IDs to export")
    export_format: str = Field("pdf", description="Export format (pdf, excel, csv)")
    include_suggestions: bool = Field(True, description="Include suggestions in export")
    include_charts: bool = Field(True, description="Include charts and visualizations")
    template_id: Optional[uuid.UUID] = Field(None, description="Custom report template")
    custom_branding: Optional[Dict[str, Any]] = Field(None, description="Custom branding options")


class AnalysisExportResponse(BaseModel):
    """Schema for analysis export response."""
    export_url: str = Field(..., description="URL to download export")
    file_name: str = Field(..., description="Export file name")
    file_size: int = Field(..., description="File size in bytes")
    export_format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="Export expiration date")
    included_analyses: List[uuid.UUID] = Field(..., description="Included analysis IDs")


# Real-time analysis schemas
class RealTimeAnalysisRequest(BaseModel):
    """Schema for real-time analysis request."""
    content: str = Field(..., min_length=10, description="Content to analyze")
    content_type: str = Field(..., description="Content type (resume_section, cover_letter_paragraph)")
    analysis_types: List[str] = Field(default=["grammar", "keyword", "tone"], description="Types of real-time analysis")
    job_context: Optional[Dict[str, Any]] = Field(None, description="Job context for analysis")
    previous_suggestions: Optional[List[uuid.UUID]] = Field(None, description="Previous suggestion IDs to track")


class RealTimeAnalysisResponse(BaseModel):
    """Schema for real-time analysis response."""
    content_score: float = Field(..., ge=0, le=100, description="Content quality score")
    issues: List[Dict[str, Any]] = Field(..., description="Issues found in content")
    suggestions: List[Dict[str, Any]] = Field(..., description="Real-time suggestions")
    improvements: List[Dict[str, Any]] = Field(..., description="Suggested improvements")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    processing_time: float = Field(..., description="Processing time in seconds")


# Analysis feedback schemas
class AnalysisFeedback(BaseModel):
    """Schema for analysis feedback."""
    analysis_id: uuid.UUID = Field(..., description="Analysis result ID")
    accuracy_rating: int = Field(..., ge=1, le=5, description="Accuracy rating (1-5)")
    usefulness_rating: int = Field(..., ge=1, le=5, description="Usefulness rating (1-5)")
    suggestion_quality_rating: int = Field(..., ge=1, le=5, description="Suggestion quality rating (1-5)")
    feedback_text: Optional[str] = Field(None, description="Detailed feedback")
    false_positives: Optional[List[uuid.UUID]] = Field(None, description="Suggestion IDs that were false positives")
    missed_issues: Optional[List[str]] = Field(None, description="Issues that were missed")
    would_recommend: Optional[bool] = Field(None, description="Would recommend the analysis")


class AnalysisFeedbackResponse(BaseModel):
    """Schema for analysis feedback response."""
    id: uuid.UUID
    analysis_id: uuid.UUID
    user_id: uuid.UUID
    accuracy_rating: int
    usefulness_rating: int
    suggestion_quality_rating: int
    feedback_text: Optional[str] = None
    false_positives: Optional[List[uuid.UUID]] = None
    missed_issues: Optional[List[str]] = None
    would_recommend: Optional[bool] = None
    created_at: datetime


# Update the AnalysisResultResponse to include suggestions properly
AnalysisResultResponse.model_rebuild()