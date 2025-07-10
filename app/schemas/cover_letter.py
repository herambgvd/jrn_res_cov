"""
Pydantic schemas for Cover Letter API endpoints.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator, ConfigDict

from app.models.cover_letter import CoverLetterStatus, CoverLetterType, CoverLetterTone


# Base schemas
class CoverLetterBase(BaseModel):
    """Base cover letter schema with common fields."""
    title: str = Field(..., min_length=1, max_length=200, description="Cover letter title")
    description: Optional[str] = Field(None, description="Cover letter description")
    type: CoverLetterType = Field(CoverLetterType.STANDARD, description="Cover letter type")
    tone: CoverLetterTone = Field(CoverLetterTone.PROFESSIONAL, description="Cover letter tone")
    content: Dict[str, Any] = Field(..., description="Cover letter content")
    job_title: Optional[str] = Field(None, max_length=200, description="Target job title")
    company_name: Optional[str] = Field(None, max_length=200, description="Target company name")
    is_public: bool = Field(False, description="Whether cover letter is publicly visible")


class CoverLetterCreate(CoverLetterBase):
    """Schema for creating a cover letter."""
    resume_id: Optional[uuid.UUID] = Field(None, description="Associated resume ID")
    template_id: Optional[uuid.UUID] = Field(None, description="Template ID to use")
    job_description: Optional[str] = Field(None, description="Job description for customization")
    job_requirements: Optional[List[str]] = Field(default_factory=list, description="Specific job requirements")
    company_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Company information")
    hiring_manager_name: Optional[str] = Field(None, max_length=200, description="Hiring manager name")
    hiring_manager_title: Optional[str] = Field(None, max_length=200, description="Hiring manager title")
    company_address: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Company address")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Target keywords")
    key_achievements: Optional[List[str]] = Field(default_factory=list, description="Key achievements to highlight")
    skills_highlighted: Optional[List[str]] = Field(default_factory=list, description="Skills to highlight")

    # AI generation settings
    ai_generated: bool = Field(False, description="Whether to use AI generation")
    ai_prompt: Optional[str] = Field(None, description="Custom AI prompt")
    generation_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="AI generation settings")
    personalization_level: Optional[int] = Field(None, ge=1, le=5, description="Personalization level (1-5)")

    @validator("content")
    def validate_content(cls, v):
        """Validate content has required structure."""
        if not isinstance(v, dict):
            raise ValueError("Content must be a dictionary")
        return v


class CoverLetterUpdate(BaseModel):
    """Schema for updating a cover letter."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[CoverLetterStatus] = None
    type: Optional[CoverLetterType] = None
    tone: Optional[CoverLetterTone] = None
    content: Optional[Dict[str, Any]] = None
    opening_paragraph: Optional[str] = None
    body_paragraphs: Optional[List[str]] = None
    closing_paragraph: Optional[str] = None
    job_title: Optional[str] = Field(None, max_length=200)
    company_name: Optional[str] = Field(None, max_length=200)
    company_info: Optional[Dict[str, Any]] = None
    job_description: Optional[str] = None
    job_requirements: Optional[List[str]] = None
    hiring_manager_name: Optional[str] = Field(None, max_length=200)
    hiring_manager_title: Optional[str] = Field(None, max_length=200)
    company_address: Optional[Dict[str, Any]] = None
    template_id: Optional[uuid.UUID] = None
    custom_css: Optional[str] = None
    theme_settings: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    key_achievements: Optional[List[str]] = None
    skills_highlighted: Optional[List[str]] = None
    personalization_level: Optional[int] = Field(None, ge=1, le=5)
    is_public: Optional[bool] = None


class CoverLetterResponse(CoverLetterBase):
    """Schema for cover letter response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    resume_id: Optional[uuid.UUID] = None
    status: CoverLetterStatus
    template_id: Optional[uuid.UUID] = None
    opening_paragraph: Optional[str] = None
    body_paragraphs: Optional[List[str]] = None
    closing_paragraph: Optional[str] = None
    company_info: Optional[Dict[str, Any]] = None
    job_description: Optional[str] = None
    job_requirements: Optional[List[str]] = None
    hiring_manager_name: Optional[str] = None
    hiring_manager_title: Optional[str] = None
    company_address: Optional[Dict[str, Any]] = None
    custom_css: Optional[str] = None
    theme_settings: Optional[Dict[str, Any]] = None

    # AI generation data
    ai_generated: bool = False
    ai_prompt: Optional[str] = None
    ai_model_used: Optional[str] = None
    generation_settings: Optional[Dict[str, Any]] = None
    personalization_level: Optional[int] = None
    keywords: Optional[List[str]] = None
    key_achievements: Optional[List[str]] = None
    skills_highlighted: Optional[List[str]] = None

    # File information
    file_path: Optional[str] = None
    pdf_path: Optional[str] = None
    file_size: Optional[int] = None

    # Analytics
    view_count: int = 0
    download_count: int = 0
    share_count: int = 0

    # Quality scores
    content_quality_score: Optional[float] = None
    personalization_score: Optional[float] = None
    keyword_match_score: Optional[float] = None
    overall_score: Optional[float] = None
    last_analyzed_at: Optional[datetime] = None

    # Sharing
    is_featured: bool = False
    share_token: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class CoverLetterListResponse(BaseModel):
    """Schema for cover letter list response."""
    cover_letters: List[CoverLetterResponse]
    total: int
    page: int
    size: int
    pages: int


class CoverLetterSummary(BaseModel):
    """Schema for cover letter summary (lightweight version)."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: CoverLetterStatus
    type: CoverLetterType
    tone: CoverLetterTone
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    overall_score: Optional[float] = None
    view_count: int = 0
    created_at: datetime
    updated_at: datetime


# Template schemas
class CoverLetterTemplateBase(BaseModel):
    """Base cover letter template schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1)
    type: CoverLetterType
    tone: CoverLetterTone
    html_template: str = Field(..., min_length=1)
    css_styles: str = Field(..., min_length=1)
    opening_template: str = Field(..., min_length=1)
    body_template: str = Field(..., min_length=1)
    closing_template: str = Field(..., min_length=1)
    placeholders: List[str] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    is_premium: bool = False
    tags: Optional[List[str]] = Field(default_factory=list)


class CoverLetterTemplateCreate(CoverLetterTemplateBase):
    """Schema for creating a cover letter template."""
    preview_image: Optional[str] = None
    ai_prompt_template: Optional[str] = None
    generation_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    target_industries: Optional[List[str]] = Field(default_factory=list)
    target_job_levels: Optional[List[str]] = Field(default_factory=list)
    target_job_types: Optional[List[str]] = Field(default_factory=list)
    sort_order: int = 0


class CoverLetterTemplateUpdate(BaseModel):
    """Schema for updating a cover letter template."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, min_length=1)
    type: Optional[CoverLetterType] = None
    tone: Optional[CoverLetterTone] = None
    html_template: Optional[str] = Field(None, min_length=1)
    css_styles: Optional[str] = Field(None, min_length=1)
    preview_image: Optional[str] = None
    opening_template: Optional[str] = Field(None, min_length=1)
    body_template: Optional[str] = Field(None, min_length=1)
    closing_template: Optional[str] = Field(None, min_length=1)
    ai_prompt_template: Optional[str] = None
    generation_settings: Optional[Dict[str, Any]] = None
    placeholders: Optional[List[str]] = None
    required_fields: Optional[List[str]] = None
    optional_fields: Optional[List[str]] = None
    target_industries: Optional[List[str]] = None
    target_job_levels: Optional[List[str]] = None
    target_job_types: Optional[List[str]] = None
    is_active: Optional[bool] = None
    is_premium: Optional[bool] = None
    is_featured: Optional[bool] = None
    sort_order: Optional[int] = None
    tags: Optional[List[str]] = None


class CoverLetterTemplateResponse(CoverLetterTemplateBase):
    """Schema for cover letter template response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    preview_image: Optional[str] = None
    ai_prompt_template: Optional[str] = None
    generation_settings: Optional[Dict[str, Any]] = None
    target_industries: Optional[List[str]] = None
    target_job_levels: Optional[List[str]] = None
    target_job_types: Optional[List[str]] = None
    is_active: bool = True
    is_featured: bool = False
    usage_count: int = 0
    rating: Optional[float] = None
    sort_order: int = 0
    created_by: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime


class CoverLetterTemplateListResponse(BaseModel):
    """Schema for cover letter template list response."""
    templates: List[CoverLetterTemplateResponse]
    total: int
    page: int
    size: int
    pages: int


# Analysis schemas
class CoverLetterAnalysisBase(BaseModel):
    """Base cover letter analysis schema."""
    analysis_type: str = Field(..., min_length=1, max_length=50)
    content_quality_score: float = Field(..., ge=0, le=100, description="Content quality score")
    personalization_score: float = Field(..., ge=0, le=100, description="Personalization score")
    keyword_match_score: float = Field(..., ge=0, le=100, description="Keyword matching score")
    tone_consistency_score: float = Field(..., ge=0, le=100, description="Tone consistency score")
    overall_score: float = Field(..., ge=0, le=100, description="Overall cover letter score")
    content_analysis: Dict[str, Any] = Field(default_factory=dict)
    keyword_analysis: Dict[str, Any] = Field(default_factory=dict)
    tone_analysis: Dict[str, Any] = Field(default_factory=dict)
    structure_analysis: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)


class CoverLetterAnalysisCreate(CoverLetterAnalysisBase):
    """Schema for creating a cover letter analysis."""
    job_match_score: Optional[float] = Field(None, ge=0, le=100)
    job_description_match: Optional[Dict[str, Any]] = Field(default_factory=dict)
    missing_keywords: Optional[List[str]] = Field(default_factory=list)
    matching_keywords: Optional[List[str]] = Field(default_factory=list)
    readability_score: Optional[float] = Field(None, ge=0, le=100)
    engagement_score: Optional[float] = Field(None, ge=0, le=100)
    word_count: Optional[int] = Field(None, ge=0)
    paragraph_count: Optional[int] = Field(None, ge=0)


class CoverLetterAnalysisResponse(CoverLetterAnalysisBase):
    """Schema for cover letter analysis response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    cover_letter_id: uuid.UUID
    analysis_version: str = "1.0"
    job_match_score: Optional[float] = None
    job_description_match: Optional[Dict[str, Any]] = None
    missing_keywords: Optional[List[str]] = None
    matching_keywords: Optional[List[str]] = None
    readability_score: Optional[float] = None
    engagement_score: Optional[float] = None
    word_count: Optional[int] = None
    paragraph_count: Optional[int] = None
    processing_time: Optional[float] = None
    ai_model_used: Optional[str] = None
    created_at: datetime


class CoverLetterAnalysisRequest(BaseModel):
    """Schema for requesting cover letter analysis."""
    analysis_types: List[str] = Field(default=["comprehensive"], description="Types of analysis to perform")
    job_description: Optional[str] = Field(None, description="Job description for matching analysis")
    job_description_id: Optional[uuid.UUID] = Field(None, description="Existing job description ID")
    include_suggestions: bool = Field(True, description="Whether to include improvement suggestions")
    target_keywords: Optional[List[str]] = Field(None, description="Specific keywords to check for")


# AI Generation schemas
class CoverLetterGenerationRequest(BaseModel):
    """Schema for AI cover letter generation request."""
    job_description: str = Field(..., min_length=10, description="Job description")
    job_title: str = Field(..., min_length=1, max_length=200, description="Job title")
    company_name: str = Field(..., min_length=1, max_length=200, description="Company name")
    hiring_manager_name: Optional[str] = Field(None, max_length=200, description="Hiring manager name")
    resume_id: Optional[uuid.UUID] = Field(None, description="Resume to base content on")
    template_id: Optional[uuid.UUID] = Field(None, description="Template to use")
    tone: CoverLetterTone = Field(CoverLetterTone.PROFESSIONAL, description="Desired tone")
    personalization_level: int = Field(3, ge=1, le=5, description="Personalization level (1-5)")
    key_points: Optional[List[str]] = Field(None, description="Key points to include")
    company_research: Optional[Dict[str, Any]] = Field(None, description="Company research data")
    custom_prompt: Optional[str] = Field(None, description="Custom AI prompt")
    generation_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation settings")


class CoverLetterGenerationResponse(BaseModel):
    """Schema for AI cover letter generation response."""
    cover_letter_id: uuid.UUID = Field(..., description="Generated cover letter ID")
    content: Dict[str, Any] = Field(..., description="Generated content")
    opening_paragraph: str = Field(..., description="Generated opening paragraph")
    body_paragraphs: List[str] = Field(..., description="Generated body paragraphs")
    closing_paragraph: str = Field(..., description="Generated closing paragraph")
    keywords_used: List[str] = Field(default_factory=list, description="Keywords incorporated")
    personalization_score: float = Field(..., ge=0, le=100, description="Personalization score")
    generation_time: float = Field(..., description="Time taken to generate")
    ai_model_used: str = Field(..., description="AI model used for generation")


# Version schemas
class CoverLetterVersionBase(BaseModel):
    """Base cover letter version schema."""
    version_name: Optional[str] = Field(None, max_length=100)
    change_description: Optional[str] = None


class CoverLetterVersionCreate(CoverLetterVersionBase):
    """Schema for creating a cover letter version."""
    cover_letter_data: Dict[str, Any] = Field(..., description="Complete cover letter data snapshot")


class CoverLetterVersionResponse(CoverLetterVersionBase):
    """Schema for cover letter version response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    cover_letter_id: uuid.UUID
    version_number: int
    cover_letter_data: Dict[str, Any]
    file_path: Optional[str] = None
    pdf_path: Optional[str] = None
    file_size: Optional[int] = None
    created_by: uuid.UUID
    created_at: datetime


# Job Description schemas
class JobDescriptionBase(BaseModel):
    """Base job description schema."""
    title: str = Field(..., min_length=1, max_length=200)
    company_name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10)
    requirements: Optional[List[str]] = Field(default_factory=list)
    responsibilities: Optional[List[str]] = Field(default_factory=list)
    location: Optional[str] = Field(None, max_length=200)
    job_type: Optional[str] = Field(None, max_length=50)
    experience_level: Optional[str] = Field(None, max_length=50)
    industry: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)


class JobDescriptionCreate(JobDescriptionBase):
    """Schema for creating a job description."""
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    currency: Optional[str] = Field(None, max_length=3)
    external_url: Optional[str] = Field(None, max_length=500)
    external_id: Optional[str] = Field(None, max_length=100)
    source: Optional[str] = Field(None, max_length=100)


class JobDescriptionUpdate(BaseModel):
    """Schema for updating a job description."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    company_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=10)
    requirements: Optional[List[str]] = None
    responsibilities: Optional[List[str]] = None
    location: Optional[str] = Field(None, max_length=200)
    job_type: Optional[str] = Field(None, max_length=50)
    experience_level: Optional[str] = Field(None, max_length=50)
    industry: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    currency: Optional[str] = Field(None, max_length=3)
    external_url: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None


class JobDescriptionResponse(JobDescriptionBase):
    """Schema for job description response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: Optional[str] = None
    external_url: Optional[str] = None
    external_id: Optional[str] = None
    source: Optional[str] = None
    extracted_keywords: Optional[List[str]] = None
    skill_requirements: Optional[List[str]] = None
    education_requirements: Optional[List[str]] = None
    view_count: int = 0
    usage_count: int = 0
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class JobDescriptionListResponse(BaseModel):
    """Schema for job description list response."""
    job_descriptions: List[JobDescriptionResponse]
    total: int
    page: int
    size: int
    pages: int


# Export schemas
class CoverLetterExportRequest(BaseModel):
    """Schema for cover letter export request."""
    format: str = Field(..., description="Export format (pdf, docx, html)")
    template_id: Optional[uuid.UUID] = Field(None, description="Override template for export")
    include_contact_info: bool = Field(True, description="Whether to include contact information")
    letterhead: Optional[Dict[str, Any]] = Field(None, description="Custom letterhead settings")
    watermark: Optional[str] = Field(None, description="Watermark text for PDF")
    custom_settings: Optional[Dict[str, Any]] = Field(None, description="Custom export settings")


class CoverLetterExportResponse(BaseModel):
    """Schema for cover letter export response."""
    file_url: str = Field(..., description="URL to download the exported file")
    file_name: str = Field(..., description="Name of the exported file")
    file_size: int = Field(..., description="Size of the exported file in bytes")
    format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="When the download link expires")


# Optimization schemas
class CoverLetterOptimizationRequest(BaseModel):
    """Schema for cover letter optimization request."""
    optimization_type: str = Field(..., description="Type of optimization (keyword, tone, length, ats)")
    job_description: Optional[str] = Field(None, description="Job description for optimization")
    target_keywords: Optional[List[str]] = Field(None, description="Specific keywords to optimize for")
    target_tone: Optional[CoverLetterTone] = Field(None, description="Target tone for optimization")
    target_length: Optional[int] = Field(None, ge=100, le=1000, description="Target word count")
    preserve_sections: Optional[List[str]] = Field(None, description="Sections to preserve during optimization")
    optimization_level: int = Field(3, ge=1, le=5, description="Optimization intensity (1-5)")


class CoverLetterOptimizationResponse(BaseModel):
    """Schema for cover letter optimization response."""
    optimized_content: Dict[str, Any] = Field(..., description="Optimized content")
    changes_made: List[Dict[str, Any]] = Field(..., description="List of changes made")
    improvement_score: float = Field(..., ge=0, le=100, description="Predicted improvement score")
    optimization_summary: str = Field(..., description="Summary of optimizations performed")
    before_scores: Dict[str, float] = Field(..., description="Scores before optimization")
    after_scores: Dict[str, float] = Field(..., description="Predicted scores after optimization")


# Bulk operations schemas
class BulkCoverLetterCreate(BaseModel):
    """Schema for bulk cover letter creation."""
    base_cover_letter_id: uuid.UUID = Field(..., description="Base cover letter to use as template")
    job_descriptions: List[Dict[str, Any]] = Field(..., description="List of job descriptions")
    customization_level: int = Field(3, ge=1, le=5, description="Level of customization per job")
    include_company_research: bool = Field(True, description="Whether to include company research")
    generation_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BulkCoverLetterResponse(BaseModel):
    """Schema for bulk cover letter creation response."""
    task_id: uuid.UUID = Field(..., description="Background task ID")
    total_jobs: int = Field(..., description="Total number of jobs to process")
    estimated_completion_time: int = Field(..., description="Estimated completion time in seconds")
    status: str = Field(..., description="Task status")


class BulkOperationStatus(BaseModel):
    """Schema for bulk operation status."""
    task_id: uuid.UUID
    status: str = Field(..., description="Task status (pending, running, completed, failed)")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    total_items: int = Field(..., description="Total items to process")
    completed_items: int = Field(..., description="Completed items")
    failed_items: int = Field(..., description="Failed items")
    created_cover_letters: List[uuid.UUID] = Field(default_factory=list, description="IDs of created cover letters")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


# Statistics schemas
class CoverLetterStatsResponse(BaseModel):
    """Schema for cover letter statistics response."""
    total_cover_letters: int
    active_cover_letters: int
    draft_cover_letters: int
    ai_generated_count: int
    total_views: int
    total_downloads: int
    average_score: Optional[float] = None
    average_personalization_score: Optional[float] = None
    most_used_template: Optional[str] = None
    most_common_tone: Optional[str] = None
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list)


# Comparison schemas
class CoverLetterComparisonRequest(BaseModel):
    """Schema for comparing cover letters."""
    cover_letter_ids: List[uuid.UUID] = Field(..., min_items=2, max_items=5, description="Cover letters to compare")
    comparison_criteria: List[str] = Field(default=["content_quality", "personalization", "keyword_match"],
                                           description="Criteria to compare")
    job_description: Optional[str] = Field(None, description="Job description for context")


class CoverLetterComparisonResponse(BaseModel):
    """Schema for cover letter comparison response."""
    comparison_results: Dict[str, Dict[str, Any]] = Field(..., description="Comparison results by cover letter ID")
    recommendations: List[Dict[str, Any]] = Field(..., description="Recommendations based on comparison")
    best_performer: Dict[str, Any] = Field(..., description="Best performing cover letter by criteria")
    summary: str = Field(..., description="Summary of comparison results")


# Feedback schemas
class CoverLetterFeedback(BaseModel):
    """Schema for cover letter feedback."""
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    feedback_text: Optional[str] = Field(None, description="Detailed feedback")
    feedback_type: str = Field(..., description="Type of feedback (quality, template, ai_generation)")
    suggestions: Optional[List[str]] = Field(None, description="Specific suggestions")
    would_recommend: Optional[bool] = Field(None, description="Would recommend to others")


class CoverLetterFeedbackResponse(BaseModel):
    """Schema for cover letter feedback response."""
    id: uuid.UUID
    cover_letter_id: uuid.UUID
    user_id: uuid.UUID
    rating: int
    feedback_text: Optional[str] = None
    feedback_type: str
    suggestions: Optional[List[str]] = None
    would_recommend: Optional[bool] = None
    created_at: datetime


# Search and filtering schemas
class CoverLetterSearchRequest(BaseModel):
    """Schema for cover letter search request."""
    query: Optional[str] = Field(None, description="Search query")
    status: Optional[CoverLetterStatus] = Field(None, description="Filter by status")
    type: Optional[CoverLetterType] = Field(None, description="Filter by type")
    tone: Optional[CoverLetterTone] = Field(None, description="Filter by tone")
    company_name: Optional[str] = Field(None, description="Filter by company name")
    job_title: Optional[str] = Field(None, description="Filter by job title")
    min_score: Optional[float] = Field(None, ge=0, le=100, description="Minimum overall score")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")
    ai_generated: Optional[bool] = Field(None, description="Filter by AI generated")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", description="Sort order (asc, desc)")
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Page size")


# Template suggestion schemas
class TemplateSuggestionRequest(BaseModel):
    """Schema for template suggestion request."""
    job_description: str = Field(..., min_length=10, description="Job description")
    job_title: str = Field(..., min_length=1, max_length=200, description="Job title")
    company_name: str = Field(..., min_length=1, max_length=200, description="Company name")
    industry: Optional[str] = Field(None, max_length=100, description="Industry")
    experience_level: Optional[str] = Field(None, max_length=50, description="Experience level")
    preferred_tone: Optional[CoverLetterTone] = Field(None, description="Preferred tone")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")


class TemplateSuggestionResponse(BaseModel):
    """Schema for template suggestion response."""
    suggested_templates: List[CoverLetterTemplateResponse] = Field(..., description="Suggested templates")
    match_scores: Dict[str, float] = Field(..., description="Match scores for each template")
    reasoning: Dict[str, str] = Field(..., description="Reasoning for each suggestion")
    best_match: CoverLetterTemplateResponse = Field(..., description="Best matching template")