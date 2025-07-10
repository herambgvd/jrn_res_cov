"""
Pydantic schemas for Resume API endpoints.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator, ConfigDict

from app.models.resume import ResumeStatus, TemplateCategory


# Base schemas
class ResumeBase(BaseModel):
    """Base resume schema with common fields."""
    title: str = Field(..., min_length=1, max_length=200, description="Resume title")
    description: Optional[str] = Field(None, description="Resume description")
    personal_info: Dict[str, Any] = Field(..., description="Personal information")
    target_job_title: Optional[str] = Field(None, max_length=200, description="Target job title")
    target_industry: Optional[str] = Field(None, max_length=100, description="Target industry")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Resume keywords")
    is_public: bool = Field(False, description="Whether resume is publicly visible")


class ResumeCreate(ResumeBase):
    """Schema for creating a resume."""
    template_id: Optional[uuid.UUID] = Field(None, description="Template ID to use")
    work_experience: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    education: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    skills: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    projects: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    certifications: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    languages: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    awards: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    references: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    custom_sections: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    @validator("personal_info")
    def validate_personal_info(cls, v):
        """Validate personal information has required fields."""
        required_fields = ["first_name", "last_name", "email"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Personal info must include {field}")
        return v


class ResumeUpdate(BaseModel):
    """Schema for updating a resume."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[ResumeStatus] = None
    personal_info: Optional[Dict[str, Any]] = None
    work_experience: Optional[List[Dict[str, Any]]] = None
    education: Optional[List[Dict[str, Any]]] = None
    skills: Optional[List[Dict[str, Any]]] = None
    projects: Optional[List[Dict[str, Any]]] = None
    certifications: Optional[List[Dict[str, Any]]] = None
    languages: Optional[List[Dict[str, Any]]] = None
    awards: Optional[List[Dict[str, Any]]] = None
    references: Optional[List[Dict[str, Any]]] = None
    custom_sections: Optional[List[Dict[str, Any]]] = None
    template_id: Optional[uuid.UUID] = None
    custom_css: Optional[str] = None
    theme_settings: Optional[Dict[str, Any]] = None
    target_job_title: Optional[str] = Field(None, max_length=200)
    target_industry: Optional[str] = Field(None, max_length=100)
    keywords: Optional[List[str]] = None
    is_public: Optional[bool] = None


class ResumeResponse(ResumeBase):
    """Schema for resume response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    status: ResumeStatus
    template_id: Optional[uuid.UUID] = None
    work_experience: Optional[List[Dict[str, Any]]] = None
    education: Optional[List[Dict[str, Any]]] = None
    skills: Optional[List[Dict[str, Any]]] = None
    projects: Optional[List[Dict[str, Any]]] = None
    certifications: Optional[List[Dict[str, Any]]] = None
    languages: Optional[List[Dict[str, Any]]] = None
    awards: Optional[List[Dict[str, Any]]] = None
    references: Optional[List[Dict[str, Any]]] = None
    custom_sections: Optional[List[Dict[str, Any]]] = None
    custom_css: Optional[str] = None
    theme_settings: Optional[Dict[str, Any]] = None

    # File information
    file_path: Optional[str] = None
    pdf_path: Optional[str] = None
    file_size: Optional[int] = None

    # Analytics
    view_count: int = 0
    download_count: int = 0
    share_count: int = 0

    # AI scores
    ats_score: Optional[float] = None
    content_score: Optional[float] = None
    format_score: Optional[float] = None
    overall_score: Optional[float] = None
    last_analyzed_at: Optional[datetime] = None

    # Sharing
    is_featured: bool = False
    share_token: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class ResumeListResponse(BaseModel):
    """Schema for resume list response."""
    resumes: List[ResumeResponse]
    total: int
    page: int
    size: int
    pages: int


class ResumeSummary(BaseModel):
    """Schema for resume summary (lightweight version)."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    status: ResumeStatus
    template_id: Optional[uuid.UUID] = None
    overall_score: Optional[float] = None
    view_count: int = 0
    created_at: datetime
    updated_at: datetime


# Template schemas
class ResumeTemplateBase(BaseModel):
    """Base resume template schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1)
    category: TemplateCategory
    html_template: str = Field(..., min_length=1)
    css_styles: str = Field(..., min_length=1)
    default_settings: Dict[str, Any] = Field(default_factory=dict)
    customizable_fields: List[str] = Field(default_factory=list)
    supported_sections: List[str] = Field(default_factory=list)
    is_premium: bool = False
    tags: Optional[List[str]] = Field(default_factory=list)


class ResumeTemplateCreate(ResumeTemplateBase):
    """Schema for creating a resume template."""
    preview_image: Optional[str] = None
    sort_order: int = 0


class ResumeTemplateUpdate(BaseModel):
    """Schema for updating a resume template."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, min_length=1)
    category: Optional[TemplateCategory] = None
    html_template: Optional[str] = Field(None, min_length=1)
    css_styles: Optional[str] = Field(None, min_length=1)
    preview_image: Optional[str] = None
    default_settings: Optional[Dict[str, Any]] = None
    customizable_fields: Optional[List[str]] = None
    supported_sections: Optional[List[str]] = None
    is_active: Optional[bool] = None
    is_premium: Optional[bool] = None
    is_featured: Optional[bool] = None
    sort_order: Optional[int] = None
    tags: Optional[List[str]] = None


class ResumeTemplateResponse(ResumeTemplateBase):
    """Schema for resume template response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    preview_image: Optional[str] = None
    is_active: bool = True
    is_featured: bool = False
    usage_count: int = 0
    rating: Optional[float] = None
    sort_order: int = 0
    created_by: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime


class ResumeTemplateListResponse(BaseModel):
    """Schema for resume template list response."""
    templates: List[ResumeTemplateResponse]
    total: int
    page: int
    size: int
    pages: int


# Analysis schemas
class ResumeAnalysisBase(BaseModel):
    """Base resume analysis schema."""
    analysis_type: str = Field(..., min_length=1, max_length=50)
    ats_score: float = Field(..., ge=0, le=100, description="ATS compatibility score")
    content_score: float = Field(..., ge=0, le=100, description="Content quality score")
    format_score: float = Field(..., ge=0, le=100, description="Format quality score")
    overall_score: float = Field(..., ge=0, le=100, description="Overall resume score")
    keyword_analysis: Dict[str, Any] = Field(default_factory=dict)
    content_analysis: Dict[str, Any] = Field(default_factory=dict)
    format_analysis: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)


class ResumeAnalysisCreate(ResumeAnalysisBase):
    """Schema for creating a resume analysis."""
    job_description_id: Optional[uuid.UUID] = None
    job_match_score: Optional[float] = Field(None, ge=0, le=100)
    missing_keywords: Optional[List[str]] = Field(default_factory=list)
    matching_keywords: Optional[List[str]] = Field(default_factory=list)


class ResumeAnalysisResponse(ResumeAnalysisBase):
    """Schema for resume analysis response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    resume_id: uuid.UUID
    analysis_version: str = "1.0"
    job_match_score: Optional[float] = None
    job_description_id: Optional[uuid.UUID] = None
    missing_keywords: Optional[List[str]] = None
    matching_keywords: Optional[List[str]] = None
    processing_time: Optional[float] = None
    ai_model_used: Optional[str] = None
    created_at: datetime


class ResumeAnalysisRequest(BaseModel):
    """Schema for requesting resume analysis."""
    analysis_types: List[str] = Field(default=["comprehensive"], description="Types of analysis to perform")
    job_description: Optional[str] = Field(None, description="Job description for matching analysis")
    job_description_id: Optional[uuid.UUID] = Field(None, description="Existing job description ID")
    include_suggestions: bool = Field(True, description="Whether to include improvement suggestions")
    target_ats_systems: Optional[List[str]] = Field(None, description="Specific ATS systems to test against")


# Version schemas
class ResumeVersionBase(BaseModel):
    """Base resume version schema."""
    version_name: Optional[str] = Field(None, max_length=100)
    change_description: Optional[str] = None


class ResumeVersionCreate(ResumeVersionBase):
    """Schema for creating a resume version."""
    resume_data: Dict[str, Any] = Field(..., description="Complete resume data snapshot")


class ResumeVersionResponse(ResumeVersionBase):
    """Schema for resume version response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    resume_id: uuid.UUID
    version_number: int
    resume_data: Dict[str, Any]
    file_path: Optional[str] = None
    pdf_path: Optional[str] = None
    file_size: Optional[int] = None
    created_by: uuid.UUID
    created_at: datetime


# Skill schemas
class ResumeSkillBase(BaseModel):
    """Base resume skill schema."""
    skill_name: str = Field(..., min_length=1, max_length=100)
    skill_category: Optional[str] = Field(None, max_length=50)
    proficiency_level: Optional[int] = Field(None, ge=1, le=5, description="Skill proficiency (1-5)")
    years_experience: Optional[int] = Field(None, ge=0, description="Years of experience with skill")
    is_featured: bool = False
    sort_order: int = 0


class ResumeSkillCreate(ResumeSkillBase):
    """Schema for creating a resume skill."""
    pass


class ResumeSkillUpdate(BaseModel):
    """Schema for updating a resume skill."""
    skill_name: Optional[str] = Field(None, min_length=1, max_length=100)
    skill_category: Optional[str] = Field(None, max_length=50)
    proficiency_level: Optional[int] = Field(None, ge=1, le=5)
    years_experience: Optional[int] = Field(None, ge=0)
    is_featured: Optional[bool] = None
    sort_order: Optional[int] = None


class ResumeSkillResponse(ResumeSkillBase):
    """Schema for resume skill response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    resume_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


# Share schemas
class ResumeShareBase(BaseModel):
    """Base resume share schema."""
    share_type: str = Field(..., description="Type of sharing (public, private, password)")
    password_hash: Optional[str] = None
    allow_download: bool = True
    expires_at: Optional[datetime] = None


class ResumeShareCreate(ResumeShareBase):
    """Schema for creating a resume share."""
    password: Optional[str] = Field(None, description="Password for protected shares")


class ResumeShareUpdate(BaseModel):
    """Schema for updating a resume share."""
    share_type: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    allow_download: Optional[bool] = None
    expires_at: Optional[datetime] = None


class ResumeShareResponse(ResumeShareBase):
    """Schema for resume share response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    resume_id: uuid.UUID
    share_token: str
    is_active: bool = True
    view_count: int = 0
    download_count: int = 0
    last_accessed_at: Optional[datetime] = None
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


# Export schemas
class ResumeExportRequest(BaseModel):
    """Schema for resume export request."""
    format: str = Field(..., description="Export format (pdf, docx, html)")
    template_id: Optional[uuid.UUID] = Field(None, description="Override template for export")
    include_contact_info: bool = Field(True, description="Whether to include contact information")
    watermark: Optional[str] = Field(None, description="Watermark text for PDF")
    custom_settings: Optional[Dict[str, Any]] = Field(None, description="Custom export settings")


class ResumeExportResponse(BaseModel):
    """Schema for resume export response."""
    file_url: str = Field(..., description="URL to download the exported file")
    file_name: str = Field(..., description="Name of the exported file")
    file_size: int = Field(..., description="Size of the exported file in bytes")
    format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="When the download link expires")


# Statistics schemas
class ResumeStatsResponse(BaseModel):
    """Schema for resume statistics response."""
    total_resumes: int
    active_resumes: int
    draft_resumes: int
    total_views: int
    total_downloads: int
    average_score: Optional[float] = None
    most_used_template: Optional[str] = None
    recent_activity: List[Dict[str, Any]] = Field(default_factory=list)