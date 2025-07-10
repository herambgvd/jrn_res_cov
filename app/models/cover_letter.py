"""
Cover Letter-related SQLAlchemy models.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text,
    UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class CoverLetterStatus(str, Enum):
    """Cover letter status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


class CoverLetterType(str, Enum):
    """Cover letter type enumeration."""
    STANDARD = "standard"
    EMAIL = "email"
    NETWORKING = "networking"
    FOLLOW_UP = "follow_up"
    COLD_OUTREACH = "cold_outreach"
    INQUIRY = "inquiry"


class CoverLetterTone(str, Enum):
    """Cover letter tone enumeration."""
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    CREATIVE = "creative"
    CONFIDENT = "confident"
    HUMBLE = "humble"


class CoverLetter(Base):
    """Main cover letter model."""

    __tablename__ = "cover_letters"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # User relationship (from external user service)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # Resume relationship (optional)
    resume_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="SET NULL"),
        index=True
    )

    # Basic information
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[CoverLetterStatus] = mapped_column(
        Enum(CoverLetterStatus),
        default=CoverLetterStatus.DRAFT,
        nullable=False,
        index=True
    )
    type: Mapped[CoverLetterType] = mapped_column(
        Enum(CoverLetterType),
        default=CoverLetterType.STANDARD,
        nullable=False
    )
    tone: Mapped[CoverLetterTone] = mapped_column(
        Enum(CoverLetterTone),
        default=CoverLetterTone.PROFESSIONAL,
        nullable=False
    )

    # Template information
    template_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cover_letter_templates.id", ondelete="SET NULL"),
        index=True
    )
    custom_css: Mapped[Optional[str]] = mapped_column(Text)
    theme_settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Content sections
    content: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    opening_paragraph: Mapped[Optional[str]] = mapped_column(Text)
    body_paragraphs: Mapped[Optional[List[str]]] = mapped_column(JSON)
    closing_paragraph: Mapped[Optional[str]] = mapped_column(Text)

    # Job and company information
    job_title: Mapped[Optional[str]] = mapped_column(String(200))
    company_name: Mapped[Optional[str]] = mapped_column(String(200))
    company_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    job_description: Mapped[Optional[str]] = mapped_column(Text)
    job_requirements: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Contact information
    hiring_manager_name: Mapped[Optional[str]] = mapped_column(String(200))
    hiring_manager_title: Mapped[Optional[str]] = mapped_column(String(200))
    company_address: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # AI Generation settings
    ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_prompt: Mapped[Optional[str]] = mapped_column(Text)
    ai_model_used: Mapped[Optional[str]] = mapped_column(String(100))
    generation_settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Customization and personalization
    personalization_level: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 scale
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    key_achievements: Mapped[Optional[List[str]]] = mapped_column(JSON)
    skills_highlighted: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # File information
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    pdf_path: Mapped[Optional[str]] = mapped_column(String(500))
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64))

    # Analytics and metrics
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    download_count: Mapped[int] = mapped_column(Integer, default=0)
    share_count: Mapped[int] = mapped_column(Integer, default=0)

    # Quality scores
    content_quality_score: Mapped[Optional[float]] = mapped_column(Float)
    personalization_score: Mapped[Optional[float]] = mapped_column(Float)
    keyword_match_score: Mapped[Optional[float]] = mapped_column(Float)
    overall_score: Mapped[Optional[float]] = mapped_column(Float)
    last_analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Visibility and sharing
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    share_token: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    template: Mapped[Optional["CoverLetterTemplate"]] = relationship(
        "CoverLetterTemplate",
        back_populates="cover_letters"
    )
    analyses: Mapped[List["CoverLetterAnalysis"]] = relationship(
        "CoverLetterAnalysis",
        back_populates="cover_letter",
        cascade="all, delete-orphan"
    )
    versions: Mapped[List["CoverLetterVersion"]] = relationship(
        "CoverLetterVersion",
        back_populates="cover_letter",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_cover_letter_user_status", "user_id", "status"),
        Index("idx_cover_letter_created", "created_at"),
        Index("idx_cover_letter_score", "overall_score"),
        Index("idx_cover_letter_job_company", "job_title", "company_name"),
    )

    def __repr__(self) -> str:
        return f"<CoverLetter(id={self.id}, title='{self.title}', status='{self.status}')>"


class CoverLetterTemplate(Base):
    """Cover letter template model."""

    __tablename__ = "cover_letter_templates"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Basic information
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[CoverLetterType] = mapped_column(
        Enum(CoverLetterType),
        nullable=False,
        index=True
    )
    tone: Mapped[CoverLetterTone] = mapped_column(
        Enum(CoverLetterTone),
        nullable=False,
        index=True
    )

    # Template content
    html_template: Mapped[str] = mapped_column(Text, nullable=False)
    css_styles: Mapped[str] = mapped_column(Text, nullable=False)
    preview_image: Mapped[Optional[str]] = mapped_column(String(500))

    # Content structure
    opening_template: Mapped[str] = mapped_column(Text, nullable=False)
    body_template: Mapped[str] = mapped_column(Text, nullable=False)
    closing_template: Mapped[str] = mapped_column(Text, nullable=False)

    # AI prompts and settings
    ai_prompt_template: Mapped[Optional[str]] = mapped_column(Text)
    generation_settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Template configuration
    placeholders: Mapped[List[str]] = mapped_column(JSON)  # Available placeholders
    required_fields: Mapped[List[str]] = mapped_column(JSON)  # Required input fields
    optional_fields: Mapped[List[str]] = mapped_column(JSON)  # Optional input fields

    # Industry and job targeting
    target_industries: Mapped[Optional[List[str]]] = mapped_column(JSON)
    target_job_levels: Mapped[Optional[List[str]]] = mapped_column(JSON)
    target_job_types: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Availability and popularity
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[Optional[float]] = mapped_column(Float)

    # Ordering and categorization
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Creator information
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    cover_letters: Mapped[List["CoverLetter"]] = relationship(
        "CoverLetter",
        back_populates="template"
    )

    # Indexes
    __table_args__ = (
        Index("idx_cl_template_type_tone", "type", "tone"),
        Index("idx_cl_template_active_featured", "is_active", "is_featured"),
    )

    def __repr__(self) -> str:
        return f"<CoverLetterTemplate(id={self.id}, name='{self.name}', type='{self.type}')>"


class CoverLetterAnalysis(Base):
    """Cover letter analysis results model."""

    __tablename__ = "cover_letter_analyses"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Cover letter relationship
    cover_letter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cover_letters.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Analysis type and version
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)
    analysis_version: Mapped[str] = mapped_column(String(20), default="1.0")

    # Quality scores
    content_quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    personalization_score: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_match_score: Mapped[float] = mapped_column(Float, nullable=False)
    tone_consistency_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Detailed analysis results
    content_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    keyword_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    tone_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    structure_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    suggestions: Mapped[List[Dict[str, Any]]] = mapped_column(JSON)

    # Job matching (if job description provided)
    job_match_score: Mapped[Optional[float]] = mapped_column(Float)
    job_description_match: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    missing_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    matching_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Readability and engagement
    readability_score: Mapped[Optional[float]] = mapped_column(Float)
    engagement_score: Mapped[Optional[float]] = mapped_column(Float)
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    paragraph_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Processing information
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    ai_model_used: Mapped[Optional[str]] = mapped_column(String(100))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    cover_letter: Mapped["CoverLetter"] = relationship(
        "CoverLetter",
        back_populates="analyses"
    )

    # Indexes
    __table_args__ = (
        Index("idx_cl_analysis_letter_created", "cover_letter_id", "created_at"),
        Index("idx_cl_analysis_overall_score", "overall_score"),
    )

    def __repr__(self) -> str:
        return f"<CoverLetterAnalysis(id={self.id}, cover_letter_id={self.cover_letter_id}, score={self.overall_score})>"


class CoverLetterVersion(Base):
    """Cover letter version history model."""

    __tablename__ = "cover_letter_versions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Cover letter relationship
    cover_letter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cover_letters.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Version information
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    version_name: Mapped[Optional[str]] = mapped_column(String(100))
    change_description: Mapped[Optional[str]] = mapped_column(Text)

    # Snapshot of cover letter data
    cover_letter_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # File information
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    pdf_path: Mapped[Optional[str]] = mapped_column(String(500))
    file_size: Mapped[Optional[int]] = mapped_column(Integer)

    # Created by information
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    cover_letter: Mapped["CoverLetter"] = relationship(
        "CoverLetter",
        back_populates="versions"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("cover_letter_id", "version_number", name="uq_cover_letter_version"),
        Index("idx_cl_version_letter_number", "cover_letter_id", "version_number"),
    )

    def __repr__(self) -> str:
        return f"<CoverLetterVersion(id={self.id}, cover_letter_id={self.cover_letter_id}, version={self.version_number})>"


class JobDescription(Base):
    """Job description model for analysis and matching."""

    __tablename__ = "job_descriptions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # User relationship
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # Job information
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    company_name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    requirements: Mapped[Optional[List[str]]] = mapped_column(JSON)
    responsibilities: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Job details
    location: Mapped[Optional[str]] = mapped_column(String(200))
    job_type: Mapped[Optional[str]] = mapped_column(String(50))  # full-time, part-time, contract
    experience_level: Mapped[Optional[str]] = mapped_column(String(50))  # entry, mid, senior
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    department: Mapped[Optional[str]] = mapped_column(String(100))

    # Compensation
    salary_min: Mapped[Optional[int]] = mapped_column(Integer)
    salary_max: Mapped[Optional[int]] = mapped_column(Integer)
    currency: Mapped[Optional[str]] = mapped_column(String(3))

    # External references
    external_url: Mapped[Optional[str]] = mapped_column(String(500))
    external_id: Mapped[Optional[str]] = mapped_column(String(100))
    source: Mapped[Optional[str]] = mapped_column(String(100))  # linkedin, indeed, company_website

    # Analysis data
    extracted_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    skill_requirements: Mapped[Optional[List[str]]] = mapped_column(JSON)
    education_requirements: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Usage tracking
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_job_user_created", "user_id", "created_at"),
        Index("idx_job_company_title", "company_name", "title"),
    )

    def __repr__(self) -> str:
        return f"<JobDescription(id={self.id}, title='{self.title}', company='{self.company_name}')>"