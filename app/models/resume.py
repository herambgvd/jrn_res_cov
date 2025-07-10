"""
Resume-related SQLAlchemy models.
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


class ResumeStatus(str, Enum):
    """Resume status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


class TemplateCategory(str, Enum):
    """Template category enumeration."""
    MODERN = "modern"
    CLASSIC = "classic"
    CREATIVE = "creative"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    ACADEMIC = "academic"


class ExperienceType(str, Enum):
    """Experience type enumeration."""
    WORK = "work"
    EDUCATION = "education"
    PROJECT = "project"
    VOLUNTEER = "volunteer"
    CERTIFICATION = "certification"
    AWARD = "award"


class Resume(Base):
    """Main resume model."""

    __tablename__ = "resumes"

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

    # Basic information
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[ResumeStatus] = mapped_column(
        Enum(ResumeStatus),
        default=ResumeStatus.DRAFT,
        nullable=False,
        index=True
    )

    # Template and styling
    template_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_templates.id", ondelete="SET NULL"),
        index=True
    )
    custom_css: Mapped[Optional[str]] = mapped_column(Text)
    theme_settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Personal information
    personal_info: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Resume sections (stored as JSON for flexibility)
    work_experience: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    education: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    skills: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    projects: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    certifications: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    languages: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    awards: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    references: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    custom_sections: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)

    # SEO and optimization
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    target_job_title: Mapped[Optional[str]] = mapped_column(String(200))
    target_industry: Mapped[Optional[str]] = mapped_column(String(100))

    # File information
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    pdf_path: Mapped[Optional[str]] = mapped_column(String(500))
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64))

    # Analytics and metrics
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    download_count: Mapped[int] = mapped_column(Integer, default=0)
    share_count: Mapped[int] = mapped_column(Integer, default=0)

    # AI Analysis scores
    ats_score: Mapped[Optional[float]] = mapped_column(Float)
    content_score: Mapped[Optional[float]] = mapped_column(Float)
    format_score: Mapped[Optional[float]] = mapped_column(Float)
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
    template: Mapped[Optional["ResumeTemplate"]] = relationship(
        "ResumeTemplate",
        back_populates="resumes"
    )
    analyses: Mapped[List["ResumeAnalysis"]] = relationship(
        "ResumeAnalysis",
        back_populates="resume",
        cascade="all, delete-orphan"
    )
    versions: Mapped[List["ResumeVersion"]] = relationship(
        "ResumeVersion",
        back_populates="resume",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_resume_user_status", "user_id", "status"),
        Index("idx_resume_created", "created_at"),
        Index("idx_resume_score", "overall_score"),
    )

    def __repr__(self) -> str:
        return f"<Resume(id={self.id}, title='{self.title}', status='{self.status}')>"


class ResumeTemplate(Base):
    """Resume template model."""

    __tablename__ = "resume_templates"

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
    category: Mapped[TemplateCategory] = mapped_column(
        Enum(TemplateCategory),
        nullable=False,
        index=True
    )

    # Template content
    html_template: Mapped[str] = mapped_column(Text, nullable=False)
    css_styles: Mapped[str] = mapped_column(Text, nullable=False)
    preview_image: Mapped[Optional[str]] = mapped_column(String(500))

    # Settings and configuration
    default_settings: Mapped[Dict[str, Any]] = mapped_column(JSON)
    customizable_fields: Mapped[List[str]] = mapped_column(JSON)
    supported_sections: Mapped[List[str]] = mapped_column(JSON)

    # Availability and popularity
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[Optional[float]] = mapped_column(Float)

    # Ordering and display
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
    resumes: Mapped[List["Resume"]] = relationship(
        "Resume",
        back_populates="template"
    )

    # Indexes
    __table_args__ = (
        Index("idx_template_category_active", "category", "is_active"),
        Index("idx_template_featured", "is_featured"),
    )

    def __repr__(self) -> str:
        return f"<ResumeTemplate(id={self.id}, name='{self.name}', category='{self.category}')>"


class ResumeAnalysis(Base):
    """Resume analysis results model."""

    __tablename__ = "resume_analyses"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Resume relationship
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Analysis type and version
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)
    analysis_version: Mapped[str] = mapped_column(String(20), default="1.0")

    # Scores and metrics
    ats_score: Mapped[float] = mapped_column(Float, nullable=False)
    content_score: Mapped[float] = mapped_column(Float, nullable=False)
    format_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Detailed analysis results
    keyword_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    content_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    format_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    suggestions: Mapped[List[Dict[str, Any]]] = mapped_column(JSON)

    # Job matching (if job description provided)
    job_match_score: Mapped[Optional[float]] = mapped_column(Float)
    job_description_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    missing_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    matching_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)

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
    resume: Mapped["Resume"] = relationship(
        "Resume",
        back_populates="analyses"
    )

    # Indexes
    __table_args__ = (
        Index("idx_analysis_resume_created", "resume_id", "created_at"),
        Index("idx_analysis_overall_score", "overall_score"),
    )

    def __repr__(self) -> str:
        return f"<ResumeAnalysis(id={self.id}, resume_id={self.resume_id}, score={self.overall_score})>"


class ResumeVersion(Base):
    """Resume version history model."""

    __tablename__ = "resume_versions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Resume relationship
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Version information
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    version_name: Mapped[Optional[str]] = mapped_column(String(100))
    change_description: Mapped[Optional[str]] = mapped_column(Text)

    # Snapshot of resume data
    resume_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

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
    resume: Mapped["Resume"] = relationship(
        "Resume",
        back_populates="versions"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("resume_id", "version_number", name="uq_resume_version"),
        Index("idx_version_resume_number", "resume_id", "version_number"),
    )

    def __repr__(self) -> str:
        return f"<ResumeVersion(id={self.id}, resume_id={self.resume_id}, version={self.version_number})>"


class ResumeSkill(Base):
    """Resume skills model with proficiency levels."""

    __tablename__ = "resume_skills"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Resume relationship
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Skill information
    skill_name: Mapped[str] = mapped_column(String(100), nullable=False)
    skill_category: Mapped[Optional[str]] = mapped_column(String(50))
    proficiency_level: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 scale
    years_experience: Mapped[Optional[int]] = mapped_column(Integer)

    # Skill metadata
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

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

    # Constraints
    __table_args__ = (
        UniqueConstraint("resume_id", "skill_name", name="uq_resume_skill"),
        Index("idx_skill_resume_category", "resume_id", "skill_category"),
    )

    def __repr__(self) -> str:
        return f"<ResumeSkill(id={self.id}, skill='{self.skill_name}', level={self.proficiency_level})>"


class ResumeShare(Base):
    """Resume sharing and access control model."""

    __tablename__ = "resume_shares"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Resume relationship
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Sharing information
    share_token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    share_type: Mapped[str] = mapped_column(String(20), nullable=False)  # public, private, password
    password_hash: Mapped[Optional[str]] = mapped_column(String(255))

    # Access control
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    allow_download: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Usage tracking
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    download_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Creator information
    created_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

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
        Index("idx_share_token", "share_token"),
        Index("idx_share_resume_active", "resume_id", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<ResumeShare(id={self.id}, resume_id={self.resume_id}, token='{self.share_token}')>"