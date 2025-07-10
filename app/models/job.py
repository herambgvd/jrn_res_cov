"""
Job-related SQLAlchemy models for job descriptions, applications, and tracking.
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


class JobStatus(str, Enum):
    """Job status enumeration."""
    ACTIVE = "active"
    CLOSED = "closed"
    DRAFT = "draft"
    EXPIRED = "expired"
    PAUSED = "paused"


class JobType(str, Enum):
    """Job type enumeration."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"
    REMOTE = "remote"
    HYBRID = "hybrid"


class ExperienceLevel(str, Enum):
    """Experience level enumeration."""
    ENTRY_LEVEL = "entry_level"
    JUNIOR = "junior"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    DIRECTOR = "director"
    VP = "vp"
    C_LEVEL = "c_level"


class ApplicationStatus(str, Enum):
    """Application status enumeration."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    INTERVIEWED = "interviewed"
    REJECTED = "rejected"
    OFFER_RECEIVED = "offer_received"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    WITHDRAWN = "withdrawn"


class Priority(str, Enum):
    """Priority level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Company(Base):
    """Company information model."""

    __tablename__ = "companies"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Basic information
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    slug: Mapped[Optional[str]] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    website: Mapped[Optional[str]] = mapped_column(String(500))

    # Company details
    industry: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    size: Mapped[Optional[str]] = mapped_column(String(50))  # startup, small, medium, large, enterprise
    founded_year: Mapped[Optional[int]] = mapped_column(Integer)
    headquarters: Mapped[Optional[str]] = mapped_column(String(200))

    # Contact information
    email: Mapped[Optional[str]] = mapped_column(String(254))
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    address: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Social media and online presence
    linkedin_url: Mapped[Optional[str]] = mapped_column(String(500))
    twitter_url: Mapped[Optional[str]] = mapped_column(String(500))
    facebook_url: Mapped[Optional[str]] = mapped_column(String(500))
    glassdoor_url: Mapped[Optional[str]] = mapped_column(String(500))

    # Company metrics
    employee_count: Mapped[Optional[int]] = mapped_column(Integer)
    revenue: Mapped[Optional[str]] = mapped_column(String(50))
    funding_stage: Mapped[Optional[str]] = mapped_column(String(50))

    # Company culture and benefits
    culture_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    benefits: Mapped[Optional[List[str]]] = mapped_column(JSON)
    tech_stack: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Ratings and reviews
    glassdoor_rating: Mapped[Optional[float]] = mapped_column(Float)
    indeed_rating: Mapped[Optional[float]] = mapped_column(Float)
    linkedin_rating: Mapped[Optional[float]] = mapped_column(Float)

    # Logo and branding
    logo_url: Mapped[Optional[str]] = mapped_column(String(500))
    cover_image_url: Mapped[Optional[str]] = mapped_column(String(500))
    brand_colors: Mapped[Optional[Dict[str, str]]] = mapped_column(JSON)

    # Data source and verification
    data_source: Mapped[Optional[str]] = mapped_column(String(100))
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    last_updated_from_source: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

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
    job_postings: Mapped[List["JobPosting"]] = relationship(
        "JobPosting",
        back_populates="company"
    )

    # Indexes
    __table_args__ = (
        Index("idx_company_name", "name"),
        Index("idx_company_industry_size", "industry", "size"),
    )

    def __repr__(self) -> str:
        return f"<Company(id={self.id}, name='{self.name}')>"


class JobPosting(Base):
    """Job posting model."""

    __tablename__ = "job_postings"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Company relationship
    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Basic job information
    title: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    slug: Mapped[Optional[str]] = mapped_column(String(300), index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)

    # Job classification
    job_type: Mapped[JobType] = mapped_column(
        Enum(JobType),
        nullable=False,
        index=True
    )
    experience_level: Mapped[ExperienceLevel] = mapped_column(
        Enum(ExperienceLevel),
        nullable=False,
        index=True
    )
    department: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), index=True)

    # Location information
    location: Mapped[Optional[str]] = mapped_column(String(200), index=True)
    remote_allowed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    hybrid_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    travel_required: Mapped[Optional[str]] = mapped_column(String(50))
    timezone_requirements: Mapped[Optional[str]] = mapped_column(String(100))

    # Requirements and qualifications
    requirements: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    preferred_qualifications: Mapped[Optional[List[str]]] = mapped_column(JSON)
    required_skills: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    preferred_skills: Mapped[Optional[List[str]]] = mapped_column(JSON)
    education_requirements: Mapped[Optional[List[str]]] = mapped_column(JSON)
    certifications_required: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Experience requirements
    min_experience_years: Mapped[Optional[int]] = mapped_column(Integer)
    max_experience_years: Mapped[Optional[int]] = mapped_column(Integer)
    industry_experience_required: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Compensation
    salary_min: Mapped[Optional[int]] = mapped_column(Integer)
    salary_max: Mapped[Optional[int]] = mapped_column(Integer)
    salary_currency: Mapped[Optional[str]] = mapped_column(String(3))
    salary_period: Mapped[Optional[str]] = mapped_column(String(20))  # hourly, monthly, annually
    equity_offered: Mapped[Optional[bool]] = mapped_column(Boolean)
    bonus_eligible: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Benefits
    benefits: Mapped[Optional[List[str]]] = mapped_column(JSON)
    perks: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Responsibilities
    responsibilities: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    key_objectives: Mapped[Optional[List[str]]] = mapped_column(JSON)
    success_metrics: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Job status and dates
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus),
        default=JobStatus.ACTIVE,
        nullable=False,
        index=True
    )
    posted_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    application_deadline: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Application process
    application_url: Mapped[Optional[str]] = mapped_column(String(500))
    application_email: Mapped[Optional[str]] = mapped_column(String(254))
    application_instructions: Mapped[Optional[str]] = mapped_column(Text)

    # Contact information
    hiring_manager_name: Mapped[Optional[str]] = mapped_column(String(200))
    hiring_manager_email: Mapped[Optional[str]] = mapped_column(String(254))
    recruiter_name: Mapped[Optional[str]] = mapped_column(String(200))
    recruiter_email: Mapped[Optional[str]] = mapped_column(String(254))

    # SEO and keywords
    keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    seo_title: Mapped[Optional[str]] = mapped_column(String(200))
    seo_description: Mapped[Optional[str]] = mapped_column(Text)

    # External source information
    external_id: Mapped[Optional[str]] = mapped_column(String(200), index=True)
    external_url: Mapped[Optional[str]] = mapped_column(String(500))
    source_platform: Mapped[Optional[str]] = mapped_column(String(100))  # linkedin, indeed, company_website

    # Analytics
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    application_count: Mapped[int] = mapped_column(Integer, default=0)

    # AI analysis data
    analyzed_keywords: Mapped[Optional[List[str]]] = mapped_column(JSON)
    difficulty_score: Mapped[Optional[float]] = mapped_column(Float)
    competitiveness_score: Mapped[Optional[float]] = mapped_column(Float)

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
    company: Mapped["Company"] = relationship(
        "Company",
        back_populates="job_postings"
    )
    applications: Mapped[List["JobApplication"]] = relationship(
        "JobApplication",
        back_populates="job_posting"
    )

    # Indexes
    __table_args__ = (
        Index("idx_job_company_status", "company_id", "status"),
        Index("idx_job_type_level", "job_type", "experience_level"),
        Index("idx_job_location_remote", "location", "remote_allowed"),
        Index("idx_job_posted_date", "posted_date"),
        Index("idx_job_salary_range", "salary_min", "salary_max"),
    )

    def __repr__(self) -> str:
        return f"<JobPosting(id={self.id}, title='{self.title}', company_id={self.company_id})>"


class JobApplication(Base):
    """Job application tracking model."""

    __tablename__ = "job_applications"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # User information
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # Job and documents
    job_posting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_postings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    resume_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="SET NULL"),
        index=True
    )
    cover_letter_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cover_letters.id", ondelete="SET NULL"),
        index=True
    )

    # Application details
    status: Mapped[ApplicationStatus] = mapped_column(
        Enum(ApplicationStatus),
        default=ApplicationStatus.DRAFT,
        nullable=False,
        index=True
    )
    priority: Mapped[Priority] = mapped_column(
        Enum(Priority),
        default=Priority.MEDIUM,
        nullable=False
    )

    # Application content
    application_notes: Mapped[Optional[str]] = mapped_column(Text)
    custom_message: Mapped[Optional[str]] = mapped_column(Text)

    # Contact information
    contact_person: Mapped[Optional[str]] = mapped_column(String(200))
    contact_email: Mapped[Optional[str]] = mapped_column(String(254))
    referral_source: Mapped[Optional[str]] = mapped_column(String(200))
    referral_person: Mapped[Optional[str]] = mapped_column(String(200))

    # Application process tracking
    applied_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    first_response_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    interview_dates: Mapped[Optional[List[str]]] = mapped_column(JSON)  # ISO datetime strings
    final_decision_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Outcome information
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text)
    feedback_received: Mapped[Optional[str]] = mapped_column(Text)
    salary_offered: Mapped[Optional[int]] = mapped_column(Integer)

    # Follow-up and reminders
    follow_up_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    reminder_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    follow_up_notes: Mapped[Optional[str]] = mapped_column(Text)

    # Application source and method
    application_method: Mapped[Optional[str]] = mapped_column(String(100))  # online, email, in_person, referral
    source_platform: Mapped[Optional[str]] = mapped_column(String(100))  # linkedin, indeed, company_website

    # Match analysis
    match_score: Mapped[Optional[float]] = mapped_column(Float)
    skills_match_percentage: Mapped[Optional[float]] = mapped_column(Float)
    experience_match_score: Mapped[Optional[float]] = mapped_column(Float)

    # Files and attachments
    additional_documents: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    portfolio_links: Mapped[Optional[List[str]]] = mapped_column(JSON)

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
    job_posting: Mapped["JobPosting"] = relationship(
        "JobPosting",
        back_populates="applications"
    )
    activities: Mapped[List["ApplicationActivity"]] = relationship(
        "ApplicationActivity",
        back_populates="application",
        cascade="all, delete-orphan"
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("user_id", "job_posting_id", name="uq_user_job_application"),
        Index("idx_application_user_status", "user_id", "status"),
        Index("idx_application_applied_date", "applied_date"),
        Index("idx_application_priority", "priority"),
    )

    def __repr__(self) -> str:
        return f"<JobApplication(id={self.id}, user_id={self.user_id}, status='{self.status}')>"


class ApplicationActivity(Base):
    """Application activity tracking model."""

    __tablename__ = "application_activities"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Application relationship
    application_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Activity details
    activity_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Activity metadata
    activity_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    activity_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Status change tracking
    old_status: Mapped[Optional[str]] = mapped_column(String(50))
    new_status: Mapped[Optional[str]] = mapped_column(String(50))

    # Contact information
    contact_person: Mapped[Optional[str]] = mapped_column(String(200))
    contact_method: Mapped[Optional[str]] = mapped_column(String(50))  # email, phone, in_person

    # Follow-up information
    requires_follow_up: Mapped[bool] = mapped_column(Boolean, default=False)
    follow_up_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # User notes
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    application: Mapped["JobApplication"] = relationship(
        "JobApplication",
        back_populates="activities"
    )

    # Indexes
    __table_args__ = (
        Index("idx_activity_application_date", "application_id", "activity_date"),
        Index("idx_activity_type", "activity_type"),
    )

    def __repr__(self) -> str:
        return f"<ApplicationActivity(id={self.id}, type='{self.activity_type}', date={self.activity_date})>"


class JobAlert(Base):
    """Job alert and notification model."""

    __tablename__ = "job_alerts"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # User information
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # Alert configuration
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Search criteria
    keywords: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    job_titles: Mapped[Optional[List[str]]] = mapped_column(JSON)
    companies: Mapped[Optional[List[str]]] = mapped_column(JSON)
    locations: Mapped[Optional[List[str]]] = mapped_column(JSON)
    remote_allowed: Mapped[Optional[bool]] = mapped_column(Boolean)

    # Job type and level filters
    job_types: Mapped[Optional[List[str]]] = mapped_column(JSON)
    experience_levels: Mapped[Optional[List[str]]] = mapped_column(JSON)
    departments: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # Salary filters
    min_salary: Mapped[Optional[int]] = mapped_column(Integer)
    max_salary: Mapped[Optional[int]] = mapped_column(Integer)
    salary_currency: Mapped[Optional[str]] = mapped_column(String(3))

    # Notification settings
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    notification_frequency: Mapped[str] = mapped_column(String(20), default="daily")  # immediate, daily, weekly
    email_notifications: Mapped[bool] = mapped_column(Boolean, default=True)
    push_notifications: Mapped[bool] = mapped_column(Boolean, default=False)

    # Alert statistics
    jobs_found: Mapped[int] = mapped_column(Integer, default=0)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_notification_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

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
        Index("idx_alert_user_active", "user_id", "is_active"),
        Index("idx_alert_last_run", "last_run_at"),
    )

    def __repr__(self) -> str:
        return f"<JobAlert(id={self.id}, name='{self.name}', user_id={self.user_id})>"