"""
Analysis and optimization models for resumes and cover letters.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text,
    Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class AnalysisType(str, Enum):
    """Analysis type enumeration."""
    ATS_SCAN = "ats_scan"
    KEYWORD_ANALYSIS = "keyword_analysis"
    CONTENT_QUALITY = "content_quality"
    FORMAT_CHECK = "format_check"
    JOB_MATCH = "job_match"
    SKILL_GAP = "skill_gap"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    INDUSTRY_BENCHMARK = "industry_benchmark"


class SuggestionType(str, Enum):
    """Suggestion type enumeration."""
    CONTENT_IMPROVEMENT = "content_improvement"
    KEYWORD_OPTIMIZATION = "keyword_optimization"
    FORMAT_ENHANCEMENT = "format_enhancement"
    SKILL_ADDITION = "skill_addition"
    EXPERIENCE_HIGHLIGHT = "experience_highlight"
    GRAMMAR_FIX = "grammar_fix"
    ATS_OPTIMIZATION = "ats_optimization"


class SuggestionPriority(str, Enum):
    """Suggestion priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisSession(Base):
    """Analysis session model to track batch analysis operations."""

    __tablename__ = "analysis_sessions"

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

    # Session information
    session_name: Mapped[Optional[str]] = mapped_column(String(200))
    analysis_types: Mapped[List[str]] = mapped_column(JSON, nullable=False)

    # Target documents
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)  # resume, cover_letter
    target_ids: Mapped[List[str]] = mapped_column(JSON, nullable=False)

    # Job matching context
    job_description_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_descriptions.id", ondelete="SET NULL"),
        index=True
    )
    job_title: Mapped[Optional[str]] = mapped_column(String(200))
    company_name: Mapped[Optional[str]] = mapped_column(String(200))

    # Session status and progress
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    total_items: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)

    # Results summary
    overall_score: Mapped[Optional[float]] = mapped_column(Float)
    score_breakdown: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON)
    total_suggestions: Mapped[int] = mapped_column(Integer, default=0)
    critical_issues: Mapped[int] = mapped_column(Integer, default=0)

    # Processing information
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    ai_model_used: Mapped[Optional[str]] = mapped_column(String(100))
    error_details: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Indexes
    __table_args__ = (
        Index("idx_session_user_created", "user_id", "created_at"),
        Index("idx_session_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisSession(id={self.id}, user_id={self.user_id}, status='{self.status}')>"


class AnalysisResult(Base):
    """Unified analysis result model for resumes and cover letters."""

    __tablename__ = "analysis_results"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Session relationship
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_sessions.id", ondelete="SET NULL"),
        index=True
    )

    # Target document
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)  # resume, cover_letter
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    # User information
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # Analysis details
    analysis_type: Mapped[AnalysisType] = mapped_column(
        Enum(AnalysisType),
        nullable=False,
        index=True
    )
    analysis_version: Mapped[str] = mapped_column(String(20), default="1.0")

    # Scores
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    sub_scores: Mapped[Dict[str, float]] = mapped_column(JSON)

    # Analysis data
    analysis_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    raw_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Processing metadata
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    ai_model_used: Mapped[Optional[str]] = mapped_column(String(100))
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    suggestions: Mapped[List["Suggestion"]] = relationship(
        "Suggestion",
        back_populates="analysis_result",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_result_target", "target_type", "target_id"),
        Index("idx_result_user_type", "user_id", "analysis_type"),
        Index("idx_result_score", "overall_score"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, type='{self.analysis_type}', score={self.overall_score})>"


class Suggestion(Base):
    """Improvement suggestions model."""

    __tablename__ = "suggestions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Analysis result relationship
    analysis_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Suggestion details
    type: Mapped[SuggestionType] = mapped_column(
        Enum(SuggestionType),
        nullable=False,
        index=True
    )
    priority: Mapped[SuggestionPriority] = mapped_column(
        Enum(SuggestionPriority),
        nullable=False,
        index=True
    )

    # Content
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    before_text: Mapped[Optional[str]] = mapped_column(Text)
    suggested_text: Mapped[Optional[str]] = mapped_column(Text)

    # Location and context
    section: Mapped[Optional[str]] = mapped_column(String(100))
    field_path: Mapped[Optional[str]] = mapped_column(String(200))
    line_number: Mapped[Optional[int]] = mapped_column(Integer)
    character_position: Mapped[Optional[int]] = mapped_column(Integer)

    # Impact and reasoning
    impact_score: Mapped[Optional[float]] = mapped_column(Float)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float)

    # Implementation details
    implementation_effort: Mapped[Optional[str]] = mapped_column(String(20))  # low, medium, high
    estimated_time_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    requires_manual_review: Mapped[bool] = mapped_column(Boolean, default=False)

    # User interaction
    is_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    is_dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    user_feedback: Mapped[Optional[str]] = mapped_column(Text)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    dismissed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    analysis_result: Mapped["AnalysisResult"] = relationship(
        "AnalysisResult",
        back_populates="suggestions"
    )

    # Indexes
    __table_args__ = (
        Index("idx_suggestion_analysis_priority", "analysis_result_id", "priority"),
        Index("idx_suggestion_type_priority", "type", "priority"),
        Index("idx_suggestion_applied", "is_applied"),
    )

    def __repr__(self) -> str:
        return f"<Suggestion(id={self.id}, type='{self.type}', priority='{self.priority}')>"


class KeywordAnalysis(Base):
    """Keyword analysis results model."""

    __tablename__ = "keyword_analyses"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Target document
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    # User information
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # Job context
    job_description_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("job_descriptions.id", ondelete="SET NULL"),
        index=True
    )
    job_title: Mapped[Optional[str]] = mapped_column(String(200))
    industry: Mapped[Optional[str]] = mapped_column(String(100))

    # Keyword data
    extracted_keywords: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    missing_keywords: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    matching_keywords: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    suggested_keywords: Mapped[List[str]] = mapped_column(JSON, nullable=False)

    # Keyword categories
    technical_keywords: Mapped[List[str]] = mapped_column(JSON)
    soft_skills_keywords: Mapped[List[str]] = mapped_column(JSON)
    industry_keywords: Mapped[List[str]] = mapped_column(JSON)
    action_keywords: Mapped[List[str]] = mapped_column(JSON)

    # Analysis scores
    keyword_density_score: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_diversity_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_keyword_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Detailed analysis
    keyword_frequency: Mapped[Dict[str, int]] = mapped_column(JSON)
    keyword_positions: Mapped[Dict[str, List[int]]] = mapped_column(JSON)
    keyword_importance: Mapped[Dict[str, float]] = mapped_column(JSON)

    # Processing metadata
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    ai_model_used: Mapped[Optional[str]] = mapped_column(String(100))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_keyword_target", "target_type", "target_id"),
        Index("idx_keyword_user_created", "user_id", "created_at"),
        Index("idx_keyword_job", "job_description_id"),
    )

    def __repr__(self) -> str:
        return f"<KeywordAnalysis(id={self.id}, target_id={self.target_id}, score={self.overall_keyword_score})>"


class ATSAnalysis(Base):
    """ATS (Applicant Tracking System) compatibility analysis."""

    __tablename__ = "ats_analyses"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Target document
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    # User information
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # ATS compatibility scores
    parsing_score: Mapped[float] = mapped_column(Float, nullable=False)
    formatting_score: Mapped[float] = mapped_column(Float, nullable=False)
    readability_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_ats_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Detailed analysis
    parseable_sections: Mapped[List[str]] = mapped_column(JSON)
    problematic_sections: Mapped[List[str]] = mapped_column(JSON)
    formatting_issues: Mapped[List[Dict[str, Any]]] = mapped_column(JSON)

    # Text extraction results
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)
    text_extraction_quality: Mapped[float] = mapped_column(Float, nullable=False)

    # Font and formatting analysis
    font_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    layout_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON)
    image_analysis: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Contact information extraction
    contact_info_extracted: Mapped[Dict[str, Any]] = mapped_column(JSON)
    contact_extraction_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Processing metadata
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    ats_simulator_used: Mapped[Optional[str]] = mapped_column(String(100))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_ats_target", "target_type", "target_id"),
        Index("idx_ats_user_created", "user_id", "created_at"),
        Index("idx_ats_score", "overall_ats_score"),
    )

    def __repr__(self) -> str:
        return f"<ATSAnalysis(id={self.id}, target_id={self.target_id}, score={self.overall_ats_score})>"


class CompetitiveAnalysis(Base):
    """Competitive analysis against industry benchmarks."""

    __tablename__ = "competitive_analyses"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )

    # Target document
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    # User information
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )

    # Benchmark context
    industry: Mapped[str] = mapped_column(String(100), nullable=False)
    job_level: Mapped[str] = mapped_column(String(50), nullable=False)
    experience_years: Mapped[Optional[int]] = mapped_column(Integer)

    # Competitive scores
    content_competitiveness: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_competitiveness: Mapped[float] = mapped_column(Float, nullable=False)
    format_competitiveness: Mapped[float] = mapped_column(Float, nullable=False)
    overall_competitiveness: Mapped[float] = mapped_column(Float, nullable=False)

    # Percentile rankings
    content_percentile: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_percentile: Mapped[float] = mapped_column(Float, nullable=False)
    format_percentile: Mapped[float] = mapped_column(Float, nullable=False)
    overall_percentile: Mapped[float] = mapped_column(Float, nullable=False)

    # Benchmark data
    industry_averages: Mapped[Dict[str, float]] = mapped_column(JSON)
    top_performer_examples: Mapped[List[Dict[str, Any]]] = mapped_column(JSON)
    improvement_opportunities: Mapped[List[Dict[str, Any]]] = mapped_column(JSON)

    # Sample size and confidence
    benchmark_sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence_interval: Mapped[float] = mapped_column(Float, nullable=False)

    # Processing metadata
    processing_time: Mapped[Optional[float]] = mapped_column(Float)
    benchmark_data_version: Mapped[str] = mapped_column(String(20), default="1.0")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_competitive_target", "target_type", "target_id"),
        Index("idx_competitive_industry", "industry", "job_level"),
        Index("idx_competitive_percentile", "overall_percentile"),
    )

    def __repr__(self) -> str:
        return f"<CompetitiveAnalysis(id={self.id}, industry='{self.industry}', percentile={self.overall_percentile})>"