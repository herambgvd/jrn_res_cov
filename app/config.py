"""
Application configuration management using Pydantic Settings.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Application
    app_name: str = "AI Resume Platform"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    secret_key: str = Field(..., alias="SECRET_KEY")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8002, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    reload: bool = Field(default=False, alias="RELOAD")

    # Database
    database_url: str = Field(..., alias="DATABASE_URL")
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_cache_ttl: int = Field(default=3600, alias="REDIS_CACHE_TTL")

    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/1", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", alias="CELERY_RESULT_BACKEND")
    celery_task_serializer: str = Field(default="json", alias="CELERY_TASK_SERIALIZER")
    celery_result_serializer: str = Field(default="json", alias="CELERY_RESULT_SERIALIZER")
    celery_accept_content: List[str] = Field(default=["json"], alias="CELERY_ACCEPT_CONTENT")
    celery_timezone: str = Field(default="UTC", alias="CELERY_TIMEZONE")

    # User Management Service
    user_service_base_url: str = Field(default="http://localhost:8001", alias="USER_SERVICE_BASE_URL")
    user_service_api_prefix: str = Field(default="/api/v1", alias="USER_SERVICE_API_PREFIX")
    user_service_timeout: int = Field(default=30, alias="USER_SERVICE_TIMEOUT")
    admin_email: str = Field(..., alias="ADMIN_EMAIL")
    admin_password: str = Field(..., alias="ADMIN_PASSWORD")

    # AI Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2000, alias="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")

    # Alternative AI providers
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")

    # File Storage
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    max_file_size: int = Field(default=10485760, alias="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(default=["pdf", "doc", "docx", "txt"], alias="ALLOWED_EXTENSIONS")
    static_files_dir: str = Field(default="static", alias="STATIC_FILES_DIR")

    # PDF Generation
    pdf_engine: str = Field(default="weasyprint", alias="PDF_ENGINE")
    pdf_dpi: int = Field(default=300, alias="PDF_DPI")
    pdf_timeout: int = Field(default=30, alias="PDF_TIMEOUT")

    # Email
    smtp_host: str = Field(default="localhost", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, alias="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, alias="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, alias="SMTP_USE_TLS")
    from_email: str = Field(default="noreply@example.com", alias="FROM_EMAIL")

    # Security
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=30, alias="REFRESH_TOKEN_EXPIRE_DAYS")
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, alias="RATE_LIMIT_BURST")

    # Monitoring
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    sentry_dsn: Optional[str] = Field(default=None, alias="SENTRY_DSN")
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")

    # Feature Flags
    enable_ai_analysis: bool = Field(default=True, alias="ENABLE_AI_ANALYSIS")
    enable_cover_letter: bool = Field(default=True, alias="ENABLE_COVER_LETTER")
    enable_template_customization: bool = Field(default=True, alias="ENABLE_TEMPLATE_CUSTOMIZATION")
    enable_bulk_operations: bool = Field(default=False, alias="ENABLE_BULK_OPERATIONS")

    # External APIs
    linkedin_api_key: Optional[str] = Field(default=None, alias="LINKEDIN_API_KEY")
    indeed_api_key: Optional[str] = Field(default=None, alias="INDEED_API_KEY")
    glassdoor_api_key: Optional[str] = Field(default=None, alias="GLASSDOOR_API_KEY")

    # Resume Analysis
    min_resume_score: int = Field(default=60, alias="MIN_RESUME_SCORE")
    ats_keywords_weight: float = Field(default=0.3, alias="ATS_KEYWORDS_WEIGHT")
    content_quality_weight: float = Field(default=0.4, alias="CONTENT_QUALITY_WEIGHT")
    format_weight: float = Field(default=0.3, alias="FORMAT_WEIGHT")

    # Cache
    cache_default_ttl: int = Field(default=300, alias="CACHE_DEFAULT_TTL")
    cache_long_ttl: int = Field(default=3600, alias="CACHE_LONG_TTL")
    cache_short_ttl: int = Field(default=60, alias="CACHE_SHORT_TTL")

    # Background Tasks
    task_retry_delay: int = Field(default=60, alias="TASK_RETRY_DELAY")
    task_max_retries: int = Field(default=3, alias="TASK_MAX_RETRIES")
    task_soft_time_limit: int = Field(default=300, alias="TASK_SOFT_TIME_LIMIT")
    task_time_limit: int = Field(default=600, alias="TASK_TIME_LIMIT")

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("allowed_extensions", pre=True)
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from string or list."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return [ext.lower() for ext in v]

    @validator("celery_accept_content", pre=True)
    def parse_celery_accept_content(cls, v):
        """Parse Celery accept content from string or list."""
        if isinstance(v, str):
            return [content.strip() for content in v.split(",")]
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment.lower() == "testing"

    @property
    def user_service_url(self) -> str:
        """Get complete user service URL."""
        return f"{self.user_service_base_url}{self.user_service_api_prefix}"

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL for Alembic."""
        return self.database_url.replace("+asyncpg", "")

    class Config:
        """Pydantic configuration."""
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()