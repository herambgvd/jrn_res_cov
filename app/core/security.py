"""
Security utilities for password hashing, token generation, and data validation.
"""

import secrets
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from passlib.context import CryptContext
from passlib.hash import bcrypt
import jwt
from cryptography.fernet import Fernet

from app.config import settings


class SecurityManager:
    """Security manager for password hashing, token generation, and encryption."""

    def __init__(self):
        # Password hashing context
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )

        # Token generation
        self.algorithm = settings.jwt_algorithm
        self.secret_key = settings.secret_key

        # Encryption for sensitive data
        self._fernet = None
        if hasattr(settings, 'encryption_key'):
            self._fernet = Fernet(settings.encryption_key.encode())

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def generate_password_reset_token(self, user_id: str) -> str:
        """Generate a password reset token."""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "type": "password_reset",
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Verify a password reset token and return user_id."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "password_reset":
                return None
            return payload.get("user_id")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def generate_verification_token(self, user_id: str, purpose: str = "email_verification") -> str:
        """Generate a verification token."""
        payload = {
            "user_id": user_id,
            "purpose": purpose,
            "exp": datetime.utcnow() + timedelta(hours=48),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_verification_token(self, token: str, purpose: str = "email_verification") -> Optional[str]:
        """Verify a verification token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("purpose") != purpose:
                return None
            return payload.get("user_id")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def generate_share_token(self, length: int = 32) -> str:
        """Generate a secure random token for sharing."""
        return secrets.token_urlsafe(length)

    def generate_api_key(self, prefix: str = "jr_") -> str:
        """Generate an API key."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}{random_part}"

    def hash_file_content(self, content: bytes) -> str:
        """Generate SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self._fernet:
            raise ValueError("Encryption key not configured")
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self._fernet:
            raise ValueError("Encryption key not configured")
        return self._fernet.decrypt(encrypted_data.encode()).decode()

    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength."""
        issues = []
        score = 0

        # Length check
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        elif len(password) >= 12:
            score += 20
        else:
            score += 10

        # Character variety checks
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        if has_lower:
            score += 15
        else:
            issues.append("Password should contain lowercase letters")

        if has_upper:
            score += 15
        else:
            issues.append("Password should contain uppercase letters")

        if has_digit:
            score += 15
        else:
            issues.append("Password should contain numbers")

        if has_special:
            score += 20
        else:
            issues.append("Password should contain special characters")

        # Common password check
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey"
        ]
        if password.lower() in common_passwords:
            issues.append("Password is too common")
            score = min(score, 30)

        # Determine strength
        if score >= 85:
            strength = "very_strong"
        elif score >= 70:
            strength = "strong"
        elif score >= 50:
            strength = "medium"
        elif score >= 30:
            strength = "weak"
        else:
            strength = "very_weak"

        return {
            "score": min(score, 100),
            "strength": strength,
            "is_valid": len(issues) == 0 and score >= 50,
            "issues": issues
        }

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        import re
        import os

        # Remove directory path
        filename = os.path.basename(filename)

        # Replace unsafe characters
        filename = re.sub(r'[^\w\-_\.]', '_', filename)

        # Remove multiple underscores
        filename = re.sub(r'_+', '_', filename)

        # Ensure it's not empty
        if not filename or filename == '.':
            filename = f"file_{secrets.token_hex(8)}"

        return filename

    def validate_file_type(self, filename: str, allowed_types: List[str] = None) -> bool:
        """Validate file type based on extension."""
        if allowed_types is None:
            allowed_types = settings.allowed_extensions

        if not filename:
            return False

        file_ext = filename.lower().split('.')[-1]
        return file_ext in [ext.lower() for ext in allowed_types]

    def generate_csrf_token(self) -> str:
        """Generate CSRF token."""
        return secrets.token_urlsafe(32)

    def create_signed_url(self, path: str, expires_in: int = 3600) -> str:
        """Create a signed URL that expires."""
        timestamp = int((datetime.utcnow() + timedelta(seconds=expires_in)).timestamp())

        # Create signature
        message = f"{path}:{timestamp}"
        signature = hashlib.hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{path}?expires={timestamp}&signature={signature}"

    def verify_signed_url(self, path: str, expires: str, signature: str) -> bool:
        """Verify a signed URL."""
        try:
            timestamp = int(expires)

            # Check if expired
            if datetime.utcnow().timestamp() > timestamp:
                return False

            # Verify signature
            message = f"{path}:{timestamp}"
            expected_signature = hashlib.hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            return secrets.compare_digest(signature, expected_signature)

        except (ValueError, TypeError):
            return False

    def rate_limit_key(self, identifier: str, action: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{action}:{identifier}"

    def generate_otp(self, length: int = 6) -> str:
        """Generate a numeric OTP."""
        return ''.join(secrets.choice('0123456789') for _ in range(length))


# Global security manager instance
security_manager = SecurityManager()


def get_password_hash(password: str) -> str:
    """Convenience function to hash password."""
    return security_manager.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Convenience function to verify password."""
    return security_manager.verify_password(plain_password, hashed_password)


def generate_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate a JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode a JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# Data masking functions
def mask_email(email: str) -> str:
    """Mask email address for privacy."""
    if not email or '@' not in email:
        return email

    local, domain = email.split('@', 1)
    if len(local) <= 2:
        masked_local = '*' * len(local)
    else:
        masked_local = local[0] + '*' * (len(local) - 2) + local[-1]

    return f"{masked_local}@{domain}"


def mask_phone(phone: str) -> str:
    """Mask phone number for privacy."""
    if not phone:
        return phone

    # Remove non-digit characters
    digits = ''.join(filter(str.isdigit, phone))

    if len(digits) < 4:
        return '*' * len(phone)

    # Show last 4 digits
    masked = '*' * (len(digits) - 4) + digits[-4:]

    # Restore original format
    result = phone
    for i, char in enumerate(phone):
        if char.isdigit():
            digit_index = len([c for c in phone[:i] if c.isdigit()])
            if digit_index < len(masked):
                result = result[:i] + masked[digit_index] + result[i + 1:]

    return result


# Input sanitization
def sanitize_html(text: str) -> str:
    """Basic HTML sanitization."""
    import html

    if not text:
        return text

    # Escape HTML characters
    sanitized = html.escape(text)

    # Remove potentially dangerous protocols
    dangerous_protocols = ['javascript:', 'data:', 'vbscript:']
    for protocol in dangerous_protocols:
        sanitized = sanitized.replace(protocol, '')

    return sanitized


def sanitize_sql_input(text: str) -> str:
    """Basic SQL injection prevention."""
    if not text:
        return text

    # Remove or escape common SQL injection patterns
    dangerous_patterns = [
        '--', ';', '/*', '*/', 'xp_', 'sp_', 'exec', 'execute',
        'union', 'select', 'insert', 'update', 'delete', 'drop'
    ]

    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern.lower(), '')
        sanitized = sanitized.replace(pattern.upper(), '')

    return sanitized.strip()


# Security headers
def get_security_headers() -> Dict[str, str]:
    """Get security headers for HTTP responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": "default-src 'self'",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }


# File validation
def is_safe_file(filename: str, content: bytes = None) -> bool:
    """Check if file is safe to process."""
    if not filename:
        return False

    # Check file extension
    allowed_extensions = settings.allowed_extensions
    file_ext = filename.lower().split('.')[-1]

    if file_ext not in [ext.lower() for ext in allowed_extensions]:
        return False

    # Check file size if content provided
    if content and len(content) > settings.max_file_size:
        return False

    # Check for malicious file signatures
    if content:
        malicious_signatures = [
            b'<script',
            b'javascript:',
            b'<?php',
            b'<%',
            b'exec(',
            b'system(',
            b'shell_exec('
        ]

        content_lower = content.lower()
        for signature in malicious_signatures:
            if signature in content_lower:
                return False

    return True