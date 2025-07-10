"""
Data validation utilities for the AI Resume Platform.
"""

import re
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
from email_validator import validate_email, EmailNotValidError
from pydantic import ValidationError

from app.utils.exceptions import ValidationException


class ResumeValidator:
    """Validator for resume data."""

    @staticmethod
    def validate_personal_info(personal_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate personal information section."""
        errors = {}

        # Required fields
        required_fields = ["first_name", "last_name", "email"]
        for field in required_fields:
            if not personal_info.get(field):
                errors.setdefault(field, []).append(f"{field.replace('_', ' ').title()} is required")

        # Email validation
        if personal_info.get("email"):
            try:
                validate_email(personal_info["email"])
            except EmailNotValidError:
                errors.setdefault("email", []).append("Invalid email format")

        # Phone validation
        if personal_info.get("phone"):
            phone = personal_info["phone"]
            if not re.match(r'^[\+]?[1-9][\d]{0,15}$', re.sub(r'[\s\-\(\)]', '', phone)):
                errors.setdefault("phone", []).append("Invalid phone number format")

        # Name validation
        for name_field in ["first_name", "last_name"]:
            if personal_info.get(name_field):
                name = personal_info[name_field]
                if len(name) < 2:
                    errors.setdefault(name_field, []).append(
                        f"{name_field.replace('_', ' ').title()} must be at least 2 characters")
                if len(name) > 50:
                    errors.setdefault(name_field, []).append(
                        f"{name_field.replace('_', ' ').title()} must be less than 50 characters")
                if not re.match(r'^[a-zA-Z\s\-\'\.]+$', name):
                    errors.setdefault(name_field, []).append(
                        f"{name_field.replace('_', ' ').title()} contains invalid characters")

        # LinkedIn URL validation
        if personal_info.get("linkedin_url"):
            linkedin_url = personal_info["linkedin_url"]
            if not re.match(r'^https?://(www\.)?linkedin\.com/in/[\w\-]+/?$', linkedin_url):
                errors.setdefault("linkedin_url", []).append("Invalid LinkedIn URL format")

        # Website URL validation
        if personal_info.get("website"):
            website = personal_info["website"]
            if not re.match(r'^https?://[\w\-\.]+\.[\w]{2,}(/.*)?$', website):
                errors.setdefault("website", []).append("Invalid website URL format")

        return errors

    @staticmethod
    def validate_work_experience(work_experience: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Validate work experience section."""
        errors = {}

        if not work_experience:
            return errors

        for i, experience in enumerate(work_experience):
            prefix = f"work_experience[{i}]"

            # Required fields
            required_fields = ["title", "company", "start_date"]
            for field in required_fields:
                if not experience.get(field):
                    errors.setdefault(f"{prefix}.{field}", []).append(f"{field.replace('_', ' ').title()} is required")

            # Date validation
            start_date = experience.get("start_date")
            end_date = experience.get("end_date")

            if start_date:
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    if start_dt > datetime.now():
                        errors.setdefault(f"{prefix}.start_date", []).append("Start date cannot be in the future")
                except ValueError:
                    errors.setdefault(f"{prefix}.start_date", []).append("Invalid date format (use YYYY-MM-DD)")

            if end_date:
                try:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    if start_date:
                        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                        if end_dt < start_dt:
                            errors.setdefault(f"{prefix}.end_date", []).append("End date must be after start date")
                except ValueError:
                    errors.setdefault(f"{prefix}.end_date", []).append("Invalid date format (use YYYY-MM-DD)")

            # Title and company validation
            for field in ["title", "company"]:
                if experience.get(field):
                    value = experience[field]
                    if len(value) < 2:
                        errors.setdefault(f"{prefix}.{field}", []).append(
                            f"{field.title()} must be at least 2 characters")
                    if len(value) > 100:
                        errors.setdefault(f"{prefix}.{field}", []).append(
                            f"{field.title()} must be less than 100 characters")

            # Achievements validation
            if experience.get("achievements"):
                achievements = experience["achievements"]
                if not isinstance(achievements, list):
                    errors.setdefault(f"{prefix}.achievements", []).append("Achievements must be a list")
                elif len(achievements) > 10:
                    errors.setdefault(f"{prefix}.achievements", []).append("Maximum 10 achievements allowed")
                else:
                    for j, achievement in enumerate(achievements):
                        if not isinstance(achievement, str):
                            errors.setdefault(f"{prefix}.achievements[{j}]", []).append("Achievement must be a string")
                        elif len(achievement) > 500:
                            errors.setdefault(f"{prefix}.achievements[{j}]", []).append(
                                "Achievement must be less than 500 characters")

        return errors

    @staticmethod
    def validate_education(education: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Validate education section."""
        errors = {}

        if not education:
            return errors

        for i, edu in enumerate(education):
            prefix = f"education[{i}]"

            # Required fields
            required_fields = ["degree", "institution"]
            for field in required_fields:
                if not edu.get(field):
                    errors.setdefault(f"{prefix}.{field}", []).append(f"{field.replace('_', ' ').title()} is required")

            # Date validation
            graduation_date = edu.get("graduation_date")
            if graduation_date:
                try:
                    grad_dt = datetime.strptime(graduation_date, "%Y-%m-%d")
                    # Allow future graduation dates (for current students)
                except ValueError:
                    errors.setdefault(f"{prefix}.graduation_date", []).append("Invalid date format (use YYYY-MM-DD)")

            # GPA validation
            if edu.get("gpa"):
                try:
                    gpa = float(edu["gpa"])
                    if gpa < 0 or gpa > 4.0:
                        errors.setdefault(f"{prefix}.gpa", []).append("GPA must be between 0.0 and 4.0")
                except (ValueError, TypeError):
                    errors.setdefault(f"{prefix}.gpa", []).append("GPA must be a valid number")

            # Field validation
            for field in ["degree", "institution", "field"]:
                if edu.get(field):
                    value = edu[field]
                    if len(value) < 2:
                        errors.setdefault(f"{prefix}.{field}", []).append(
                            f"{field.title()} must be at least 2 characters")
                    if len(value) > 100:
                        errors.setdefault(f"{prefix}.{field}", []).append(
                            f"{field.title()} must be less than 100 characters")

        return errors

    @staticmethod
    def validate_skills(skills: List[Any]) -> Dict[str, List[str]]:
        """Validate skills section."""
        errors = {}

        if not skills:
            return errors

        if len(skills) > 50:
            errors["skills"] = ["Maximum 50 skills allowed"]
            return errors

        for i, skill in enumerate(skills):
            if isinstance(skill, str):
                # Simple string skill
                if len(skill) < 1:
                    errors.setdefault(f"skills[{i}]", []).append("Skill name cannot be empty")
                elif len(skill) > 50:
                    errors.setdefault(f"skills[{i}]", []).append("Skill name must be less than 50 characters")
            elif isinstance(skill, dict):
                # Structured skill object
                if not skill.get("name"):
                    errors.setdefault(f"skills[{i}].name", []).append("Skill name is required")
                elif len(skill["name"]) > 50:
                    errors.setdefault(f"skills[{i}].name", []).append("Skill name must be less than 50 characters")

                # Proficiency level validation
                if skill.get("proficiency_level"):
                    try:
                        level = int(skill["proficiency_level"])
                        if level < 1 or level > 5:
                            errors.setdefault(f"skills[{i}].proficiency_level", []).append(
                                "Proficiency level must be between 1 and 5")
                    except (ValueError, TypeError):
                        errors.setdefault(f"skills[{i}].proficiency_level", []).append(
                            "Proficiency level must be a number")

                # Years experience validation
                if skill.get("years_experience"):
                    try:
                        years = int(skill["years_experience"])
                        if years < 0 or years > 50:
                            errors.setdefault(f"skills[{i}].years_experience", []).append(
                                "Years of experience must be between 0 and 50")
                    except (ValueError, TypeError):
                        errors.setdefault(f"skills[{i}].years_experience", []).append(
                            "Years of experience must be a number")
            else:
                errors.setdefault(f"skills[{i}]", []).append("Skill must be a string or object")

        return errors

    @staticmethod
    def validate_projects(projects: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Validate projects section."""
        errors = {}

        if not projects:
            return errors

        for i, project in enumerate(projects):
            prefix = f"projects[{i}]"

            # Required fields
            if not project.get("name"):
                errors.setdefault(f"{prefix}.name", []).append("Project name is required")

            # Field length validation
            for field, max_length in [("name", 100), ("description", 1000)]:
                if project.get(field) and len(project[field]) > max_length:
                    errors.setdefault(f"{prefix}.{field}", []).append(
                        f"{field.title()} must be less than {max_length} characters")

            # URL validation
            if project.get("url"):
                url = project["url"]
                if not re.match(r'^https?://[\w\-\.]+\.[\w]{2,}(/.*)?$', url):
                    errors.setdefault(f"{prefix}.url", []).append("Invalid URL format")

            # Technologies validation
            if project.get("technologies"):
                technologies = project["technologies"]
                if not isinstance(technologies, list):
                    errors.setdefault(f"{prefix}.technologies", []).append("Technologies must be a list")
                elif len(technologies) > 20:
                    errors.setdefault(f"{prefix}.technologies", []).append("Maximum 20 technologies allowed")

        return errors


class CoverLetterValidator:
    """Validator for cover letter data."""

    @staticmethod
    def validate_content(content: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate cover letter content."""
        errors = {}

        # Check if content has required structure
        if not isinstance(content, dict):
            errors["content"] = ["Content must be an object"]
            return errors

        # Validate opening paragraph
        opening = content.get("opening_paragraph", "")
        if not opening:
            errors["content.opening_paragraph"] = ["Opening paragraph is required"]
        elif len(opening) < 50:
            errors["content.opening_paragraph"] = ["Opening paragraph must be at least 50 characters"]
        elif len(opening) > 1000:
            errors["content.opening_paragraph"] = ["Opening paragraph must be less than 1000 characters"]

        # Validate body paragraphs
        body_paragraphs = content.get("body_paragraphs", [])
        if not body_paragraphs or not isinstance(body_paragraphs, list):
            errors["content.body_paragraphs"] = ["At least one body paragraph is required"]
        else:
            if len(body_paragraphs) > 5:
                errors["content.body_paragraphs"] = ["Maximum 5 body paragraphs allowed"]

            for i, paragraph in enumerate(body_paragraphs):
                if not isinstance(paragraph, str):
                    errors[f"content.body_paragraphs[{i}]"] = ["Body paragraph must be a string"]
                elif len(paragraph) < 100:
                    errors[f"content.body_paragraphs[{i}]"] = ["Body paragraph must be at least 100 characters"]
                elif len(paragraph) > 1500:
                    errors[f"content.body_paragraphs[{i}]"] = ["Body paragraph must be less than 1500 characters"]

        # Validate closing paragraph
        closing = content.get("closing_paragraph", "")
        if not closing:
            errors["content.closing_paragraph"] = ["Closing paragraph is required"]
        elif len(closing) < 30:
            errors["content.closing_paragraph"] = ["Closing paragraph must be at least 30 characters"]
        elif len(closing) > 500:
            errors["content.closing_paragraph"] = ["Closing paragraph must be less than 500 characters"]

        return errors

    @staticmethod
    def validate_job_details(job_title: str = None, company_name: str = None) -> Dict[str, List[str]]:
        """Validate job details."""
        errors = {}

        if job_title:
            if len(job_title) < 2:
                errors["job_title"] = ["Job title must be at least 2 characters"]
            elif len(job_title) > 200:
                errors["job_title"] = ["Job title must be less than 200 characters"]

        if company_name:
            if len(company_name) < 2:
                errors["company_name"] = ["Company name must be at least 2 characters"]
            elif len(company_name) > 200:
                errors["company_name"] = ["Company name must be less than 200 characters"]

        return errors


class FileValidator:
    """Validator for file uploads."""

    @staticmethod
    def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
        """Validate file type based on extension."""
        if not filename:
            return False

        file_ext = filename.lower().split('.')[-1]
        return file_ext in [ext.lower() for ext in allowed_types]

    @staticmethod
    def validate_file_size(file_size: int, max_size: int) -> bool:
        """Validate file size."""
        return 0 < file_size <= max_size

    @staticmethod
    def validate_filename(filename: str) -> Dict[str, List[str]]:
        """Validate filename format and safety."""
        errors = {}

        if not filename:
            errors["filename"] = ["Filename is required"]
            return errors

        # Check length
        if len(filename) > 255:
            errors["filename"] = ["Filename must be less than 255 characters"]

        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        if any(char in filename for char in dangerous_chars):
            errors["filename"] = ["Filename contains invalid characters"]

        # Check for dangerous extensions
        dangerous_exts = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.js', '.vbs']
        file_ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
        if file_ext in dangerous_exts:
            errors["filename"] = ["File type not allowed for security reasons"]

        return errors


class CommonValidator:
    """Common validation utilities."""

    @staticmethod
    def validate_uuid(value: str) -> bool:
        """Validate UUID format."""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_date_string(date_string: str, format_str: str = "%Y-%m-%d") -> bool:
        """Validate date string format."""
        try:
            datetime.strptime(date_string, format_str)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None

    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength."""
        issues = []
        score = 0

        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        elif len(password) >= 12:
            score += 25
        else:
            score += 15

        if re.search(r'[a-z]', password):
            score += 20
        else:
            issues.append("Password must contain lowercase letters")

        if re.search(r'[A-Z]', password):
            score += 20
        else:
            issues.append("Password must contain uppercase letters")

        if re.search(r'\d', password):
            score += 20
        else:
            issues.append("Password must contain numbers")

        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 15
        else:
            issues.append("Password must contain special characters")

        return {
            "score": min(score, 100),
            "is_strong": score >= 70 and len(issues) == 0,
            "issues": issues
        }

    @staticmethod
    def sanitize_input(text: str, max_length: int = None) -> str:
        """Sanitize user input."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Truncate if too long
        if max_length and len(text) > max_length:
            text = text[:max_length]

        return text

    @staticmethod
    def validate_pagination_params(page: int, size: int, max_size: int = 100) -> Dict[str, List[str]]:
        """Validate pagination parameters."""
        errors = {}

        if page < 1:
            errors["page"] = ["Page must be greater than 0"]

        if size < 1:
            errors["size"] = ["Size must be greater than 0"]
        elif size > max_size:
            errors["size"] = [f"Size must be less than or equal to {max_size}"]

        return errors


# Decorator for automatic validation
def validate_data(validator_func):
    """Decorator to automatically validate data and raise ValidationException if errors found."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract data to validate (assume it's the first argument after self)
            data = args[1] if len(args) > 1 else kwargs.get('data')

            if data is not None:
                errors = validator_func(data)
                if errors:
                    raise ValidationException(
                        message="Validation failed",
                        field_errors=errors
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Validation schemas for complex objects
class ValidationSchema:
    """Base validation schema."""

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate data against schema."""
        raise NotImplementedError


class ResumeValidationSchema(ValidationSchema):
    """Complete resume validation schema."""

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate complete resume data."""
        all_errors = {}

        # Validate personal info
        if data.get("personal_info"):
            errors = ResumeValidator.validate_personal_info(data["personal_info"])
            all_errors.update(errors)

        # Validate work experience
        if data.get("work_experience"):
            errors = ResumeValidator.validate_work_experience(data["work_experience"])
            all_errors.update(errors)

        # Validate education
        if data.get("education"):
            errors = ResumeValidator.validate_education(data["education"])
            all_errors.update(errors)

        # Validate skills
        if data.get("skills"):
            errors = ResumeValidator.validate_skills(data["skills"])
            all_errors.update(errors)

        # Validate projects
        if data.get("projects"):
            errors = ResumeValidator.validate_projects(data["projects"])
            all_errors.update(errors)

        return all_errors


class CoverLetterValidationSchema(ValidationSchema):
    """Complete cover letter validation schema."""

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate complete cover letter data."""
        all_errors = {}

        # Validate content
        if data.get("content"):
            errors = CoverLetterValidator.validate_content(data["content"])
            all_errors.update(errors)

        # Validate job details
        errors = CoverLetterValidator.validate_job_details(
            data.get("job_title"),
            data.get("company_name")
        )
        all_errors.update(errors)

        return all_errors