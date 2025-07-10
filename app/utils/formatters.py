"""
Data formatting utilities for the AI Resume Platform.
"""

import re
import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal


class DateFormatter:
    """Date formatting utilities."""

    @staticmethod
    def format_date(date_obj: Union[datetime, date, str], format_str: str = "%B %Y") -> str:
        """Format date object to string."""
        if isinstance(date_obj, str):
            try:
                date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
            except ValueError:
                return date_obj

        if isinstance(date_obj, (datetime, date)):
            return date_obj.strftime(format_str)

        return str(date_obj)

    @staticmethod
    def format_date_range(start_date: Any, end_date: Any = None, format_str: str = "%B %Y") -> str:
        """Format date range for display."""
        start_formatted = DateFormatter.format_date(start_date, format_str)

        if end_date and str(end_date).lower() not in ['present', 'current', '']:
            end_formatted = DateFormatter.format_date(end_date, format_str)
            return f"{start_formatted} - {end_formatted}"
        else:
            return f"{start_formatted} - Present"

    @staticmethod
    def parse_date_string(date_str: str) -> Optional[datetime]:
        """Parse various date string formats."""
        if not date_str:
            return None

        # Common date formats to try
        formats = [
            "%Y-%m-%d",
            "%Y-%m",
            "%Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %Y",
            "%b %Y",
            "%Y-%m-%d %H:%M:%S"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    @staticmethod
    def calculate_duration(start_date: Any, end_date: Any = None) -> str:
        """Calculate and format duration between dates."""
        start = DateFormatter.parse_date_string(str(start_date)) if isinstance(start_date, str) else start_date
        end = DateFormatter.parse_date_string(str(end_date)) if isinstance(end_date, str) else end_date

        if not start:
            return ""

        if not end or str(end_date).lower() in ['present', 'current']:
            end = datetime.now()

        if isinstance(start, (datetime, date)) and isinstance(end, (datetime, date)):
            delta = end - start
            years = delta.days // 365
            months = (delta.days % 365) // 30

            if years > 0:
                if months > 0:
                    return f"{years} year{'s' if years != 1 else ''}, {months} month{'s' if months != 1 else ''}"
                else:
                    return f"{years} year{'s' if years != 1 else ''}"
            elif months > 0:
                return f"{months} month{'s' if months != 1 else ''}"
            else:
                return "Less than a month"

        return ""


class TextFormatter:
    """Text formatting utilities."""

    @staticmethod
    def capitalize_title(text: str) -> str:
        """Capitalize title properly."""
        if not text:
            return ""

        # Words that should not be capitalized (except at the beginning)
        articles_prepositions = {
            'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in',
            'nor', 'of', 'on', 'or', 'so', 'the', 'to', 'up', 'yet'
        }

        words = text.lower().split()
        capitalized_words = []

        for i, word in enumerate(words):
            if i == 0 or word not in articles_prepositions:
                capitalized_words.append(word.capitalize())
            else:
                capitalized_words.append(word)

        return ' '.join(capitalized_words)

    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to specified length."""
        if not text or len(text) <= max_length:
            return text

        truncated = text[:max_length - len(suffix)].rstrip()
        return truncated + suffix

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean excessive whitespace from text."""
        if not text:
            return ""

        # Replace multiple whitespace characters with single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()

    @staticmethod
    def format_bullet_points(items: List[str]) -> str:
        """Format list items as bullet points."""
        if not items:
            return ""

        return '\n'.join(f"• {item.strip()}" for item in items if item.strip())

    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text."""
        if not text:
            return []

        # Remove punctuation and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned_text.split()

        # Filter out short words and common words
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'do', 'does', 'did', 'get', 'got', 'make', 'made'
        }

        keywords = []
        for word in words:
            if len(word) >= min_length and word not in common_words:
                keywords.append(word)

        return list(set(keywords))  # Remove duplicates

    @staticmethod
    def format_phone_number(phone: str) -> str:
        """Format phone number for display."""
        if not phone:
            return ""

        # Remove all non-digit characters
        digits = ''.join(filter(str.isdigit, phone))

        # Format based on length
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            return phone  # Return original if can't format

    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-friendly slug."""
        if not text:
            return ""

        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug.strip('-')


class NumberFormatter:
    """Number formatting utilities."""

    @staticmethod
    def format_percentage(value: Union[int, float], decimal_places: int = 1) -> str:
        """Format number as percentage."""
        if value is None:
            return ""

        try:
            return f"{float(value):.{decimal_places}f}%"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def format_currency(amount: Union[int, float], currency: str = "USD") -> str:
        """Format number as currency."""
        if amount is None:
            return ""

        try:
            amount = float(amount)
            if currency.upper() == "USD":
                return f"${amount:,.2f}"
            else:
                return f"{amount:,.2f} {currency.upper()}"
        except (ValueError, TypeError):
            return str(amount)

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    @staticmethod
    def format_score(score: Union[int, float], max_score: int = 100) -> str:
        """Format score with proper formatting."""
        if score is None:
            return "N/A"

        try:
            score = float(score)
            return f"{score:.1f}/{max_score}"
        except (ValueError, TypeError):
            return str(score)


class ResumeFormatter:
    """Resume-specific formatting utilities."""

    @staticmethod
    def format_work_experience(experience: Dict[str, Any]) -> Dict[str, Any]:
        """Format work experience entry for display."""
        formatted = experience.copy()

        # Format title
        if formatted.get("title"):
            formatted["title"] = TextFormatter.capitalize_title(formatted["title"])

        # Format company
        if formatted.get("company"):
            formatted["company"] = TextFormatter.capitalize_title(formatted["company"])

        # Format date range
        if formatted.get("start_date"):
            formatted["date_range"] = DateFormatter.format_date_range(
                formatted["start_date"],
                formatted.get("end_date")
            )

        # Calculate duration
        if formatted.get("start_date"):
            formatted["duration"] = DateFormatter.calculate_duration(
                formatted["start_date"],
                formatted.get("end_date")
            )

        # Format achievements as bullet points
        if formatted.get("achievements"):
            formatted["achievements_formatted"] = TextFormatter.format_bullet_points(
                formatted["achievements"]
            )

        return formatted

    @staticmethod
    def format_education(education: Dict[str, Any]) -> Dict[str, Any]:
        """Format education entry for display."""
        formatted = education.copy()

        # Format degree
        if formatted.get("degree"):
            formatted["degree"] = TextFormatter.capitalize_title(formatted["degree"])

        # Format institution
        if formatted.get("institution"):
            formatted["institution"] = TextFormatter.capitalize_title(formatted["institution"])

        # Format field
        if formatted.get("field"):
            formatted["field"] = TextFormatter.capitalize_title(formatted["field"])

        # Format graduation date
        if formatted.get("graduation_date"):
            formatted["graduation_date_formatted"] = DateFormatter.format_date(
                formatted["graduation_date"], "%B %Y"
            )

        # Format GPA
        if formatted.get("gpa"):
            try:
                gpa = float(formatted["gpa"])
                formatted["gpa_formatted"] = f"{gpa:.2f}"
            except (ValueError, TypeError):
                formatted["gpa_formatted"] = str(formatted["gpa"])

        return formatted

    @staticmethod
    def format_skills(skills: List[Any]) -> List[Dict[str, Any]]:
        """Format skills list for display."""
        formatted_skills = []

        for skill in skills:
            if isinstance(skill, str):
                formatted_skills.append({
                    "name": TextFormatter.capitalize_title(skill),
                    "display_name": TextFormatter.capitalize_title(skill)
                })
            elif isinstance(skill, dict):
                formatted_skill = skill.copy()
                if formatted_skill.get("name"):
                    formatted_skill["display_name"] = TextFormatter.capitalize_title(
                        formatted_skill["name"]
                    )

                # Format proficiency level
                if formatted_skill.get("proficiency_level"):
                    level = formatted_skill["proficiency_level"]
                    level_names = {1: "Beginner", 2: "Novice", 3: "Intermediate", 4: "Advanced", 5: "Expert"}
                    formatted_skill["proficiency_name"] = level_names.get(level, "Unknown")

                formatted_skills.append(formatted_skill)

        return formatted_skills

    @staticmethod
    def format_projects(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format projects list for display."""
        formatted_projects = []

        for project in projects:
            formatted = project.copy()

            # Format name
            if formatted.get("name"):
                formatted["name"] = TextFormatter.capitalize_title(formatted["name"])

            # Truncate description if too long
            if formatted.get("description"):
                formatted["description_short"] = TextFormatter.truncate_text(
                    formatted["description"], 200
                )

            # Format technologies as tags
            if formatted.get("technologies"):
                formatted["technologies_formatted"] = ", ".join(formatted["technologies"])

            formatted_projects.append(formatted)

        return formatted_projects

    @staticmethod
    def format_personal_info(personal_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format personal information for display."""
        formatted = personal_info.copy()

        # Format name
        if formatted.get("first_name") and formatted.get("last_name"):
            formatted["full_name"] = f"{formatted['first_name']} {formatted['last_name']}"

        # Format phone
        if formatted.get("phone"):
            formatted["phone_formatted"] = TextFormatter.format_phone_number(formatted["phone"])

        # Format address
        address_parts = []
        for field in ["city", "state", "country"]:
            if formatted.get(field):
                address_parts.append(formatted[field])

        if address_parts:
            formatted["location"] = ", ".join(address_parts)

        return formatted


class CoverLetterFormatter:
    """Cover letter-specific formatting utilities."""

    @staticmethod
    def format_content(content: Dict[str, Any]) -> Dict[str, Any]:
        """Format cover letter content for display."""
        formatted = content.copy()

        # Clean whitespace in all text fields
        for field in ["opening_paragraph", "closing_paragraph"]:
            if formatted.get(field):
                formatted[field] = TextFormatter.clean_whitespace(formatted[field])

        # Format body paragraphs
        if formatted.get("body_paragraphs"):
            formatted["body_paragraphs"] = [
                TextFormatter.clean_whitespace(para)
                for para in formatted["body_paragraphs"]
            ]

        # Calculate word count
        all_text = []
        if formatted.get("opening_paragraph"):
            all_text.append(formatted["opening_paragraph"])
        if formatted.get("body_paragraphs"):
            all_text.extend(formatted["body_paragraphs"])
        if formatted.get("closing_paragraph"):
            all_text.append(formatted["closing_paragraph"])

        full_text = " ".join(all_text)
        formatted["word_count"] = len(full_text.split())

        return formatted

    @staticmethod
    def format_for_preview(cover_letter: Dict[str, Any]) -> str:
        """Format cover letter as plain text preview."""
        content = cover_letter.get("content", {})

        sections = []

        # Add header
        if cover_letter.get("hiring_manager_name"):
            sections.append(f"Dear {cover_letter['hiring_manager_name']},")
        else:
            sections.append("Dear Hiring Manager,")

        # Add opening
        if content.get("opening_paragraph"):
            sections.append(content["opening_paragraph"])

        # Add body paragraphs
        if content.get("body_paragraphs"):
            sections.extend(content["body_paragraphs"])

        # Add closing
        if content.get("closing_paragraph"):
            sections.append(content["closing_paragraph"])

        sections.append("Sincerely,")
        sections.append("[Your Name]")

        return "\n\n".join(sections)


class AnalysisFormatter:
    """Analysis results formatting utilities."""

    @staticmethod
    def format_score(score: Union[int, float]) -> Dict[str, Any]:
        """Format analysis score with color coding and description."""
        if score is None:
            return {"value": "N/A", "color": "gray", "description": "Not analyzed"}

        score = float(score)

        if score >= 90:
            return {"value": f"{score:.1f}", "color": "green", "description": "Excellent"}
        elif score >= 80:
            return {"value": f"{score:.1f}", "color": "blue", "description": "Very Good"}
        elif score >= 70:
            return {"value": f"{score:.1f}", "color": "yellow", "description": "Good"}
        elif score >= 60:
            return {"value": f"{score:.1f}", "color": "orange", "description": "Fair"}
        else:
            return {"value": f"{score:.1f}", "color": "red", "description": "Needs Improvement"}

    @staticmethod
    def format_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format suggestions for display."""
        formatted_suggestions = []

        for suggestion in suggestions:
            formatted = suggestion.copy()

            # Format priority with color
            priority = suggestion.get("priority", "medium")
            priority_colors = {
                "critical": "red",
                "high": "orange",
                "medium": "yellow",
                "low": "blue"
            }
            formatted["priority_color"] = priority_colors.get(priority, "gray")

            # Format impact score as percentage
            if suggestion.get("impact_score"):
                formatted["impact_percentage"] = NumberFormatter.format_percentage(
                    suggestion["impact_score"] * 100
                )

            # Truncate description if too long
            if suggestion.get("description"):
                formatted["description_short"] = TextFormatter.truncate_text(
                    suggestion["description"], 150
                )

            formatted_suggestions.append(formatted)

        return formatted_suggestions

    @staticmethod
    def format_keyword_analysis(keywords: Dict[str, Any]) -> Dict[str, Any]:
        """Format keyword analysis results."""
        formatted = {}

        # Format keyword lists
        for key in ["matching_keywords", "missing_keywords", "suggested_keywords"]:
            if keywords.get(key):
                formatted[f"{key}_count"] = len(keywords[key])
                formatted[f"{key}_display"] = ", ".join(keywords[key][:10])  # Show first 10

        # Format scores
        for score_key in ["keyword_density_score", "keyword_relevance_score", "overall_keyword_score"]:
            if keywords.get(score_key):
                formatted[score_key] = AnalysisFormatter.format_score(keywords[score_key])

        return formatted


class ExportFormatter:
    """Export formatting utilities."""

    @staticmethod
    def format_for_pdf(data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for PDF export."""
        formatted = data.copy()

        # Ensure all text fields are properly formatted
        if isinstance(formatted, dict):
            for key, value in formatted.items():
                if isinstance(value, str):
                    formatted[key] = TextFormatter.clean_whitespace(value)
                elif isinstance(value, list):
                    formatted[key] = [
                        TextFormatter.clean_whitespace(item) if isinstance(item, str) else item
                        for item in value
                    ]

        return formatted

    @staticmethod
    def format_for_json(data: Any) -> str:
        """Format data for JSON export."""

        def json_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(data, indent=2, default=json_serializer, ensure_ascii=False)

    @staticmethod
    def format_filename(base_name: str, format_type: str, timestamp: bool = True) -> str:
        """Format filename for export."""
        # Clean base name
        clean_name = TextFormatter.slugify(base_name)

        # Add timestamp if requested
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = f"{clean_name}_{timestamp_str}"

        return f"{clean_name}.{format_type.lower()}"


class ValidationFormatter:
    """Validation error formatting utilities."""

    @staticmethod
    def format_validation_errors(errors: Dict[str, List[str]]) -> Dict[str, Any]:
        """Format validation errors for API response."""
        formatted_errors = []

        for field, messages in errors.items():
            for message in messages:
                formatted_errors.append({
                    "field": field,
                    "message": message,
                    "code": "validation_error"
                })

        return {
            "errors": formatted_errors,
            "error_count": len(formatted_errors),
            "fields_with_errors": list(errors.keys())
        }

    @staticmethod
    def format_field_error(field: str, message: str, value: Any = None) -> Dict[str, Any]:
        """Format single field error."""
        error = {
            "field": field,
            "message": message,
            "code": "validation_error"
        }

        if value is not None:
            error["rejected_value"] = str(value)

        return error


# Utility functions for common formatting tasks
def format_resume_for_display(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format complete resume data for display."""
    formatted = resume_data.copy()

    # Format personal info
    if formatted.get("personal_info"):
        formatted["personal_info"] = ResumeFormatter.format_personal_info(formatted["personal_info"])

    # Format work experience
    if formatted.get("work_experience"):
        formatted["work_experience"] = [
            ResumeFormatter.format_work_experience(exp)
            for exp in formatted["work_experience"]
        ]

    # Format education
    if formatted.get("education"):
        formatted["education"] = [
            ResumeFormatter.format_education(edu)
            for edu in formatted["education"]
        ]

    # Format skills
    if formatted.get("skills"):
        formatted["skills"] = ResumeFormatter.format_skills(formatted["skills"])

    # Format projects
    if formatted.get("projects"):
        formatted["projects"] = ResumeFormatter.format_projects(formatted["projects"])

    return formatted


def format_cover_letter_for_display(cover_letter_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format complete cover letter data for display."""
    formatted = cover_letter_data.copy()

    # Format content
    if formatted.get("content"):
        formatted["content"] = CoverLetterFormatter.format_content(formatted["content"])

    # Add preview text
    formatted["preview_text"] = CoverLetterFormatter.format_for_preview(formatted)

    return formatted


def format_analysis_results(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format analysis results for display."""
    formatted = analysis_data.copy()

    # Format scores
    for score_field in ["overall_score", "ats_score", "content_score", "format_score"]:
        if formatted.get(score_field):
            formatted[f"{score_field}_formatted"] = AnalysisFormatter.format_score(
                formatted[score_field]
            )

    # Format suggestions
    if formatted.get("suggestions"):
        formatted["suggestions_formatted"] = AnalysisFormatter.format_suggestions(
            formatted["suggestions"]
        )

    # Format keyword analysis
    if formatted.get("keyword_analysis"):
        formatted["keyword_analysis_formatted"] = AnalysisFormatter.format_keyword_analysis(
            formatted["keyword_analysis"]
        )

    return formatted