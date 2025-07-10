"""
PDF generation service for resumes and cover letters using WeasyPrint and templates.
"""

import os
import uuid
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import weasyprint
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

from app.config import settings
from app.models.resume import Resume, ResumeTemplate
from app.models.cover_letter import CoverLetter, CoverLetterTemplate
from app.core.redis import cache_manager
from app.utils.exceptions import AppException


class PDFService:
    """Service for generating PDFs from templates."""

    def __init__(self):
        self.template_dir = Path(__file__).parent.parent / "templates"
        self.static_dir = Path(__file__).parent.parent / "static"
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader([str(self.template_dir)]),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Add custom filters
        self._add_custom_filters()

        # Font configuration for WeasyPrint
        self.font_config = FontConfiguration()

        # Default styles
        self.base_css = """
        @page {
            size: A4;
            margin: 0.75in;
        }

        body {
            font-family: 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #333;
            margin: 0;
            padding: 0;
        }

        h1, h2, h3, h4, h5, h6 {
            margin-top: 0;
            margin-bottom: 0.5em;
            font-weight: bold;
        }

        h1 { font-size: 18pt; }
        h2 { font-size: 14pt; }
        h3 { font-size: 12pt; }

        p {
            margin: 0 0 0.5em 0;
        }

        .header {
            text-align: center;
            margin-bottom: 1em;
            border-bottom: 2px solid #333;
            padding-bottom: 0.5em;
        }

        .section {
            margin-bottom: 1em;
        }

        .section-title {
            font-weight: bold;
            font-size: 12pt;
            border-bottom: 1px solid #ccc;
            margin-bottom: 0.5em;
            padding-bottom: 0.2em;
        }

        .contact-info {
            font-size: 10pt;
            margin-top: 0.5em;
        }

        .job-entry, .education-entry {
            margin-bottom: 0.8em;
        }

        .job-title, .degree {
            font-weight: bold;
        }

        .company, .school {
            font-style: italic;
        }

        .date-range {
            float: right;
            font-style: italic;
            font-size: 10pt;
        }

        .skills-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5em;
        }

        .skill-item {
            background: #f0f0f0;
            padding: 0.2em 0.5em;
            border-radius: 3px;
            font-size: 10pt;
        }

        ul {
            margin: 0.5em 0;
            padding-left: 1.2em;
        }

        li {
            margin-bottom: 0.3em;
        }

        .clearfix::after {
            content: "";
            display: table;
            clear: both;
        }
        """

    async def generate_resume_pdf(
            self,
            resume: Resume,
            template_id: Optional[uuid.UUID] = None,
            custom_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate PDF for a resume.

        Args:
            resume: Resume instance
            template_id: Optional template override
            custom_settings: Custom generation settings

        Returns:
            Dict with file information
        """
        try:
            # Get template
            template = await self._get_resume_template(template_id or resume.template_id)

            # Prepare data for template
            template_data = await self._prepare_resume_data(resume)

            # Generate HTML
            html_content = await self._render_resume_template(template, template_data)

            # Generate PDF
            pdf_path = await self._generate_pdf_from_html(
                html_content,
                template.css_styles if template else None,
                custom_settings
            )

            # Calculate file size
            file_size = os.path.getsize(pdf_path)

            # Generate download URL
            file_name = f"resume_{resume.id}_{int(datetime.now().timestamp())}.pdf"
            download_url = f"/downloads/{file_name}"

            return {
                "file_url": download_url,
                "file_name": file_name,
                "file_path": str(pdf_path),
                "file_size": file_size,
                "format": "pdf",
                "expires_at": datetime.utcnow() + timedelta(hours=24)
            }

        except Exception as e:
            raise AppException(
                message=f"PDF generation failed: {str(e)}",
                status_code=500,
                error_code="PDF_GENERATION_FAILED"
            )

    async def generate_cover_letter_pdf(
            self,
            cover_letter: CoverLetter,
            template_id: Optional[uuid.UUID] = None,
            custom_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate PDF for a cover letter.

        Args:
            cover_letter: CoverLetter instance
            template_id: Optional template override
            custom_settings: Custom generation settings

        Returns:
            Dict with file information
        """
        try:
            # Get template
            template = await self._get_cover_letter_template(template_id or cover_letter.template_id)

            # Prepare data for template
            template_data = await self._prepare_cover_letter_data(cover_letter)

            # Generate HTML
            html_content = await self._render_cover_letter_template(template, template_data)

            # Generate PDF
            pdf_path = await self._generate_pdf_from_html(
                html_content,
                template.css_styles if template else None,
                custom_settings
            )

            # Calculate file size
            file_size = os.path.getsize(pdf_path)

            # Generate download URL
            file_name = f"cover_letter_{cover_letter.id}_{int(datetime.now().timestamp())}.pdf"
            download_url = f"/downloads/{file_name}"

            return {
                "file_url": download_url,
                "file_name": file_name,
                "file_path": str(pdf_path),
                "file_size": file_size,
                "format": "pdf",
                "expires_at": datetime.utcnow() + timedelta(hours=24)
            }

        except Exception as e:
            raise AppException(
                message=f"PDF generation failed: {str(e)}",
                status_code=500,
                error_code="PDF_GENERATION_FAILED"
            )

    async def generate_docx(
            self,
            document: Union[Resume, CoverLetter],
            custom_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate DOCX format (placeholder implementation).

        Args:
            document: Resume or CoverLetter instance
            custom_settings: Custom generation settings

        Returns:
            Dict with file information
        """
        # For now, generate PDF and return as placeholder
        # In production, you would use python-docx library
        if isinstance(document, Resume):
            return await self.generate_resume_pdf(document, custom_settings=custom_settings)
        else:
            return await self.generate_cover_letter_pdf(document, custom_settings=custom_settings)

    async def generate_html(
            self,
            document: Union[Resume, CoverLetter],
            template_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Generate HTML format.

        Args:
            document: Resume or CoverLetter instance
            template_id: Optional template override

        Returns:
            Dict with file information
        """
        try:
            if isinstance(document, Resume):
                template = await self._get_resume_template(template_id or document.template_id)
                template_data = await self._prepare_resume_data(document)
                html_content = await self._render_resume_template(template, template_data)
                file_prefix = "resume"
            else:
                template = await self._get_cover_letter_template(template_id or document.template_id)
                template_data = await self._prepare_cover_letter_data(document)
                html_content = await self._render_cover_letter_template(template, template_data)
                file_prefix = "cover_letter"

            # Save HTML file
            file_name = f"{file_prefix}_{document.id}_{int(datetime.now().timestamp())}.html"
            file_path = self.upload_dir / file_name

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            file_size = os.path.getsize(file_path)
            download_url = f"/downloads/{file_name}"

            return {
                "file_url": download_url,
                "file_name": file_name,
                "file_path": str(file_path),
                "file_size": file_size,
                "format": "html",
                "expires_at": datetime.utcnow() + timedelta(hours=24)
            }

        except Exception as e:
            raise AppException(
                message=f"HTML generation failed: {str(e)}",
                status_code=500,
                error_code="HTML_GENERATION_FAILED"
            )

    # Helper methods
    async def _get_resume_template(self, template_id: Optional[uuid.UUID]) -> Optional[ResumeTemplate]:
        """Get resume template by ID."""
        if not template_id:
            return None

        # This would fetch from database
        # For now, return None to use default template
        return None

    async def _get_cover_letter_template(self, template_id: Optional[uuid.UUID]) -> Optional[CoverLetterTemplate]:
        """Get cover letter template by ID."""
        if not template_id:
            return None

        # This would fetch from database
        # For now, return None to use default template
        return None

    async def _prepare_resume_data(self, resume: Resume) -> Dict[str, Any]:
        """Prepare resume data for template rendering."""
        return {
            "personal_info": resume.personal_info or {},
            "work_experience": resume.work_experience or [],
            "education": resume.education or [],
            "skills": resume.skills or [],
            "projects": resume.projects or [],
            "certifications": resume.certifications or [],
            "languages": resume.languages or [],
            "awards": resume.awards or [],
            "references": resume.references or [],
            "custom_sections": resume.custom_sections or [],
            "title": resume.title,
            "description": resume.description,
            "target_job_title": resume.target_job_title,
            "keywords": resume.keywords or [],
            "generated_at": datetime.now().strftime("%B %d, %Y")
        }

    async def _prepare_cover_letter_data(self, cover_letter: CoverLetter) -> Dict[str, Any]:
        """Prepare cover letter data for template rendering."""
        return {
            "content": cover_letter.content or {},
            "opening_paragraph": cover_letter.opening_paragraph or "",
            "body_paragraphs": cover_letter.body_paragraphs or [],
            "closing_paragraph": cover_letter.closing_paragraph or "",
            "job_title": cover_letter.job_title,
            "company_name": cover_letter.company_name,
            "hiring_manager_name": cover_letter.hiring_manager_name,
            "hiring_manager_title": cover_letter.hiring_manager_title,
            "company_address": cover_letter.company_address or {},
            "title": cover_letter.title,
            "type": cover_letter.type.value,
            "tone": cover_letter.tone.value,
            "generated_at": datetime.now().strftime("%B %d, %Y")
        }

    async def _render_resume_template(
            self,
            template: Optional[ResumeTemplate],
            data: Dict[str, Any]
    ) -> str:
        """Render resume template with data."""
        if template and template.html_template:
            # Use custom template
            jinja_template = self.jinja_env.from_string(template.html_template)
            return jinja_template.render(**data)
        else:
            # Use default template
            return await self._render_default_resume_template(data)

    async def _render_cover_letter_template(
            self,
            template: Optional[CoverLetterTemplate],
            data: Dict[str, Any]
    ) -> str:
        """Render cover letter template with data."""
        if template and template.html_template:
            # Use custom template
            jinja_template = self.jinja_env.from_string(template.html_template)
            return jinja_template.render(**data)
        else:
            # Use default template
            return await self._render_default_cover_letter_template(data)

    async def _render_default_resume_template(self, data: Dict[str, Any]) -> str:
        """Render default resume template."""
        personal = data.get("personal_info", {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Resume - {personal.get('first_name', '')} {personal.get('last_name', '')}</title>
        </head>
        <body>
            <div class="header">
                <h1>{personal.get('first_name', '')} {personal.get('last_name', '')}</h1>
                <div class="contact-info">
                    {personal.get('email', '')} | {personal.get('phone', '')} | {personal.get('location', '')}
                </div>
            </div>

            {await self._render_resume_summary(data)}
            {await self._render_work_experience(data)}
            {await self._render_education(data)}
            {await self._render_skills(data)}
            {await self._render_projects(data)}
            {await self._render_certifications(data)}
        </body>
        </html>
        """
        return html

    async def _render_default_cover_letter_template(self, data: Dict[str, Any]) -> str:
        """Render default cover letter template."""
        today = datetime.now().strftime("%B %d, %Y")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Cover Letter - {data.get('job_title', 'Position')}</title>
        </head>
        <body>
            <div class="header">
                <p style="text-align: right;">{today}</p>
                <p>{data.get('hiring_manager_name', 'Hiring Manager')}</p>
                <p>{data.get('company_name', 'Company')}</p>
                <br>
                <p>Dear {data.get('hiring_manager_name', 'Hiring Manager')},</p>
            </div>

            <div class="content">
                <p>{data.get('opening_paragraph', '')}</p>

                {' '.join([f'<p>{para}</p>' for para in data.get('body_paragraphs', [])])}

                <p>{data.get('closing_paragraph', '')}</p>

                <p>Sincerely,<br>
                [Your Name]</p>
            </div>
        </body>
        </html>
        """
        return html

    async def _render_resume_summary(self, data: Dict[str, Any]) -> str:
        """Render resume summary section."""
        description = data.get("description", "")
        if not description:
            return ""

        return f"""
        <div class="section">
            <div class="section-title">Professional Summary</div>
            <p>{description}</p>
        </div>
        """

    async def _render_work_experience(self, data: Dict[str, Any]) -> str:
        """Render work experience section."""
        experiences = data.get("work_experience", [])
        if not experiences:
            return ""

        html = '<div class="section"><div class="section-title">Work Experience</div>'

        for exp in experiences:
            html += f"""
            <div class="job-entry clearfix">
                <div class="job-title">{exp.get('title', '')}</div>
                <div class="date-range">{exp.get('start_date', '')} - {exp.get('end_date', 'Present')}</div>
                <div class="company">{exp.get('company', '')}</div>
                <div class="location">{exp.get('location', '')}</div>
                <ul>
            """

            for achievement in exp.get('achievements', []):
                html += f"<li>{achievement}</li>"

            html += "</ul></div>"

        html += "</div>"
        return html

    async def _render_education(self, data: Dict[str, Any]) -> str:
        """Render education section."""
        education = data.get("education", [])
        if not education:
            return ""

        html = '<div class="section"><div class="section-title">Education</div>'

        for edu in education:
            html += f"""
            <div class="education-entry clearfix">
                <div class="degree">{edu.get('degree', '')} in {edu.get('field', '')}</div>
                <div class="date-range">{edu.get('graduation_date', '')}</div>
                <div class="school">{edu.get('institution', '')}</div>
                <div class="location">{edu.get('location', '')}</div>
                {f"<div>GPA: {edu.get('gpa', '')}</div>" if edu.get('gpa') else ""}
            </div>
            """

        html += "</div>"
        return html

    async def _render_skills(self, data: Dict[str, Any]) -> str:
        """Render skills section."""
        skills = data.get("skills", [])
        if not skills:
            return ""

        html = '<div class="section"><div class="section-title">Skills</div><div class="skills-list">'

        for skill in skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', skill.get('skill_name', ''))
            else:
                skill_name = str(skill)

            html += f'<span class="skill-item">{skill_name}</span>'

        html += "</div></div>"
        return html

    async def _render_projects(self, data: Dict[str, Any]) -> str:
        """Render projects section."""
        projects = data.get("projects", [])
        if not projects:
            return ""

        html = '<div class="section"><div class="section-title">Projects</div>'

        for project in projects:
            html += f"""
            <div class="project-entry">
                <div class="project-title">{project.get('name', '')}</div>
                <div class="project-description">{project.get('description', '')}</div>
                {f"<div class='project-technologies'>Technologies: {', '.join(project.get('technologies', []))}</div>" if project.get('technologies') else ""}
                {f"<div class='project-url'><a href='{project.get('url', '')}'>{project.get('url', '')}</a></div>" if project.get('url') else ""}
            </div>
            """

        html += "</div>"
        return html

    async def _render_certifications(self, data: Dict[str, Any]) -> str:
        """Render certifications section."""
        certifications = data.get("certifications", [])
        if not certifications:
            return ""

        html = '<div class="section"><div class="section-title">Certifications</div>'

        for cert in certifications:
            html += f"""
            <div class="certification-entry">
                <div class="cert-name">{cert.get('name', '')}</div>
                <div class="cert-issuer">{cert.get('issuer', '')}</div>
                <div class="cert-date">{cert.get('date', '')}</div>
                {f"<div class='cert-id'>ID: {cert.get('credential_id', '')}</div>" if cert.get('credential_id') else ""}
            </div>
            """

        html += "</div>"
        return html

    async def _generate_pdf_from_html(
            self,
            html_content: str,
            custom_css: Optional[str] = None,
            custom_settings: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Generate PDF from HTML content."""
        try:
            # Combine CSS
            css_content = self.base_css
            if custom_css:
                css_content += "\n" + custom_css

            # Create temporary file for PDF
            pdf_file = tempfile.NamedTemporaryFile(
                suffix='.pdf',
                delete=False,
                dir=self.upload_dir
            )
            pdf_path = Path(pdf_file.name)
            pdf_file.close()

            # Configure WeasyPrint
            html_doc = HTML(string=html_content)
            css_doc = CSS(string=css_content, font_config=self.font_config)

            # Generate PDF
            document = html_doc.render([css_doc], font_config=self.font_config)
            document.write_pdf(str(pdf_path))

            return pdf_path

        except Exception as e:
            # Clean up on error
            if 'pdf_path' in locals() and pdf_path.exists():
                pdf_path.unlink()
            raise e

    def _add_custom_filters(self):
        """Add custom Jinja2 filters."""

        def format_date(date_str, format_str="%B %Y"):
            """Format date string."""
            try:
                if isinstance(date_str, str):
                    # Try to parse common date formats
                    for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            return date_obj.strftime(format_str)
                        except ValueError:
                            continue
                return date_str
            except:
                return date_str

        def truncate_words(text, max_words=50):
            """Truncate text to specified number of words."""
            if not text:
                return ""
            words = text.split()
            if len(words) <= max_words:
                return text
            return " ".join(words[:max_words]) + "..."

        def format_phone(phone):
            """Format phone number."""
            if not phone:
                return ""
            # Simple phone formatting
            digits = ''.join(filter(str.isdigit, phone))
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            return phone

        # Register filters
        self.jinja_env.filters['format_date'] = format_date
        self.jinja_env.filters['truncate_words'] = truncate_words
        self.jinja_env.filters['format_phone'] = format_phone

    async def cleanup_expired_files(self):
        """Clean up expired PDF files."""
        try:
            # Remove files older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)

            for file_path in self.upload_dir.glob("*.pdf"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()

            for file_path in self.upload_dir.glob("*.html"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()

        except Exception as e:
            # Log error but don't raise
            pass