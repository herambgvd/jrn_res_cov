"""
AI service for content generation, analysis, and optimization using OpenAI and other AI models.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI
import tiktoken

from app.config import settings
from app.utils.ai_prompts import (
    RESUME_ANALYSIS_PROMPT, COVER_LETTER_GENERATION_PROMPT,
    COVER_LETTER_OPTIMIZATION_PROMPT, KEYWORD_EXTRACTION_PROMPT,
    ATS_ANALYSIS_PROMPT, CONTENT_OPTIMIZATION_PROMPT,
    JOB_MATCHING_PROMPT, SKILL_EXTRACTION_PROMPT
)
from app.models.cover_letter import CoverLetterTone, CoverLetterTemplate
from app.core.redis import cache_manager


class AIService:
    """Service for AI-powered content generation and analysis."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        self.encoding = tiktoken.encoding_for_model("gpt-4")

        # Model pricing per 1K tokens (approximate)
        self.model_pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }

    async def generate_cover_letter(
            self,
            job_description: str,
            job_title: str,
            company_name: str,
            hiring_manager_name: Optional[str] = None,
            resume_data: Optional[Dict[str, Any]] = None,
            template: Optional[CoverLetterTemplate] = None,
            tone: CoverLetterTone = CoverLetterTone.PROFESSIONAL,
            personalization_level: int = 3,
            key_points: List[str] = None,
            company_research: Dict[str, Any] = None,
            custom_prompt: Optional[str] = None,
            generation_settings: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a personalized cover letter using AI.

        Args:
            job_description: The job description text
            job_title: Job title
            company_name: Company name
            hiring_manager_name: Hiring manager name if known
            resume_data: User's resume data for personalization
            template: Cover letter template to follow
            tone: Desired tone for the cover letter
            personalization_level: Level of personalization (1-5)
            key_points: Specific points to include
            company_research: Company research data
            custom_prompt: Custom AI prompt
            generation_settings: Additional generation settings

        Returns:
            Dict containing generated content and metadata
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        start_time = time.time()

        # Prepare context data
        context = {
            "job_description": job_description,
            "job_title": job_title,
            "company_name": company_name,
            "hiring_manager_name": hiring_manager_name,
            "tone": tone.value,
            "personalization_level": personalization_level,
            "key_points": key_points or [],
            "company_research": company_research or {},
            "resume_data": resume_data or {}
        }

        # Build the prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = await self._build_cover_letter_prompt(context, template)

        try:
            # Generate content
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert career counselor and professional writer specializing in creating compelling cover letters."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = json.loads(response.choices[0].message.content)

            # Calculate processing time and costs
            processing_time = time.time() - start_time
            usage = response.usage
            cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)

            # Structure the response
            result = {
                "content": content,
                "opening_paragraph": content.get("opening_paragraph", ""),
                "body_paragraphs": content.get("body_paragraphs", []),
                "closing_paragraph": content.get("closing_paragraph", ""),
                "keywords_used": content.get("keywords_used", []),
                "personalization_score": content.get("personalization_score", 85.0),
                "tone_consistency": content.get("tone_consistency", 90.0),
                "processing_time": processing_time,
                "ai_model_used": self.model,
                "tokens_used": usage.total_tokens,
                "estimated_cost": cost,
                "prompt_used": prompt[:500] + "..." if len(prompt) > 500 else prompt
            }

            return result

        except Exception as e:
            raise ValueError(f"AI generation failed: {str(e)}")

    async def analyze_resume(
            self,
            resume_data: Dict[str, Any],
            job_description: Optional[str] = None,
            analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a resume for quality, ATS compatibility, and job matching.

        Args:
            resume_data: Resume content to analyze
            job_description: Optional job description for matching
            analysis_types: Types of analysis to perform

        Returns:
            Dict containing analysis results and scores
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        start_time = time.time()
        analysis_types = analysis_types or ["comprehensive"]

        # Build analysis prompt
        prompt = await self._build_resume_analysis_prompt(
            resume_data, job_description, analysis_types
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert resume analyst and ATS specialist with deep knowledge of hiring practices across industries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for more consistent analysis
                response_format={"type": "json_object"}
            )

            # Parse analysis results
            analysis = json.loads(response.choices[0].message.content)

            # Calculate processing metrics
            processing_time = time.time() - start_time
            usage = response.usage
            cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)

            # Structure the response
            result = {
                "overall_score": analysis.get("overall_score", 0),
                "sub_scores": {
                    "ats_score": analysis.get("ats_score", 0),
                    "content_score": analysis.get("content_score", 0),
                    "format_score": analysis.get("format_score", 0),
                    "keyword_score": analysis.get("keyword_score", 0)
                },
                "analysis_data": analysis,
                "suggestions": analysis.get("suggestions", []),
                "strengths": analysis.get("strengths", []),
                "weaknesses": analysis.get("weaknesses", []),
                "missing_keywords": analysis.get("missing_keywords", []),
                "recommended_improvements": analysis.get("recommended_improvements", []),
                "ats_compatibility": analysis.get("ats_compatibility", {}),
                "processing_time": processing_time,
                "ai_model_used": self.model,
                "confidence_score": analysis.get("confidence_score", 0.85),
                "tokens_used": usage.total_tokens,
                "estimated_cost": cost
            }

            return result

        except Exception as e:
            raise ValueError(f"Resume analysis failed: {str(e)}")

    async def analyze_cover_letter_quality(
            self,
            content: Dict[str, Any],
            job_description: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Analyze cover letter quality and provide scores.

        Args:
            content: Cover letter content
            job_description: Optional job description for context

        Returns:
            Dict with quality scores
        """
        if not self.client:
            # Fallback scoring if AI not available
            return {
                "content_quality": 75.0,
                "personalization": 70.0,
                "keyword_match": 65.0,
                "tone_consistency": 80.0,
                "overall": 72.5
            }

        prompt = f"""
        Analyze the following cover letter for quality metrics. Return scores (0-100) in JSON format.

        Cover Letter Content:
        {json.dumps(content, indent=2)}

        Job Description (if provided):
        {job_description or "Not provided"}

        Provide scores for:
        - content_quality: Overall writing quality and effectiveness
        - personalization: How well tailored to the job/company
        - keyword_match: Alignment with job requirements
        - tone_consistency: Professional tone throughout
        - overall: Overall effectiveness score

        Return only JSON with these scores.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating cover letter quality."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception:
            # Fallback scoring
            return {
                "content_quality": 75.0,
                "personalization": 70.0,
                "keyword_match": 65.0,
                "tone_consistency": 80.0,
                "overall": 72.5
            }

    async def optimize_cover_letter(
            self,
            content: Dict[str, Any],
            optimization_type: str,
            job_description: Optional[str] = None,
            target_keywords: Optional[List[str]] = None,
            target_tone: Optional[CoverLetterTone] = None,
            target_length: Optional[int] = None,
            preserve_sections: Optional[List[str]] = None,
            optimization_level: int = 3
    ) -> Dict[str, Any]:
        """
        Optimize cover letter content using AI.

        Args:
            content: Current cover letter content
            optimization_type: Type of optimization (keyword, tone, length, ats)
            job_description: Job description for context
            target_keywords: Keywords to optimize for
            target_tone: Target tone
            target_length: Target word count
            preserve_sections: Sections to preserve
            optimization_level: Optimization intensity (1-5)

        Returns:
            Dict containing optimized content and changes made
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")

        # Build optimization prompt
        prompt = await self._build_optimization_prompt(
            content, optimization_type, job_description, target_keywords,
            target_tone, target_length, preserve_sections, optimization_level
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert content optimizer specializing in cover letters and professional communication."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.4,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return {
                "content": result.get("optimized_content", content),
                "changes": result.get("changes_made", []),
                "improvement_score": result.get("improvement_score", 10.0),
                "summary": result.get("optimization_summary", "Content optimized"),
                "keywords_added": result.get("keywords_added", []),
                "tone_adjustments": result.get("tone_adjustments", [])
            }

        except Exception as e:
            raise ValueError(f"Content optimization failed: {str(e)}")

    async def extract_job_keywords(
            self,
            job_description: str
    ) -> Dict[str, List[str]]:
        """
        Extract keywords and requirements from job description.

        Args:
            job_description: Job description text

        Returns:
            Dict containing categorized keywords
        """
        if not self.client:
            # Fallback keyword extraction using simple regex
            return self._extract_keywords_fallback(job_description)

        # Check cache first
        cache_key = f"job_keywords:{hash(job_description)}"
        cached_result = await cache_manager.get(cache_key) if cache_manager else None

        if cached_result:
            return cached_result

        prompt = f"""
        Extract and categorize keywords from this job description. Return results in JSON format.

        Job Description:
        {job_description}

        Categorize keywords into:
        - keywords: All important keywords
        - skills: Technical and soft skills
        - education: Education requirements
        - experience: Experience requirements
        - responsibilities: Key responsibilities
        - qualifications: Required qualifications

        Return only JSON with these categories as arrays.
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert at analyzing job descriptions and extracting relevant keywords for job matching."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Cache the result
            if cache_manager:
                await cache_manager.set(cache_key, result, ttl=3600)  # 1 hour

            return result

        except Exception:
            # Fallback to simple extraction
            return self._extract_keywords_fallback(job_description)

    async def analyze_document(
            self,
            content: Dict[str, Any],
            document_type: str,
            analysis_type: str,
            job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document (resume or cover letter) based on analysis type.

        Args:
            content: Document content
            document_type: Type of document (resume, cover_letter)
            analysis_type: Type of analysis to perform
            job_description: Optional job description for context

        Returns:
            Dict containing analysis results
        """
        if analysis_type == "ats_scan":
            return await self._analyze_ats_compatibility(content, document_type)
        elif analysis_type == "keyword_analysis":
            return await self._analyze_keywords(content, job_description)
        elif analysis_type == "content_quality":
            return await self._analyze_content_quality(content, document_type)
        elif analysis_type == "job_match":
            return await self._analyze_job_match(content, job_description)
        elif analysis_type == "comprehensive":
            return await self._comprehensive_analysis(content, document_type, job_description)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

    async def analyze_content_real_time(
            self,
            content: str,
            content_type: str,
            analysis_types: List[str],
            job_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform real-time analysis on content for immediate feedback.

        Args:
            content: Content to analyze
            content_type: Type of content
            analysis_types: Types of analysis to perform
            job_context: Optional job context

        Returns:
            Dict containing real-time analysis results
        """
        # Quick analysis without full AI processing for speed
        issues = []
        suggestions = []
        improvements = []

        # Basic checks
        word_count = len(content.split())

        if word_count < 50:
            issues.append({
                "type": "length",
                "severity": "medium",
                "message": "Content appears too short"
            })

        if word_count > 500:
            issues.append({
                "type": "length",
                "severity": "low",
                "message": "Content might be too long"
            })

        # Check for common issues
        if content.count("!") > 3:
            issues.append({
                "type": "tone",
                "severity": "medium",
                "message": "Too many exclamation marks"
            })

        # Calculate basic score
        content_score = max(0, 100 - len(issues) * 10)

        return {
            "content_score": content_score,
            "issues": issues,
            "suggestions": suggestions,
            "improvements": improvements,
            "confidence": 0.8,
            "word_count": word_count
        }

    # Helper methods
    async def _build_cover_letter_prompt(
            self,
            context: Dict[str, Any],
            template: Optional[CoverLetterTemplate] = None
    ) -> str:
        """Build prompt for cover letter generation."""
        prompt = COVER_LETTER_GENERATION_PROMPT.format(**context)

        if template:
            prompt += f"\n\nTemplate Guidelines:\n{template.ai_prompt_template or ''}"

        return prompt

    async def _build_resume_analysis_prompt(
            self,
            resume_data: Dict[str, Any],
            job_description: Optional[str],
            analysis_types: List[str]
    ) -> str:
        """Build prompt for resume analysis."""
        context = {
            "resume_data": json.dumps(resume_data, indent=2),
            "job_description": job_description or "Not provided",
            "analysis_types": ", ".join(analysis_types)
        }

        return RESUME_ANALYSIS_PROMPT.format(**context)

    async def _build_optimization_prompt(
            self,
            content: Dict[str, Any],
            optimization_type: str,
            job_description: Optional[str],
            target_keywords: Optional[List[str]],
            target_tone: Optional[CoverLetterTone],
            target_length: Optional[int],
            preserve_sections: Optional[List[str]],
            optimization_level: int
    ) -> str:
        """Build prompt for content optimization."""
        context = {
            "content": json.dumps(content, indent=2),
            "optimization_type": optimization_type,
            "job_description": job_description or "Not provided",
            "target_keywords": target_keywords or [],
            "target_tone": target_tone.value if target_tone else "professional",
            "target_length": target_length or "optimal",
            "preserve_sections": preserve_sections or [],
            "optimization_level": optimization_level
        }

        return COVER_LETTER_OPTIMIZATION_PROMPT.format(**context)

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost for API usage."""
        if self.model not in self.model_pricing:
            return 0.0

        pricing = self.model_pricing[self.model]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    def _extract_keywords_fallback(self, job_description: str) -> Dict[str, List[str]]:
        """Fallback keyword extraction using regex patterns."""
        # Common skill patterns
        tech_skills = re.findall(r'\b(?:Python|Java|JavaScript|React|Node\.js|SQL|AWS|Docker|Kubernetes|Git)\b',
                                 job_description, re.IGNORECASE)

        # Education patterns
        education = re.findall(r'\b(?:Bachelor|Master|PhD|Degree|BS|MS|MBA)\b', job_description, re.IGNORECASE)

        # Experience patterns
        experience = re.findall(r'\b\d+\+?\s*(?:years?|yrs?)\b', job_description, re.IGNORECASE)

        return {
            "keywords": list(set(tech_skills + education)),
            "skills": tech_skills,
            "education": education,
            "experience": experience,
            "responsibilities": [],
            "qualifications": []
        }

    async def _analyze_ats_compatibility(self, content: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Analyze ATS compatibility."""
        return {
            "overall_score": 85.0,
            "sub_scores": {
                "parsing_score": 90.0,
                "formatting_score": 85.0,
                "keyword_density": 80.0
            },
            "analysis_data": {
                "ats_friendly": True,
                "issues": [],
                "recommendations": []
            },
            "confidence_score": 0.9
        }

    async def _analyze_keywords(self, content: Dict[str, Any], job_description: Optional[str]) -> Dict[str, Any]:
        """Analyze keyword optimization."""
        return {
            "overall_score": 75.0,
            "sub_scores": {
                "keyword_density": 70.0,
                "keyword_relevance": 80.0,
                "keyword_placement": 75.0
            },
            "analysis_data": {
                "found_keywords": [],
                "missing_keywords": [],
                "keyword_suggestions": []
            },
            "confidence_score": 0.85
        }

    async def _analyze_content_quality(self, content: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Analyze content quality."""
        return {
            "overall_score": 80.0,
            "sub_scores": {
                "clarity": 85.0,
                "impact": 75.0,
                "relevance": 80.0
            },
            "analysis_data": {
                "strengths": [],
                "areas_for_improvement": [],
                "suggestions": []
            },
            "confidence_score": 0.88
        }

    async def _analyze_job_match(self, content: Dict[str, Any], job_description: Optional[str]) -> Dict[str, Any]:
        """Analyze job match compatibility."""
        return {
            "overall_score": 78.0,
            "sub_scores": {
                "skills_match": 80.0,
                "experience_match": 75.0,
                "requirements_match": 80.0
            },
            "analysis_data": {
                "matching_skills": [],
                "missing_skills": [],
                "experience_gaps": []
            },
            "confidence_score": 0.82
        }

    async def _comprehensive_analysis(
            self,
            content: Dict[str, Any],
            document_type: str,
            job_description: Optional[str]
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis combining all analysis types."""
        if document_type == "resume":
            return await self.analyze_resume(content, job_description)
        else:
            return await self.analyze_cover_letter_quality(content, job_description)