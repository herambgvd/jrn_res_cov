"""
AI prompt templates for various analysis and generation tasks.
"""

# Cover Letter Generation Prompt
COVER_LETTER_GENERATION_PROMPT = """
Generate a personalized cover letter based on the following information.

Job Details:
- Job Title: {job_title}
- Company: {company_name}
- Hiring Manager: {hiring_manager_name}

Job Description:
{job_description}

User Context:
- Tone: {tone}
- Personalization Level: {personalization_level}/5
- Key Points to Include: {key_points}
- Company Research: {company_research}

Resume Data (if available):
{resume_data}

Requirements:
1. Create a compelling cover letter that matches the specified tone
2. Personalize based on the job description and company information
3. Highlight relevant experience from the resume data (if provided)
4. Include specific examples and achievements
5. Maintain professional formatting and structure
6. Optimize for ATS compatibility
7. Keep length between 250-400 words

Return the response as JSON with the following structure:
{{
  "opening_paragraph": "Engaging opening that hooks the reader",
  "body_paragraphs": ["First body paragraph", "Second body paragraph (if needed)"],
  "closing_paragraph": "Strong closing with call to action",
  "keywords_used": ["keyword1", "keyword2"],
  "personalization_score": 85.0,
  "tone_consistency": 90.0,
  "word_count": 350,
  "key_achievements_highlighted": ["achievement1", "achievement2"]
}}
"""

# Resume Analysis Prompt
RESUME_ANALYSIS_PROMPT = """
Analyze the following resume for quality, ATS compatibility, and job matching.

Resume Data:
{resume_data}

Job Description (for matching):
{job_description}

Analysis Types Requested: {analysis_types}

Perform comprehensive analysis including:
1. ATS Compatibility (0-100)
   - Parsing friendliness
   - Format compatibility
   - Keyword optimization
   - Section organization

2. Content Quality (0-100)
   - Impact and achievements
   - Quantified results
   - Relevance to target role
   - Professional presentation

3. Format Score (0-100)
   - Visual appeal
   - Readability
   - Consistency
   - Length appropriateness

4. Keyword Analysis
   - Missing important keywords
   - Keyword density
   - Industry-specific terms
   - Action verbs usage

5. Job Matching (if job description provided)
   - Skills alignment
   - Experience relevance
   - Requirements fulfillment

Return detailed analysis as JSON:
{{
  "overall_score": 85.0,
  "ats_score": 88.0,
  "content_score": 82.0,
  "format_score": 90.0,
  "keyword_score": 75.0,
  "job_match_score": 80.0,
  "strengths": ["Clear achievements", "Quantified results"],
  "weaknesses": ["Missing keywords", "Weak action verbs"],
  "missing_keywords": ["keyword1", "keyword2"],
  "suggestions": [
    {{
      "type": "keyword_optimization",
      "priority": "high",
      "title": "Add missing keywords",
      "description": "Include specific technical skills mentioned in job description",
      "before_text": "Managed projects",
      "suggested_text": "Led cross-functional projects using Agile methodology",
      "impact_score": 0.8,
      "section": "work_experience"
    }}
  ],
  "ats_compatibility": {{
    "parsing_score": 95.0,
    "format_issues": [],
    "recommendations": ["Use standard section headers"]
  }},
  "confidence_score": 0.92
}}
"""

# Cover Letter Optimization Prompt
COVER_LETTER_OPTIMIZATION_PROMPT = """
Optimize the following cover letter content based on the specified criteria.

Current Content:
{content}

Optimization Type: {optimization_type}
Job Description: {job_description}
Target Keywords: {target_keywords}
Target Tone: {target_tone}
Target Length: {target_length} words
Preserve Sections: {preserve_sections}
Optimization Level: {optimization_level}/5

Optimization Instructions:
1. If optimization_type is "keyword": Focus on incorporating target keywords naturally
2. If optimization_type is "tone": Adjust language to match target tone
3. If optimization_type is "length": Adjust content to meet target word count
4. If optimization_type is "ats": Optimize for ATS parsing and compatibility

Maintain the core message while improving:
- Keyword density and relevance
- Professional tone consistency
- Impact and persuasiveness
- ATS compatibility
- Overall effectiveness

Return optimized content as JSON:
{{
  "optimized_content": {{
    "opening_paragraph": "Optimized opening",
    "body_paragraphs": ["Optimized body paragraphs"],
    "closing_paragraph": "Optimized closing"
  }},
  "changes_made": [
    {{
      "type": "keyword_addition",
      "section": "opening_paragraph",
      "original": "I am writing to apply",
      "optimized": "I am writing to apply for the Senior Developer position",
      "reason": "Added specific job title keyword"
    }}
  ],
  "improvement_score": 15.0,
  "optimization_summary": "Enhanced keyword density and improved professional tone",
  "keywords_added": ["Senior Developer", "Python", "AWS"],
  "tone_adjustments": ["More confident language", "Action-oriented verbs"],
  "word_count": 320,
  "readability_improvement": 8.5
}}
"""

# Keyword Extraction Prompt
KEYWORD_EXTRACTION_PROMPT = """
Extract and categorize important keywords from the following job description.

Job Description:
{job_description}

Extract keywords in the following categories:
1. Technical Skills - Programming languages, tools, technologies
2. Soft Skills - Communication, leadership, problem-solving
3. Industry Terms - Domain-specific terminology
4. Experience Requirements - Years of experience, level requirements
5. Education Requirements - Degrees, certifications
6. Responsibilities - Key job functions and duties
7. Qualifications - Must-have vs nice-to-have requirements

Return results as JSON:
{{
  "keywords": ["all_important_keywords"],
  "technical_skills": ["Python", "AWS", "Docker"],
  "soft_skills": ["leadership", "communication"],
  "industry_terms": ["fintech", "compliance"],
  "experience_requirements": ["5+ years", "senior level"],
  "education_requirements": ["Bachelor's degree", "Computer Science"],
  "responsibilities": ["lead development", "mentor team"],
  "qualifications": ["required: Python", "preferred: AWS"],
  "priority_keywords": ["most_important_keywords"],
  "keyword_frequency": {{"keyword": count}}
}}
"""

# ATS Analysis Prompt
ATS_ANALYSIS_PROMPT = """
Analyze the following document for ATS (Applicant Tracking System) compatibility.

Document Content:
{content}

Document Type: {document_type}

Evaluate ATS compatibility across these dimensions:
1. Format Compatibility
   - File format suitability
   - Text extraction quality
   - Layout parsing ability

2. Content Structure
   - Standard section headers
   - Logical information flow
   - Contact information placement

3. Keyword Optimization
   - Keyword density
   - Keyword placement
   - Industry-relevant terms

4. Technical Compliance
   - Font compatibility
   - Character encoding
   - Image/graphic usage

Return analysis as JSON:
{{
  "overall_ats_score": 88.0,
  "parsing_score": 95.0,
  "formatting_score": 85.0,
  "keyword_score": 80.0,
  "compatibility_issues": [
    {{
      "issue": "Non-standard section header",
      "severity": "medium",
      "location": "Skills section",
      "recommendation": "Use 'Technical Skills' instead of 'My Expertise'"
    }}
  ],
  "optimization_suggestions": [
    "Add more industry keywords",
    "Use standard section headers",
    "Improve keyword density"
  ],
  "ats_friendly_score": 92.0,
  "parsing_confidence": 0.95
}}
"""

# Content Optimization Prompt
CONTENT_OPTIMIZATION_PROMPT = """
Optimize the content for maximum impact and professional presentation.

Content to Optimize:
{content}

Content Type: {content_type}
Target Audience: {target_audience}
Optimization Goals: {optimization_goals}

Focus on improving:
1. Clarity and Readability
2. Impact and Persuasiveness
3. Professional Tone
4. Quantified Achievements
5. Action-Oriented Language

Return optimized content with explanations:
{{
  "optimized_content": "Improved content here",
  "improvements_made": [
    {{
      "type": "clarity",
      "original": "Did various tasks",
      "optimized": "Led cross-functional team of 8 developers",
      "impact": "Added specificity and quantification"
    }}
  ],
  "impact_score_improvement": 25.0,
  "readability_improvement": 15.0,
  "professional_tone_score": 95.0
}}
"""

# Job Matching Prompt
JOB_MATCHING_PROMPT = """
Analyze how well the candidate matches the job requirements.

Candidate Profile:
{candidate_profile}

Job Requirements:
{job_requirements}

Analyze matching across:
1. Required Skills vs Candidate Skills
2. Experience Level Alignment
3. Education Requirements Match
4. Industry Experience Relevance
5. Role Responsibilities Fit

Provide detailed matching analysis:
{{
  "overall_match_score": 85.0,
  "skills_match": {{
    "score": 88.0,
    "matching_skills": ["Python", "AWS"],
    "missing_skills": ["Kubernetes"],
    "transferable_skills": ["Docker to Kubernetes"]
  }},
  "experience_match": {{
    "score": 80.0,
    "required_years": 5,
    "candidate_years": 4,
    "relevant_experience": ["Similar role for 3 years"]
  }},
  "education_match": {{
    "score": 100.0,
    "meets_requirements": true,
    "details": "BS Computer Science matches requirement"
  }},
  "gap_analysis": [
    {{
      "gap": "Missing Kubernetes experience",
      "severity": "medium",
      "mitigation": "Docker experience shows containerization knowledge"
    }}
  ],
  "recommendation": "Strong candidate with minor skill gaps",
  "confidence": 0.87
}}
"""

# Skill Extraction Prompt
SKILL_EXTRACTION_PROMPT = """
Extract and categorize skills from the provided content.

Content:
{content}

Extract skills in categories:
1. Technical Skills
2. Programming Languages
3. Frameworks/Libraries
4. Tools/Software
5. Cloud Platforms
6. Databases
7. Soft Skills
8. Industry Knowledge

Return categorized skills:
{{
  "technical_skills": ["REST APIs", "Microservices"],
  "programming_languages": ["Python", "JavaScript"],
  "frameworks": ["React", "Django"],
  "tools": ["Git", "Docker"],
  "cloud_platforms": ["AWS", "Azure"],
  "databases": ["PostgreSQL", "MongoDB"],
  "soft_skills": ["Leadership", "Communication"],
  "industry_knowledge": ["Financial Services", "Healthcare"],
  "all_skills": ["comprehensive_list"],
  "skill_proficiency": {{"skill": "level"}},
  "years_experience": {{"skill": years}}
}}
"""

# Prompt for real-time content suggestions
REAL_TIME_SUGGESTIONS_PROMPT = """
Provide immediate suggestions for improving this content snippet.

Content: {content}
Content Type: {content_type}
Context: {context}

Provide quick, actionable suggestions:
{{
  "suggestions": [
    {{
      "type": "improvement",
      "message": "Use more specific metrics",
      "example": "Increased sales by 25% instead of 'increased sales'"
    }}
  ],
  "tone_feedback": "Professional and confident",
  "grammar_issues": [],
  "word_choice_improvements": [],
  "structure_suggestions": []
}}
"""