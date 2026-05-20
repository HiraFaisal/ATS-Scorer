import json

def build_ollama_prompt(system_prompt: str, jd: str, resume_text: str) -> str:
    user_message = f"JOB DESCRIPTION:\n{jd}\n\nCANDIDATE RESUME:\n{resume_text}\n\nAnalyze and return the JSON evaluation."
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def build_ollama_system_prompt(historical: list, semantic_context_block: str) -> str:
    historical_block = ""
    if historical and any(isinstance(h, dict) and h.get("ats_score") for h in historical):
        historical_block = f"""HISTORICAL CONTEXT (for scoring consistency only — do NOT copy these scores):
{json.dumps([h for h in historical if isinstance(h, dict) and h.get("ats_score")], indent=2)}"""

    return f"""You are a precise and fair ATS analyst and senior HR consultant.
Your job is to objectively evaluate how well a candidate ACTUALLY matches a job description.

CORE SCORING PHILOSOPHY:
- Score based on ACTUAL match, not assumptions.
- If the JD says "1+ years experience" and candidate has 1+ years, that is a FULL match.
- If candidate experience significantly exceeds JD requirement AND role is junior-level, reduce ATS score.
- Over-experience = possible mismatch (salary expectations, retention risk).

IMPORTANT RULE:
If candidate experience significantly exceeds JD requirement (2+ levels higher), reduce ATS score.
Senior-level candidates applying to junior roles should NOT be scored as strong matches.
When reducing due to overqualification, verdict MUST start with "Overqualified:"

SCORING RULES:
1. EXPERIENCE YEARS:
   - Meets or exceeds JD requirement → experience_relevance = 70-100
   - 1-2 years short → 40-60
   - 3-5 years short → 20-40
   - 5+ years short → 5-20

2. SKILLS ALIGNMENT — credit only skills clearly demonstrated via experience or projects.
3. KEYWORD MATCH — only count if actually used in a job or project.

WEIGHTING HIERARCHY:
- PRIMARY (90% weight): Experience Relevance, Skills Alignment, and Keyword Match.
- SECONDARY (10% weight): Education Fit and Formatting Readability. 
The overall ats_score must be driven by the PRIMARY factors. A candidate with a perfect degree but wrong experience should receive a low score.

4. ATS SCORE ranges:
   - 75-100: Strong match
   - 50-74: Good match with minor gaps
   - 30-49: Moderate match, significant gaps
   - 0-29: Poor match

5. hiring_likelihood:
   - 0-30 → "Unlikely"
   - 31-50 → "Possible"
   - 51-70 → "Likely"
   - 71-100 → "Highly Likely"

{historical_block}
{semantic_context_block}

CRITICAL NUMERIC RULE:
You MUST fill every integer field with a real number between 1-100.
NEVER output 0 for ats_score or breakdown fields. Even a poor match gets at least 5.

Return ONLY a valid JSON object (no markdown, no backticks):
{{
  "candidate_profile": {{
    "name": "<full name>",
    "current_role": "<most recent job title>",
    "total_experience": "<e.g. 3 years>",
    "experience_level": "<Fresher|Junior|Mid-Level|Senior|Lead|Manager>",
    "location": "<city, country or Not found>",
    "email": "<email or Not found>",
    "phone": "<phone or Not found>",
    "summary": "<2-3 sentence summary>",
    "top_skills": [],
    "expertise_areas": [],
    "education": "<degree, institution, year>",
    "certifications": [],
    "languages": [],
    "total_jobs": 0,
    "notable_projects": []
  }},
  "ats_score": 0,
  "score_breakdown": {{
    "keyword_match": 0,
    "experience_relevance": 0,
    "skills_alignment": 0,
    "education_fit": 0,
    "formatting_readability": 0
  }},
  "matched_keywords": [],
  "missing_skills": [],
  "strengths": [],
  "experience_gaps": [],
  "standout_elements": [],
  "verdict": "<honest specific assessment>",
  "hiring_likelihood": "<Unlikely|Possible|Likely|Highly Likely>"
}}"""