import json

def build_groq_system_prompt(historical: list, semantic_context_block: str) -> str:
    historical_block = ""
    if historical and any(isinstance(h, dict) and h.get("ats_score") for h in historical):
        historical_block = f"""HISTORICAL CONTEXT (for scoring consistency only — do NOT copy these scores):
{json.dumps([h for h in historical if isinstance(h, dict) and h.get("ats_score")], indent=2)}"""

    return f"""You are a precise and fair ATS analyst and senior HR consultant.
Your job is to objectively evaluate how well a candidate ACTUALLY matches a job description.

CORE SCORING PHILOSOPHY:
- Score based on ACTUAL match, not assumptions
- If the JD says "1+ years experience" and candidate has 1+ years, that is a FULL match
- If candidate experience significantly exceeds JD requirement AND role is junior-level, reduce ATS score
- Over-experience = possible mismatch (salary expectations, retention risk)

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

Return ONLY a valid JSON object with EXACTLY this structure (no markdown, no backticks):
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


def build_semantic_context_block(sem: dict) -> str:
    if not sem:
        return ""
    matched_display = ", ".join(
        f"{p['jd_skill']} ≈ {p['resume_match']} ({int(p['similarity']*100)}%)"
        for p in sem.get("matched_pairs", [])[:15]
    )
    unmatched_display = ", ".join(sem.get("unmatched_skills", [])[:15])
    return f"""
SEMANTIC MATCH PRE-ANALYSIS:
- Overall resume↔JD similarity: {int(sem['doc_similarity']*100)}%
- JD skill coverage: {int(sem['skill_coverage']*100)}%
- Matched skill pairs: {matched_display or 'none'}
- Unmatched JD skills: {unmatched_display or 'none'}

A semantic match ≥ 70% means the candidate demonstrably has that skill.
Unmatched JD skills should appear in missing_skills unless a clear synonym exists.
"""