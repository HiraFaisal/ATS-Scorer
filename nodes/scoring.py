import json
import re
import hashlib
from openai import OpenAI
from state import ResumeState
from config import OPENAI_API_KEY

def scoring_node(state: ResumeState) -> ResumeState:
    if state.get("error"):
        return state

    resume_text = state.get("resume_text", "")
    jd          = state.get("job_description", "")
    nlp_an      = state.get("nlp_analysis", {})
    classif     = state.get("classification", {})
    sem_sim     = state.get("semantic_similarity", {})

    # Determine API key (check state first, then config)
    api_key = state.get("openai_api_key") or OPENAI_API_KEY
    if not api_key:
        return {**state, "error": "OpenAI API Key is missing. Please provide it in the sidebar or setup a .env file."}

    # Determine model (default to gpt-4o-mini)
    model_name = state.get("model_name") or "gpt-4o-mini"
    # Compile the NLP pre-analysis metrics to feed into the prompt
    nlp_metrics = {
        "spaCy_syntactic_verb_noun_phrases": nlp_an.get("syntactic_phrases", []),
        "spaCy_ner_entities": nlp_an.get("ner_entities", {}),
        "sentence_transformers_semantic_similarity": {
            "overall_document_similarity": sem_sim.get("doc_similarity", 0.0),
            "skill_coverage": sem_sim.get("skill_coverage", 0.0),
            "matched_skill_count": len(sem_sim.get("matched_pairs", [])),
            "unmatched_skill_count": len(sem_sim.get("unmatched_skills", []))
        },
        "n_grams_overlap": {
            "matched_bigrams": nlp_an.get("ngrams", {}).get("bigrams_matched", []),
            "matched_trigrams": nlp_an.get("ngrams", {}).get("trigrams_matched", []),
            "bigram_overlap_ratio": nlp_an.get("ngrams", {}).get("bigram_overlap_pct", 0.0),
            "trigram_overlap_ratio": nlp_an.get("ngrams", {}).get("trigram_overlap_pct", 0.0)
        },
        "domain_classification": {
            "resume_classified_domain": classif.get("resume_domain", "Unknown"),
            "job_classified_domain": classif.get("jd_domain", "Unknown"),
            "domain_alignment_match": classif.get("domain_alignment", False)
        }
    }

    # Build prompt
    system_prompt = """You are a highly precise and objective ATS (Applicant Tracking System) analyst and HR screener.
Your task is to analyze the candidate's Resume (Text and visual image) against the provided Job Description (JD).
You must integrate the provided pre-computed NLP metrics (Syntactic Parsing, NER, N-Grams, Text Classification, Semantic Similarity) to inform your rating.

If a resume image is provided:
Analyze the visual layout, formatting, spacing, margins, readable fonts, and single-page/multi-page structured design. Rate it in the "formatting_readability" metric.

SCORING POLICY (Objectivity is critical):
1. EXPERIENCE LEVEL:
   - Does the candidate meet the year requirements in the JD? Rate accordingly.
   - If the candidate is excessively overqualified for a junior/entry role (e.g. 5+ years over), reduce their overall score slightly to account for retention risk.
2. SKILLS ALIGNMENT:
   - Check which JD skills are explicitly verified in the resume text or supported by semantic synonyms.
3. STATUS ASSIGNMENT:
   - HIRE: Score 80-100. Candidate has near-perfect skills, correct experience level, and clear alignment.
   - LIKELY HIRE: Score 65-79. Candidate has strong skills, minor gaps in secondary requirements.
   - RESERVE: Score 45-64. Candidate has partial skills or is somewhat under/overqualified but has core competence.
   - UNLIKELY: Score 0-44. Major skill mismatches or severe lack of required experience.

You must return a valid JSON object matching the schema below. Keep descriptions concise and impactful."""

    schema_instruction = """
JSON structure to return:
{
  "candidate_profile": {
    "name": "<candidate name or Unknown>",
    "current_role": "<most recent job title or N/A>",
    "total_experience": "<e.g. 5 years>",
    "experience_level": "<Fresher|Junior|Mid-Level|Senior|Lead|Manager>",
    "location": "<city, country or Not found>",
    "email": "<email or Not found>",
    "phone": "<phone or Not found>",
    "summary": "<2-3 sentence overview of candidate's background>",
    "top_skills": ["skill1", "skill2"],
    "education": "<degree, institution, graduation year or N/A>",
    "certifications": ["cert1"],
    "languages": ["lang1"]
  },
  "ats_score": <integer between 0 and 100>,
  "hiring_status": "<HIRE|LIKELY HIRE|RESERVE|UNLIKELY>",
  "score_breakdown": {
    "keyword_match": <integer 0-100, exact and near-exact phrase matches>,
    "experience_relevance": <integer 0-100, years and role match>,
    "skills_alignment": <integer 0-100, semantic skill match & depth>,
    "education_fit": <integer 0-100, degree levels and major field alignment>,
    "formatting_readability": <integer 0-100, check design, layout, visual hierarchy from image>
  },
  "short_description": "<1-2 sentence description explaining candidate's fit relative to the Job Description>",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["missing1", "missing2"],
  "strengths": ["strength1", "strength2"],
  "experience_gaps": ["gap1", "gap2"],
  "standout_elements": ["standout1", "standout2"],
  "verdict": "<detailed final verdict summarizing recommendation>",
  "visual_formatting_critique": "<feedback on margins, font readability, layout spacing, etc.>"
}
"""

    user_text = f"Job Description:\n{jd}\n\nResume Text:\n{resume_text}\n\nPre-calculated NLP Metrics:\n{json.dumps(nlp_metrics, indent=2)}\n\nFormat instructions: Return JSON matching this schema:\n{schema_instruction}"

    try:
        client = OpenAI(api_key=api_key)
        
        # User payload
        user_content = [
            {"type": "text", "text": user_text}
        ]

        # Attach image if available
        if state.get("resume_images") and len(state["resume_images"]) > 0:
            # Send the first page image for visual evaluation (multi-modal)
            first_page_b64 = state["resume_images"][0]
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{first_page_b64}"
                }
            })
            print("Attached first page image to OpenAI multi-modal request.")

        # Execute API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=2500
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        # Update candidate_profile fields if they were missing or generic
        profile = result.get("candidate_profile", {})
        
        return {
            **state,
            "analysis_result": result,
            "candidate_profile": profile,
            "model_name": model_name,
            "use_fallback": False
        }

    except Exception as e:
        print(f"OpenAI scoring failed: {e}")
        return {**state, "error": f"OpenAI Scoring Node failed: {str(e)}"}