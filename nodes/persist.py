import json
import time
from datetime import datetime
from state import ResumeState
from db import get_chroma_client, get_embedding_function

def persistence_node(state: ResumeState) -> ResumeState:
    if state.get("error") or not state.get("analysis_result"):
        return state

    chroma = get_chroma_client()
    res    = state["analysis_result"]
    profile = res.get("candidate_profile", {})
    
    # Keeping the record dictionary creation just in case it is needed for other logging or future extensions
    record = {
        "timestamp":        datetime.now(),
        "job_post_id":      state.get("job_post_id"),
        "application_id":   state.get("application_id"),
        "name":             profile.get("name", "Unknown"),
        "email":            profile.get("email", "Not found"),
        "phone":            profile.get("phone", "Not found"),
        "location":         profile.get("location", "Not found"),
        "experience_years": profile.get("total_experience", "N/A"),
        "experience_level": profile.get("experience_level", "N/A"),
        "education": [
            {"degree": profile.get("education", ""), "institution": "", "year": None}
        ] if profile.get("education") else [],
        "skills":           profile.get("top_skills", []),
        "certificates":     profile.get("certifications", []),
        "languages":        profile.get("languages", []),
        "expertise_areas":  profile.get("expertise_areas", []),
        "work_experience": [
            {"title": profile.get("current_role", ""), "company": "", "duration": profile.get("total_experience", "")}
        ],
        "notable_projects": profile.get("notable_projects", []),
        "ats_score":        res.get("ats_score", 0),
        "hiring_status":     res.get("hiring_status", "Unknown"),
        "hiring_likelihood": res.get("hiring_status", "Unknown"),  # For backward compatibility
        "overview":         res.get("verdict", ""),
        "ats_analysis": {
            "score":            res.get("ats_score", 0),
            "hiring_status":    res.get("hiring_status", "Unknown"),
            "score_breakdown":  res.get("score_breakdown", {}),
            "summary":          res.get("verdict", ""),
            "key_strengths":    res.get("strengths", []),
            "standout_elements": res.get("standout_elements", []),
            "gaps":             res.get("experience_gaps", []),
            "matched_keywords": res.get("matched_keywords", []),
            "missing_skills":   res.get("missing_skills", []),
        },
        "semantic_similarity": state.get("semantic_similarity", {}),
        "nlp_analysis":     state.get("nlp_analysis", {}),
        "classification":   state.get("classification", {}),
        "model_used":       state.get("model_name", "unknown"),
        "fallback_triggered": state.get("use_fallback", False),
    }

    if chroma is not None:
        try:
            collection = chroma.get_or_create_collection(
                name="historical_analyses",
                embedding_function=get_embedding_function(),
                metadata={"hnsw:space": "cosine"}
            )
            collection.add(
                documents=[state["resume_text"]],
                metadatas=[{"data": json.dumps(res)}],
                ids=[f"res_{int(time.time())}"]
            )
        except Exception as e:
            print(f"ChromaDB index failed: {e}")

    return state