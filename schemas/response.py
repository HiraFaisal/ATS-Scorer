from pydantic import BaseModel
from typing import Optional, Dict, Any

class ScoreResponse(BaseModel):
    status:             str
    mongo_id:           Optional[str]
    job_post_id:        Optional[str]
    application_id:     Optional[str]
    name:               Optional[str]
    ats_score:          Optional[int]
    hiring_likelihood:  Optional[str]
    overview:           Optional[str]
    experience_level:   Optional[str]
    matched_keywords:   list
    missing_skills:     list
    strengths:          list
    gaps:               list
    score_breakdown:    Dict[str, Any]
    semantic_similarity: Dict[str, Any]
    fallback_used:      bool