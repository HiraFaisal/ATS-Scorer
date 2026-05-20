from typing import TypedDict, List, Dict, Any, Optional

class ResumeState(TypedDict):
    resume_bytes:       Optional[bytes]
    resume_text:        Optional[str]
    resume_images:      Optional[List[str]]  # List of base64 encoded strings of pages
    job_description:    Optional[str]
    job_post_id:        Optional[str]
    application_id:     Optional[str]
    mongo_id:           Optional[str]
    candidate_profile:  Optional[Dict[str, Any]]
    analysis_result:    Optional[Dict[str, Any]]
    historical_context: Optional[List[Dict[str, Any]]]
    semantic_similarity: Optional[Dict[str, Any]]
    nlp_analysis:       Optional[Dict[str, Any]]  # spaCy NER and Syntactic results, N-grams
    classification:     Optional[Dict[str, Any]]  # Domain classification
    error:              Optional[str]
    use_fallback:       bool
    model_name:         Optional[str]
    final_output:       Optional[Dict[str, Any]]