import re
from typing import List
from state import ResumeState

def semantic_match_node(state: ResumeState) -> ResumeState:
    if state.get("error") or not state.get("resume_text") or not state.get("job_description"):
        return state
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        resume_text = state["resume_text"]
        jd = state["job_description"]

        resume_emb = model.encode(resume_text[:2000], convert_to_tensor=True)
        jd_emb     = model.encode(jd[:2000],          convert_to_tensor=True)
        doc_score  = float(util.cos_sim(resume_emb, jd_emb)[0][0])

        def extract_skill_candidates(text: str) -> List[str]:
            raw = re.split(r"[,\n•\-/|()]+", text)
            skills = []
            for chunk in raw:
                chunk = chunk.strip()
                words = chunk.split()
                if 1 <= len(words) <= 4 and len(chunk) > 2:
                    skills.append(chunk.lower())
            return list(dict.fromkeys(skills))

        jd_skills     = extract_skill_candidates(jd)[:60]
        resume_skills = extract_skill_candidates(resume_text)[:80]
        matched_pairs, unmatched_jd, skill_coverage = [], jd_skills, 0.0

        if jd_skills and resume_skills:
            jd_embs     = model.encode(jd_skills,     convert_to_tensor=True)
            resume_embs = model.encode(resume_skills, convert_to_tensor=True)
            cos_scores  = util.cos_sim(jd_embs, resume_embs)
            matched_pairs, unmatched_jd = [], []
            for i, jd_skill in enumerate(jd_skills):
                best_idx   = int(cos_scores[i].argmax())
                best_score = float(cos_scores[i][best_idx])
                if best_score >= 0.80:
                    matched_pairs.append({"jd_skill": jd_skill, "resume_match": resume_skills[best_idx], "similarity": round(best_score, 3)})
                else:
                    unmatched_jd.append(jd_skill)
            skill_coverage = len(matched_pairs) / len(jd_skills)

        return {**state, "semantic_similarity": {
            "doc_similarity":   round(doc_score, 3),
            "skill_coverage":   round(skill_coverage, 3),
            "matched_pairs":    matched_pairs,
            "unmatched_skills": unmatched_jd[:20],
        }}
    except Exception as e:
        print(f"Semantic match failed: {e}")
        return state