from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from bson import ObjectId

from workflow import create_workflow
from db.mongo import ensure_mongo_indexes, get_mongodb_client
from schemas.response import ScoreResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_mongo_indexes()
    yield

app = FastAPI(title="ATS Scorer API", version="2.0", lifespan=lifespan)


@app.post("/score", response_model=ScoreResponse)
async def score_resume(
    resume:          UploadFile = File(...),
    job_description: str        = Form(...),
    job_post_id:     str        = Form(None),
    application_id:  str        = Form(None),
):
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF resumes accepted.")
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is required.")

    initial_state = {
        "resume_bytes":       await resume.read(),
        "job_description":    job_description,
        "job_post_id":        job_post_id,
        "application_id":     application_id,
        "use_fallback":       False,
        "historical_context": [],
        "semantic_similarity": None,
        "error":              None,
        "resume_text":        None,
        "candidate_profile":  None,
        "analysis_result":    None,
        "mongo_id":           None,
        "model_name":         None,
        "final_output":       None,
    }

    result_state = create_workflow().invoke(initial_state)

    if result_state.get("error"):
        raise HTTPException(status_code=500, detail=result_state["error"])

    res     = result_state["analysis_result"]
    profile = res.get("candidate_profile", {})

    return ScoreResponse(
        status            = "success",
        mongo_id          = result_state.get("mongo_id"),
        job_post_id       = job_post_id,
        application_id    = application_id,
        name              = profile.get("name"),
        ats_score         = res.get("ats_score"),
        hiring_likelihood = res.get("hiring_likelihood"),
        overview          = res.get("verdict"),
        experience_level  = profile.get("experience_level"),
        matched_keywords  = res.get("matched_keywords", []),
        missing_skills    = res.get("missing_skills", []),
        strengths         = res.get("strengths", []),
        gaps              = res.get("experience_gaps", []),
        score_breakdown   = res.get("score_breakdown", {}),
        semantic_similarity = {
            "doc_similarity": result_state.get("semantic_similarity", {}).get("doc_similarity"),
            "skill_coverage": result_state.get("semantic_similarity", {}).get("skill_coverage"),
        },
        fallback_used     = result_state.get("use_fallback", False),
    )


@app.get("/candidate/{mongo_id}")
async def get_candidate(mongo_id: str):
    db = get_mongodb_client()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    try:
        doc = db.candidates.find_one({"_id": ObjectId(mongo_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Candidate not found.")
        doc["_id"]       = str(doc["_id"])
        doc["timestamp"] = str(doc["timestamp"])
        return JSONResponse(doc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/job/{job_post_id}/candidates")
async def get_candidates_by_job(job_post_id: str, min_score: int = 0, limit: int = 20):
    db = get_mongodb_client()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    cursor = db.candidates.find(
        {"job_post_id": job_post_id, "ats_score": {"$gte": min_score}},
        {"resume_text": 0}
    ).sort("ats_score", -1).limit(limit)
    results = []
    for doc in cursor:
        doc["_id"]       = str(doc["_id"])
        doc["timestamp"] = str(doc["timestamp"])
        results.append(doc)
    return JSONResponse({"job_post_id": job_post_id, "total": len(results), "candidates": results})