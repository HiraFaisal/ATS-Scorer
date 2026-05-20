import functools
from pymongo import MongoClient
from config import MONGO_URI

@functools.lru_cache(maxsize=1)
def get_mongodb_client():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        return client["resume_scorer"]
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None

def ensure_mongo_indexes():
    db = get_mongodb_client()
    if db is None:
        return
    try:
        db.candidates.create_index("job_post_id")
        db.candidates.create_index("ats_score")
        db.candidates.create_index("application_id", unique=True, sparse=True)
        db.candidates.create_index([("skills", 1)])
        db.candidates.create_index([("ats_score", -1)])
        print("MongoDB indexes ensured.")
    except Exception as e:
        print(f"Index creation failed: {e}")