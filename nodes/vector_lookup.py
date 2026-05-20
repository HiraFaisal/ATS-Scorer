import json
from state import ResumeState
from db import get_chroma_client, get_embedding_function

def vector_lookup_node(state: ResumeState) -> ResumeState:
    chroma = get_chroma_client()
    if not chroma or not state.get("resume_text"):
        return state
    try:
        collection = chroma.get_or_create_collection(
            name="historical_analyses",
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"}
        )
        if collection.count() == 0:
            return state
        results = collection.query(
            query_texts=[state["resume_text"][:1500]],
            n_results=min(3, collection.count()),
            include=["metadatas", "distances"]
        )
        historical = []
        if results and results["metadatas"]:
            for meta_list, dist_list in zip(results["metadatas"], results["distances"]):
                for meta, dist in zip(meta_list, dist_list):
                    if dist < 0.45:
                        try:
                            historical.append(json.loads(meta["data"]))
                        except Exception:
                            pass
        return {**state, "historical_context": historical}
    except Exception as e:
        print(f"Vector lookup failed: {e}")
        return state