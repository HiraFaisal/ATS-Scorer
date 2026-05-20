import functools
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import CHROMA_PATH

@functools.lru_cache(maxsize=1)
def get_chroma_client():
    try:
        return chromadb.PersistentClient(path=CHROMA_PATH)
    except Exception as e:
        print(f"ChromaDB initialization failed: {e}")
        return None

@functools.lru_cache(maxsize=1)
def get_embedding_function():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")