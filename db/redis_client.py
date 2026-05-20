import functools
import redis
from config import REDIS_HOST, REDIS_PORT

@functools.lru_cache(maxsize=1)
def get_redis_client():
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return None