from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT      = int(os.getenv("REDIS_PORT", 6379))
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
TESSERACT_PATH  = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")