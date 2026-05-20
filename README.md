# ATS Scorer API

## 🚀 Features

- **Multi-Layered Scoring**:
  - **Tier 1 (Groq)**: Llama-3.3-70b for high-speed, high-intelligence analysis.
  - **Tier 2 (Ollama)**: Llama-3.1-8b for local, private fallback analysis.
  - **Tier 3 (Cosine)**: Deterministic mathematical fallback using semantic embeddings.
- **Intelligent Parsing**: Supports standard PDFs and scanned documents via Tesseract OCR fallback.
- **Semantic Matching**: Uses `sentence-transformers` for deep skill alignment (0.80 threshold).
- **Caching & Persistence**: Redis for fast repeat queries and MongoDB for long-term storage.
- **Historical Context**: ChromaDB vector store to provide context from previous analyses.

---

## 📋 Prerequisites

Ensure the following are installed and running on your system:

1.  **Python 3.10+**
2.  **MongoDB**: Default `mongodb://localhost:27017/`
3.  **Redis**: Default `localhost:6379`
4.  **Ollama**:
    - Download from [ollama.com](https://ollama.com)
    - Pull the required model: `ollama pull llama3.1:8b`
5.  **Tesseract OCR**:
    - [Download for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
    - Default path expected: `C:\Program Files\Tesseract-OCR\tesseract.exe`

---

## 🛠️ Installation

1.  **Clone the repository** (or navigate to the project folder).
2.  **Install dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Configure Environment**:
    Create a `.env` file in the root directory (see `.env.example` if available):
    ```env
    MONGO_URI=mongodb://localhost:27017/
    REDIS_HOST=localhost
    REDIS_PORT=6379
    CHROMA_PATH=./chroma_db
    OLLAMA_URL=http://localhost:11434/api/generate
    GROQ_API_KEY=your_groq_api_key_here
    TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
    ```

---

## 🚦 How to Run

1.  **Start the API**:
    ```powershell
    uvicorn main:app --reload
    ```
2.  **Access Documentation**:
    Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to use the Interactive Swagger UI.

---

## 🧪 Testing Fallbacks

The system is designed to never fail a request. You can test the tiers by:

1.  **Groq**: Enabled by default with a valid API key.
2.  **Ollama Fallback**: Rename/disable your `GROQ_API_KEY` in `.env`. The system will automatically switch to your local Llama 3.1 model.
3.  **Cosine Fallback**: Stop the Ollama service or provide an invalid model name. The system will use mathematical semantic similarity to generate a score.


## 🏗️ Architecture

The project uses **LangGraph** to manage the stateful workflow:
`Parse` -> `Vector Lookup` -> `Semantic Match` -> `Scoring (LLM/Math)` -> `Persistence`
