# IKMS RAG Agent System

A Multi-Agent RAG (Retrieval-Augmented Generation) system built with FastAPI (Backend) and React/Vite (Frontend).

---

## 🚀 Quick Start

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=ikms-rag-agent-system
OPENAI_API_KEY=your_openai_key
OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small
```

### 2. Run Backend (FastAPI)
From the root directory:
```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn src.app.api:app --reload
```
- **API URL**: `http://localhost:8000`
- **Docs**: `http://localhost:8000/docs`

### 3. Run Frontend (React/Vite)
Open a **new terminal** and navigate to the `frontend` folder:
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
- **Web URL**: `http://localhost:5173`

---

## 🛠️ Troubleshooting

### "Incorrect API key provided" (401 Error)
If you see a 401 error even after updating your `.env` file, you likely have "ghost" processes running.
**Fix**: Run this in PowerShell to kill all old processes:
```powershell
Stop-Process -Name "python", "uvicorn" -Force
```
Then restart the server.

### "uv run uvicorn: command not found"
Make sure you have `uv` installed. If not, install it first or use:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn src.app.api:app --reload
```
