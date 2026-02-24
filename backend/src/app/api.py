"""FastAPI entry point for the IKMS RAG system."""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .models import QuestionRequest, QAResponse
from .services.qa_service import answer_question
from .core.retrieval.vector_store import index_documents_from_bytes

app = FastAPI(title="IKMS RAG Agent System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(request: QuestionRequest):
    """Expose the multi-agent QA flow via POST /qa."""
    try:
        result = await answer_question(request.question)
        return QAResponse(
            answer=result.get("answer", "No answer generated."),
            context=result.get("context", "No context retrieved.")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index-pdf")
async def index_pdf_endpoint(file: UploadFile = File(...)):
    """Handle PDF upload and indexing into vector store (in-memory, no disk writes)."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        num_chunks = index_documents_from_bytes(file_bytes, filename=file.filename)
        return {
            "message": f"Successfully indexed {file.filename}",
            "chunks": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
