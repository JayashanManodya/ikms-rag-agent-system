"""FastAPI entry point for the IKMS RAG system."""

from fastapi import FastAPI, HTTPException
from .models import QuestionRequest, QAResponse
from .services.qa_service import answer_question

app = FastAPI(title="IKMS RAG Agent System")

@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(request: QuestionRequest):
    """Expose the multi-agent QA flow via POST /qa."""
    try:
        result = answer_question(request.question)
        return QAResponse(
            answer=result.get("answer", "No answer generated."),
            context=result.get("context", "No context retrieved.")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
