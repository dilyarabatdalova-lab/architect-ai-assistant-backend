import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from services.document_service import DocumentService
from services.openai_service import OpenAIService
from services.vector_service import VectorService


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
NO_INFO_ANSWER = "Информация по данному вопросу в базе знаний отсутствует."

# AI_PROVIDER controls how the final answer is generated:
# none - free demo mode, answer is built from prepared documents only;
# openai - OpenAI API is used after searching the prepared documents;
# deepseek - DeepSeek API is used after searching the prepared documents.
AI_PROVIDER = os.getenv("AI_PROVIDER", "").lower().strip()

# Backward compatibility with the previous USE_OPENAI setting.
if not AI_PROVIDER:
    AI_PROVIDER = "openai" if os.getenv("USE_OPENAI", "false").lower() == "true" else "none"

if AI_PROVIDER not in {"none", "openai", "deepseek"}:
    AI_PROVIDER = "none"

USE_VECTOR_EMBEDDINGS = AI_PROVIDER == "openai"

app = FastAPI(title="Architect AI Assistant")

document_service = DocumentService(DOCUMENTS_DIR)
vector_service = VectorService(VECTOR_DB_DIR, USE_VECTOR_EMBEDDINGS)
ai_service = OpenAIService(AI_PROVIDER) if AI_PROVIDER in {"openai", "deepseek"} else None


class ChatRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    document: str
    fragment: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


@app.on_event("startup")
def startup() -> None:
    """Prepare the document base when the server starts."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    if USE_VECTOR_EMBEDDINGS and vector_service.has_index():
        return

    chunks = document_service.load_documents()

    if USE_VECTOR_EMBEDDINGS and chunks:
        vector_service.add_chunks(chunks)
    else:
        vector_service.set_fallback_chunks(chunks)


@app.get("/")
def health_check() -> dict:
    """Simple check that the published server is running."""
    return {
        "status": "ok",
        "message": "Architect AI Assistant backend is running",
        "ai_provider": AI_PROVIDER,
        "vector_embeddings": USE_VECTOR_EMBEDDINGS,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a Unity chat question using only prepared local documents."""
    question = request.question.strip()

    if not question:
        return ChatResponse(answer=NO_INFO_ANSWER, sources=[])

    chunks = vector_service.search(question, limit=5)

    if not chunks:
        return ChatResponse(answer=NO_INFO_ANSWER, sources=[])

    if ai_service is not None and ai_service.is_ready():
        answer = ai_service.answer(question, chunks)
    else:
        answer = vector_service.build_fallback_answer(question, chunks)

    if not answer.strip():
        answer = NO_INFO_ANSWER

    sources = [
        SourceInfo(document=chunk["source"], fragment=chunk["text"][:350])
        for chunk in chunks
    ]

    return ChatResponse(answer=answer, sources=sources)
