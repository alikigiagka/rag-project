"""
FastAPI Web Service for Multi-Course RAG Application.
Exposes RESTful endpoints for listing courses, handling chat queries, and serving the frontend UI.

Execution:
    uvicorn api:app --reload
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import COURSES
from src.retriever import Retriever
from src.llm import generate_response

app = FastAPI(title="Multi-Course RAG QA API", description="RESTful API for multi-course document QA using Hybrid RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for course retriever instances
_retrievers: dict[str, Retriever] = {}

def get_retriever(course_key: str) -> Retriever:
    """
    Lazy loader for course-specific Retriever objects.
    Initializes and caches the hybrid retriever (ChromaDB + BM25) for a given course key.
    """
    if course_key not in COURSES:
        raise HTTPException(
            status_code=404,
            detail=f"Άγνωστο μάθημα '{course_key}'. Διαθέσιμα: {list(COURSES.keys())}",
        )
    if course_key not in _retrievers:
        try:
            _retrievers[course_key] = Retriever(course_key=course_key)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Σφάλμα φόρτωσης retriever για '{course_key}': {e}",
            )
    return _retrievers[course_key]

# --- Pydantic API Data Schemas ---

class ChatRequest(BaseModel):
    course_key: str
    query: str

class SourceItem(BaseModel):
    name: str
    pages: list[int] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

class CourseItem(BaseModel):
    key: str
    name: str

# --- API Endpoints ---

@app.get("/api/courses", response_model=list[CourseItem], summary="Get list of available courses")
def list_courses():
    """Returns all available courses configured in the RAG system."""
    return [{"key": k, "name": v["name"]} for k, v in COURSES.items()]

@app.post("/api/chat", response_model=ChatResponse, summary="Query the RAG system for an answer")
def chat(req: ChatRequest):
    """
    Handles RAG QA queries:
    1. Retrieves relevant text chunks using hybrid search for the requested course.
    2. Sends context and query to Gemini LLM for answer generation.
    3. Returns the answer along with cited sources and page numbers.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Η ερώτηση είναι κενή.")

    retriever = get_retriever(req.course_key)
    course_name = COURSES[req.course_key]["name"]

    context_texts, sources_with_pages = retriever.retrieve(query)

    if not context_texts:
        return ChatResponse(
            answer="Δεν βρέθηκαν αρκετά σχετικά έγγραφα για να απαντηθεί η ερώτηση.",
            sources=[],
        )

    answer, error = generate_response(query, context_texts, course_name)

    if error:
        raise HTTPException(
            status_code=502,
            detail=f"Σφάλμα κατά την επικοινωνία με το Gemini API: {error}",
        )

    sources = [
        SourceItem(name=name, pages=sorted(pages))
        for name, pages in sorted(sources_with_pages.items())
    ]

    return ChatResponse(answer=answer, sources=sources)

@app.get("/", summary="Serve main Web Interface")
def serve_index():
    """Serves the static web application index page."""
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

