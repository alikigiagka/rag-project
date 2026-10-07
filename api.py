"""FastAPI Web Service for Multi-Course RAG Application.

Exposes RESTful API endpoints for listing available courses, processing natural language Q&A queries
with hybrid retrieval, and serving the static single-page web UI.

Execution:
    uvicorn api:app --reload
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import COURSES
from src.retriever import Retriever
from src.llm import generate_response
from main import save_queries_to_json

app = FastAPI(
    title="Multi-Course RAG QA API",
    description="RESTful API for multi-course document QA using Hybrid RAG"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dictionary cache preventing expensive repeated initialization of embedding models and vector stores
_retrievers: dict[str, Retriever] = {}


def get_retriever(course_key: str) -> Retriever:
    """Lazy loader and in-memory cache manager for course-specific Retriever instances.

    Initializes and caches the hybrid retriever instance for a given course key on first request,
    reusing the instance for subsequent API queries to avoid redundant model loading overhead.

    Args:
        course_key (str): Key identifier of the target course.

    Returns:
        Retriever: Cached hybrid retriever instance.

    Raises:
        HTTPException: 404 error if course_key is invalid, or 500 error if retriever instantiation fails.
    """
    if course_key not in COURSES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown course '{course_key}'. Available: {list(COURSES.keys())}",
        )
    if course_key not in _retrievers:
        try:
            _retrievers[course_key] = Retriever(course_key=course_key)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Retriever loading error for '{course_key}': {e}",
            )
    return _retrievers[course_key]


# Pydantic API Data Schemas

class ChatRequest(BaseModel):
    """API request payload schema for chat queries."""
    course_key: str = Field(..., description="Unique string identifier of the selected course")
    query: str = Field(..., description="User question string")
    question_type: str = Field(default="Factual Recall", description="Evaluation taxonomy category")


class SourceItem(BaseModel):
    """Schema for individual document citation metadata."""
    name: str = Field(..., description="Filename of the cited source document")
    pages: list[int] = Field(default_factory=list, description="List of 1-based page numbers cited")


class ChatResponse(BaseModel):
    """API response payload schema for chat queries."""
    answer: str = Field(..., description="Generated natural language response string")
    sources: list[SourceItem] = Field(..., description="List of cited document sources and page numbers")


class CourseItem(BaseModel):
    """Schema for available course metadata."""
    key: str = Field(..., description="Course key identifier")
    name: str = Field(..., description="Human-readable course title in Greek")


# API Endpoints

@app.get("/api/courses", response_model=list[CourseItem], summary="Get list of available courses")
def list_courses() -> list[CourseItem]:
    """Retrieves all available educational courses configured in the system.

    Returns:
        list[CourseItem]: List of course objects containing key identifiers and display names.
    """
    return [CourseItem(key=k, name=v["name"]) for k, v in COURSES.items()]


@app.post("/api/chat", response_model=ChatResponse, summary="Query the RAG system for an answer")
def chat(req: ChatRequest) -> ChatResponse:
    """Processes a user question using hybrid retrieval and LLM generation.

    Retrieves context chunks from ChromaDB and BM25, prompts Gemini for a grounded answer,
    appends QA metrics to the evaluation dataset, and formats cited source page numbers.

    Args:
        req (ChatRequest): Incoming chat request payload containing query, course key, and question type taxonomy.

    Returns:
        ChatResponse: Structured response containing answer text and cited source documents with page numbers.

    Raises:
        HTTPException: 400 if query is empty, 404/500 if course retriever fails, 502 if LLM call fails.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="The question query cannot be empty.")

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
            detail=f"Gemini API communication error: {error}",
        )

    # Fall back to default Factual Recall classification if whitespace or null input is passed
    question_type = req.question_type.strip() if req.question_type and req.question_type.strip() else "Factual Recall"
    save_queries_to_json([{
        "query": query,
        "question_type": question_type,
        "retrieved_context": "\n\n".join(context_texts),
        "generated_answer": answer
    }])

    sources = [
        SourceItem(name=name, pages=sorted(pages))
        for name, pages in sorted(sources_with_pages.items())
    ]

    return ChatResponse(answer=answer, sources=sources)


@app.get("/", summary="Serve main Web Interface")
def serve_index() -> FileResponse:
    """Serves the main single-page web application frontend.

    Returns:
        FileResponse: Static HTML response for index.html.
    """
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


