"""Central Configuration Module for Multi-Course RAG System.

Defines course metadata, persistent database storage locations, model designations,
and retrieval hyperparameters for dense vector search and sparse BM25 indexing.
"""

import os
from dotenv import load_dotenv

# Course definitions and file paths mapping
COURSES = {
    "databases": {
        "name": "Βάσεις Δεδομένων",
        "path": "data/databases",
        "collection": "db_course",
        "bm25_index": "bm25_databases.pkl"
    },
    "parallel-distributed": {
        "name": "Παράλληλος και Κατανεμημένος Υπολογισμός",
        "path": "data/parallel-distributed",
        "collection": "parallel_course",
        "bm25_index": "bm25_parallel.pkl"
    }
}

# Base directory paths
DATA_BASE_DIR = "data"
CHROMA_PATH = "chroma_db"

# Multilingual embedding model tailored for dense semantic retrieval across English and Greek
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
GEMINI_VISION_MODEL_NAME = 'gemini-2.5-flash'
PROCESS_IMAGES = True

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Standard Reciprocal Rank Fusion constant k=60 to dampen impact of high ranks across retrieval algorithms
RRF_K = 60

# Candidate pool sizes over-fetched from individual retrievers prior to fusion
TOP_K_DENSE = 8
TOP_K_SPARSE = 8

# Maximum context items retained post-fusion to fit within LLM context window without diluting attention
FINAL_TOP_K = 5

# Minimum token count threshold to discard isolated slide titles, page numbers, and structural noise
MIN_CHUNK_WORDS = 15

os.makedirs(DATA_BASE_DIR, exist_ok=True)
for course_key, course_info in COURSES.items():
    os.makedirs(course_info["path"], exist_ok=True)


