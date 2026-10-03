"""
Central Configuration Module for Multi-Course RAG System.
Defines course paths, vector DB settings, model names, and retrieval hyperparameters.
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

# Base directories
DATA_BASE_DIR = "data"
CHROMA_PATH = "chroma_db"

# Embedding and Generative AI Model Configurations
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
GEMINI_VISION_MODEL_NAME = 'gemini-2.5-flash'
PROCESS_IMAGES = True

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Retrieval & Hybrid Search Hyperparameters
RRF_K = 60           # Constant k for Reciprocal Rank Fusion formula: 1 / (rank + k)
TOP_K_DENSE = 8      # Number of candidates retrieved from ChromaDB (dense search)
TOP_K_SPARSE = 8     # Number of candidates retrieved from BM25 (sparse search)
FINAL_TOP_K = 5      # Final number of top chunks passed to the LLM context
MIN_CHUNK_WORDS = 15 # Minimum word count threshold for valid chunks

# Ensure target directories exist
os.makedirs(DATA_BASE_DIR, exist_ok=True)
for course_key, course_info in COURSES.items():
    os.makedirs(course_info["path"], exist_ok=True)

