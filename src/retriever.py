"""
Hybrid Retriever Module combining Dense Vector Search (ChromaDB + Sentence Transformers)
and Sparse Keyword Search (BM25Okapi) using Reciprocal Rank Fusion (RRF).
"""

import os
import string
import pickle
import numpy as np
import chromadb
import unicodedata
from sentence_transformers import SentenceTransformer
from src.config import (
    COURSES, CHROMA_PATH, EMBEDDING_MODEL_NAME,
    RRF_K, TOP_K_DENSE, TOP_K_SPARSE, FINAL_TOP_K
)

def tokenize_greek(text: str) -> list[str]:
    """
    Tokenizes Greek text by normalizing accents/diacritics (NFD), removing punctuation,
    converting to lowercase, and filtering out single-character tokens.
    """
    text = text.lower()
    # Strip Greek accents using Unicode NFD normalization
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    # Remove punctuation & special symbols
    for char in ['-', '«', '»', '“', '”', '·', ':', ';', '!', '?', ',', '.', '(', ')', '[', ']', '{', '}']:
        text = text.replace(char, ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.split() if len(t) > 1]

def rrf_score(dense_ranks: dict[str, int], sparse_ranks: dict[str, int], k_rrf: int = RRF_K) -> list[tuple[str, float]]:
    """
    Calculates Reciprocal Rank Fusion (RRF) scores to combine rankings from dense and sparse search.
    RRF Formula: RRF_score(d) = sum(1 / (rank(d) + k)) for each retrieval model.
    """
    scores = {}
    for chunk_id, rank in dense_ranks.items():
        scores[chunk_id] = 1 / (rank + k_rrf)
    for chunk_id, rank in sparse_ranks.items():
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + k_rrf)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

class Retriever:
    """
    Hybrid Retriever class that manages course-specific vector stores and BM25 indexes.
    """
    def __init__(self, course_key: str = "databases"):
        if course_key not in COURSES:
            raise ValueError(f"Άγνωστο μάθημα: {course_key}. Διαθέσιμα: {list(COURSES.keys())}")

        self.course_key = course_key
        self.course_info = COURSES[course_key]

        print(f"Φόρτωση μοντέλων και ευρετηρίων για '{self.course_info['name']}'...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection(name=self.course_info["collection"])

        bm25_path = self.course_info["bm25_index"]
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο BM25: {bm25_path}. Τρέξτε πρώτα το ingestion.")

        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]
        self.all_chunks = bm25_data["chunks"]
        self.all_metadata = bm25_data["metadata"]
        self.all_ids = bm25_data["ids"]
        self.id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(self.all_ids)}

    def retrieve(self, query: str):
        """
        Executes hybrid retrieval for a given query:
        1. Queries ChromaDB for top dense candidates via vector similarity.
        2. Queries BM25 for top sparse candidates via token matching.
        3. Fuses rankings using RRF and returns formatted context and source metadata.
        """
        # 1. Dense Search (Semantic Similarity)
        query_embedding = self.model.encode([query]).tolist()
        dense_results = self.collection.query(query_embeddings=query_embedding, n_results=TOP_K_DENSE)

        dense_ranks = {}
        if dense_results["ids"] and len(dense_results["ids"]) > 0:
            for idx, chunk_id in enumerate(dense_results["ids"][0]):
                dense_ranks[chunk_id] = idx + 1

        # 2. Sparse Search (Keyword Matching)
        tokenized_query = tokenize_greek(query)
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:TOP_K_SPARSE]

        sparse_ranks = {}
        for rank, idx in enumerate(top_sparse_indices):
            chunk_id = self.all_ids[idx]
            if sparse_scores[idx] > 0:
                sparse_ranks[chunk_id] = rank + 1

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_ranking = rrf_score(dense_ranks, sparse_ranks)
        rrf_ranking = [(cid, score) for cid, score in rrf_ranking if score > 0.01]
        top_n = [item[0] for item in rrf_ranking[:FINAL_TOP_K]]

        if not top_n:
            return None, None

        # Format retrieved context and map source pages
        context_texts = []
        sources_with_pages = {}

        for chunk_id in top_n:
            idx = self.id_to_idx[chunk_id]
            metadata = self.all_metadata[idx]
            source_name = metadata['source']

            if 'page' in metadata:
                page_num = metadata['page'] + 1
                context_texts.append(f"Πηγή: {source_name} (Σελίδα: {page_num})\nΚείμενο: {self.all_chunks[idx]}")

                if source_name not in sources_with_pages:
                    sources_with_pages[source_name] = set()
                sources_with_pages[source_name].add(page_num)
            else:
                context_texts.append(f"Πηγή: {source_name}\nΚείμενο: {self.all_chunks[idx]}")

                if source_name not in sources_with_pages:
                    sources_with_pages[source_name] = set()
        return context_texts, sources_with_pages

