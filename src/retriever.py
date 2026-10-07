"""Hybrid Retriever Module combining Dense Vector Search (ChromaDB + Sentence Transformers)

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
    """Tokenizes Greek text for BM25 sparse keyword search.

    Normalizes accents and diacritics using Unicode NFD decomposition, converts text to lowercase,
    strips punctuation marks, and excludes single-character tokens.

    Args:
        text (str): Input query or text string.

    Returns:
        list[str]: Normalized token string sequence.
    """
    text = text.lower()
    # NFD decomposition isolates character accents to facilitate stripping combining marks
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    for char in ['-', '«', '»', '“', '”', '·', ':', ';', '!', '?', ',', '.', '(', ')', '[', ']', '{', '}']:
        text = text.replace(char, ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.split() if len(t) > 1]


def rrf_score(
    dense_ranks: dict[str, int],
    sparse_ranks: dict[str, int],
    k_rrf: int = RRF_K
) -> list[tuple[str, float]]:
    """Calculates Reciprocal Rank Fusion (RRF) scores to combine candidate rankings.

    Evaluates candidate document scores using the formula:
    score(d) = sum(1 / (rank(d) + k)) across dense and sparse rank sets.

    Args:
        dense_ranks (dict[str, int]): Mapping of chunk IDs to their dense vector retrieval rank.
        sparse_ranks (dict[str, int]): Mapping of chunk IDs to their sparse BM25 retrieval rank.
        k_rrf (int, optional): Rank dampening constant. Defaults to RRF_K (60).

    Returns:
        list[tuple[str, float]]: List of (chunk_id, fused_score) tuples sorted in descending relevance order.
    """
    scores = {}
    for chunk_id, rank in dense_ranks.items():
        scores[chunk_id] = 1 / (rank + k_rrf)
    for chunk_id, rank in sparse_ranks.items():
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank + k_rrf)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Retriever:
    """Hybrid Retriever executing combined dense vector similarity and sparse BM25 search.

    Attributes:
        course_key (str): Identifier of the active course.
        course_info (dict): Metadata dictionary for the active course.
        model (SentenceTransformer): Embedding model instance.
        chroma_client (chromadb.PersistentClient): Client connection to ChromaDB vector store.
        collection (chromadb.Collection): ChromaDB collection storing course chunk embeddings.
        bm25 (BM25Okapi): Sparse BM25 index built over course document corpus.
        all_chunks (list[str]): Complete raw text chunks matching index positions.
        all_metadata (list[dict]): Document metadata associated with each chunk.
        all_ids (list[str]): Unique string identifiers assigned to document chunks.
        id_to_idx (dict[str, int]): Mapping dictionary resolving chunk IDs to corpus list indices.
    """

    def __init__(self, course_key: str = "databases") -> None:
        """Initializes the hybrid retriever for a given course.

        Args:
            course_key (str, optional): Key identifier of the course. Defaults to "databases".

        Raises:
            ValueError: If course_key is not configured in COURSES.
            FileNotFoundError: If the pickled BM25 index file does not exist on disk.
        """
        if course_key not in COURSES:
            raise ValueError(f"Unknown course: {course_key}. Available: {list(COURSES.keys())}")

        self.course_key = course_key
        self.course_info = COURSES[course_key]

        print(f"Loading retriever models and indexes for '{self.course_info['name']}'...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection(name=self.course_info["collection"])

        bm25_path = self.course_info["bm25_index"]
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index file not found at: {bm25_path}. Run ingestion first.")

        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]
        self.all_chunks = bm25_data["chunks"]
        self.all_metadata = bm25_data["metadata"]
        self.all_ids = bm25_data["ids"]
        # Pre-build hash index for O(1) metadata lookups by chunk ID
        self.id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(self.all_ids)}

    def retrieve(self, query: str) -> tuple[list[str] | None, dict[str, set[int]] | None]:
        """Executes hybrid retrieval combining dense vector similarity and sparse BM25 scoring.

        Performs semantic query embedding for ChromaDB search, tokenizes query for BM25 score calculation,
        fuses candidate rankings using Reciprocal Rank Fusion (RRF), and formats top results.

        Args:
            query (str): User natural language search query.

        Returns:
            tuple[list[str] | None, dict[str, set[int]] | None]: A tuple containing:
                - context_texts (list[str] | None): Formatted context strings with source citations,
                  or None if no relevant documents match.
                - sources_with_pages (dict[str, set[int]] | None): Mapping of document filenames
                  to sets of 1-based page numbers cited, or None if no documents match.
        """
        query_embedding = self.model.encode([query]).tolist()
        dense_results = self.collection.query(query_embeddings=query_embedding, n_results=TOP_K_DENSE)

        dense_ranks = {}
        if dense_results["ids"] and len(dense_results["ids"]) > 0:
            for idx, chunk_id in enumerate(dense_results["ids"][0]):
                dense_ranks[chunk_id] = idx + 1

        tokenized_query = tokenize_greek(query)
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:TOP_K_SPARSE]

        sparse_ranks = {}
        for rank, idx in enumerate(top_sparse_indices):
            chunk_id = self.all_ids[idx]
            # Ignore zero-score BM25 results to avoid polluting fusion rankings with non-matching documents
            if sparse_scores[idx] > 0:
                sparse_ranks[chunk_id] = rank + 1

        rrf_ranking = rrf_score(dense_ranks, sparse_ranks)
        # Threshold at 0.01 excludes candidates appearing low in both rank branches
        rrf_ranking = [(cid, score) for cid, score in rrf_ranking if score > 0.01]
        top_n = [item[0] for item in rrf_ranking[:FINAL_TOP_K]]

        if not top_n:
            return None, None

        context_texts = []
        sources_with_pages = {}

        for chunk_id in top_n:
            idx = self.id_to_idx[chunk_id]
            metadata = self.all_metadata[idx]
            source_name = metadata['source']

            if 'page' in metadata:
                # Convert zero-indexed internal PDF page offset to 1-indexed citation label
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


