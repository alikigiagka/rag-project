"""Data Ingestion Pipeline Module.

Processes educational documents (TXT, PDF), extracts text and embedded figures using
Gemini Vision LLM with persistent JSON caching, generates dense vector embeddings for ChromaDB,
and constructs sparse BM25 index files.
"""

import os
import glob
import fitz
import chromadb
import json
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import string
from tqdm import tqdm
import io
import hashlib
import unicodedata
from PIL import Image
from google import genai
from src.config import (
    COURSES, CHROMA_PATH, EMBEDDING_MODEL_NAME, MIN_CHUNK_WORDS,
    GEMINI_VISION_MODEL_NAME, GEMINI_API_KEY, PROCESS_IMAGES, DATA_BASE_DIR
)

# Minimum pixel dimensions to filter out UI logos, icons, bullet points, and decorative elements
MIN_IMAGE_WIDTH = 500
MIN_IMAGE_HEIGHT = 250

VISION_CACHE_FILE = os.path.join(DATA_BASE_DIR, "vision_cache.json")


def load_vision_cache() -> dict[str, str]:
    """Loads the persistent image descriptions cache from disk.

    Returns:
        dict[str, str]: Dictionary mapping MD5 image hashes to generated text descriptions.
    """
    if os.path.exists(VISION_CACHE_FILE):
        try:
            with open(VISION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading vision cache: {e}")
    return {}


def save_vision_cache(cache: dict[str, str]) -> None:
    """Saves the image descriptions cache to a persistent JSON file on disk.

    Args:
        cache (dict[str, str]): Mapping of MD5 image hashes to their text descriptions.
    """
    try:
        os.makedirs(os.path.dirname(VISION_CACHE_FILE), exist_ok=True)
        with open(VISION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving vision cache: {e}")


def tokenize_greek(text: str) -> list[str]:
    """Tokenizes Greek text for BM25 sparse index construction.

    Strips diacritics via Unicode NFD decomposition, converts text to lower case,
    removes punctuation marks, and filters out single-character tokens.

    Args:
        text (str): Input raw text string.

    Returns:
        list[str]: Filtered list of normalized token strings.
    """
    text = text.lower()
    # NFD decomposition separates base characters from combining diacritical accents
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    for char in ['-', '«', '»', '“', '”', '·', ':', ';', '!', '?', ',', '.', '(', ')', '[', ']', '{', '}']:
        text = text.replace(char, ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.split() if len(t) > 1]


def describe_image_with_vision_llm(image_bytes: bytes, page_num: int = 1) -> str:
    """Generates detailed text descriptions for extracted images using Gemini Vision LLM.

    Transcribes diagrams, ER models, and mathematical formulas present in PDF slides
    with backoff and retry handling for rate limits.

    Args:
        image_bytes (bytes): Raw binary stream of the image file.
        page_num (int, optional): Source PDF page index for logging. Defaults to 1.

    Returns:
        str: Detailed textual description of the visual content in Greek.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    img = Image.open(io.BytesIO(image_bytes))

    vision_prompt = (
        "Περίγραψε αναλυτικά αυτή την εικόνα στα Ελληνικά. "
        "Αν περιέχει διαγράμματα οντοτήτων-συσχετίσεων (ER), ανέλυσε τις οντότητες και τις σχέσεις. "
        "Αν περιέχει μαθηματικούς τύπους, κώδικα ή σύμβολα σχεσιακής άλγεβρας, μετάγραψέ τα με απόλυτη ακρίβεια. "
        "Μην προσθέσεις περιττά σχόλια, εστίασε στο περιεχόμενο της εικόνας."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_VISION_MODEL_NAME,
                contents=[img, vision_prompt]
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg:
                # 65-second sleep ensures clearing the minute-level quota window imposed by Gemini API limits
                wait_time = 65
                print(f"\n  [Rate Limit] Page {page_num}: Waiting {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  Vision LLM error (Page {page_num}): {e}")
                return f"[Σφάλμα περιγραφής εικόνας: {e}]"

    print(f"  Vision LLM failed after {max_retries} retries due to quota limits.")
    return "[Αποτυχία λόγω εξάντλησης Rate Limit]"


def run_ingestion(course_key: str | None = None) -> None:
    """Executes the end-to-end data ingestion pipeline.

    Processes documents (TXT, PDF), extracts slide text and multimodal image content,
    chunks text into semantic blocks, computes dense embeddings for ChromaDB,
    and builds sparse BM25 index files.

    Args:
        course_key (str | None, optional): Specific course identifier to process,
            or None to process all configured courses. Defaults to None.
    """
    print("Loading Sentence-Transformer model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    vision_cache = load_vision_cache()
    print(f"Loaded {len(vision_cache)} cached image descriptions from disk.")

    if course_key is None:
        courses_to_process = COURSES.items()
    else:
        if course_key not in COURSES:
            print(f"Error: Unknown course '{course_key}'")
            print(f"Available courses: {', '.join(COURSES.keys())}")
            return
        courses_to_process = [(course_key, COURSES[course_key])]

    for course_key, course_info in courses_to_process:
        print(f"\n=== Processing course: {course_info['name']} ===")

        data_dir = course_info["path"]
        collection_name = course_info["collection"]
        bm25_path = course_info["bm25_index"]

        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Removed previous vector collection '{collection_name}'")
        except Exception:
            pass

        collection = chroma_client.get_or_create_collection(name=collection_name)

        chunks = []
        metadata = []
        ids = []
        chunk_id = 0

        print(f"Processing TXT files in {data_dir}...")
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        for filepath in tqdm(txt_files, desc="TXT Processing"):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
            for block in blocks:
                if len(block.split()) < MIN_CHUNK_WORDS:
                    continue
                chunks.append(block)
                metadata.append({"source": os.path.basename(filepath), "type": "txt"})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

        print(f"Processing PDF files in {data_dir}...")
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

        for filepath in tqdm(pdf_files, desc="PDF Processing"):
            doc = fitz.open(filepath)
            page_texts = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()

                if PROCESS_IMAGES:
                    image_list = page.get_images(full=True)
                    for image_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            image_data = doc.extract_image(xref)

                            w = image_data.get("width", 0)
                            h = image_data.get("height", 0)

                            # Filter low-resolution images and small slide decorations by pixel area
                            if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                                continue
                            if w * h < 150000:
                                continue

                            image_bytes = image_data["image"]
                            # MD5 hash identifies duplicate images across slides to reuse cached LLM descriptions
                            img_hash = hashlib.md5(image_bytes).hexdigest()

                            if img_hash in vision_cache:
                                vision_text = vision_cache[img_hash]
                            else:
                                print(f"\n  Image {image_index+1} page {page_num+1} ({w}x{h}px) - submitting to Vision LLM...")
                                vision_text = describe_image_with_vision_llm(image_bytes, page_num=page_num+1)
                                if vision_text and not vision_text.startswith("[Σφάλμα") and not vision_text.startswith("[Αποτυχία"):
                                    vision_cache[img_hash] = vision_text
                                    save_vision_cache(vision_cache)

                            if len(vision_text.strip()) > 20 and not vision_text.startswith("[Σφάλμα") and not vision_text.startswith("[Αποτυχία"):
                                text += f"\n[VISION_EXTRACT - Εικόνα {image_index+1}]: {vision_text}"

                        except Exception as e:
                            print(f"  Image extraction error at index {image_index} page {page_num}: {e}")
                            continue

                if text:
                    page_texts.append((page_num, text))

            # Concatenate sequential short pages until achieving minimum semantic chunk size
            buffer_text = ""
            buffer_start_page = 0
            for page_num, text in page_texts:
                if not buffer_text:
                    buffer_text = text
                    buffer_start_page = page_num
                elif len(buffer_text.split()) < MIN_CHUNK_WORDS:
                    buffer_text += " " + text
                else:
                    if len(buffer_text.split()) >= MIN_CHUNK_WORDS:
                        chunks.append(buffer_text)
                        metadata.append({
                            "source": os.path.basename(filepath),
                            "page": buffer_start_page,
                            "type": "pdf"
                        })
                        ids.append(f"chunk_{chunk_id}")
                        chunk_id += 1
                    buffer_text = text
                    buffer_start_page = page_num

            if buffer_text and len(buffer_text.split()) >= MIN_CHUNK_WORDS:
                chunks.append(buffer_text)
                metadata.append({
                    "source": os.path.basename(filepath),
                    "page": buffer_start_page,
                    "type": "pdf"
                })
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

        if not chunks:
            print(f"No valid documents found in {data_dir}. Skipping...")
            continue

        print(f"Storing {len(chunks)} chunks in ChromaDB...")
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

        print("Building BM25 index...")
        tokenized_corpus = [tokenize_greek(doc) for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        if os.path.exists(bm25_path):
            os.remove(bm25_path)

        with open(bm25_path, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "chunks": chunks,
                "metadata": metadata,
                "ids": ids
            }, f)

        print(f"Course '{course_info['name']}' ingestion completed successfully.")

    print("\n=== Ingestion process finished successfully! ===")



