"""
Data Ingestion Pipeline Module.
Processes educational documents (TXT, PDF), extracts text and embedded images (using Gemini Vision LLM with persistent JSON caching & Rate Limit retries),
generates dense vector embeddings for ChromaDB, and builds sparse BM25 index files.
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
from src.config import COURSES, CHROMA_PATH, EMBEDDING_MODEL_NAME, MIN_CHUNK_WORDS, GEMINI_VISION_MODEL_NAME, GEMINI_API_KEY, PROCESS_IMAGES, DATA_BASE_DIR

# Thresholds for filtering out low-resolution images/icons during PDF extraction
MIN_IMAGE_WIDTH  = 500
MIN_IMAGE_HEIGHT = 250

# Persistent Vision Cache file location
VISION_CACHE_FILE = os.path.join(DATA_BASE_DIR, "vision_cache.json")

def load_vision_cache() -> dict[str, str]:
    """Loads persistent image descriptions cache from JSON file."""
    if os.path.exists(VISION_CACHE_FILE):
        try:
            with open(VISION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Σφάλμα κατά τη φόρτωση του vision cache: {e}")
    return {}

def save_vision_cache(cache: dict[str, str]):
    """Saves image descriptions cache to persistent JSON file."""
    try:
        os.makedirs(os.path.dirname(VISION_CACHE_FILE), exist_ok=True)
        with open(VISION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Σφάλμα κατά την αποθήκευση του vision cache: {e}")

def tokenize_greek(text: str) -> list[str]:
    """
    Tokenizes Greek text for BM25 sparse index building by removing diacritics and punctuation.
    """
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    for char in ['-', '«', '»', '“', '”', '·', ':', ';', '!', '?', ',', '.', '(', ')', '[', ']', '{', '}']:
        text = text.replace(char, ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.split() if len(t) > 1]

def describe_image_with_vision_llm(image_bytes: bytes, page_num: int = 1) -> str:
    """
    Sends an extracted image byte stream to Google Gemini Vision LLM to generate a detailed Greek text description 
    of diagrams, ER models, equations, or structural figures found in course slides.
    Includes rate-limit retry handling (429/RESOURCE_EXHAUSTED).
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
                wait_time = 65
                print(f"\n  [Rate Limit] Σελίδα {page_num}: Αναμονή {wait_time}s... (Προσπάθεια {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  Σφάλμα Vision LLM (Σελίδα {page_num}): {e}")
                return f"[Σφάλμα περιγραφής εικόνας: {e}]"

    print(f"  Αποτυχία Vision LLM μετά από {max_retries} προσπάθειες λόγω εξάντλησης Quota.")
    return "[Αποτυχία λόγω εξάντλησης Rate Limit]"

def run_ingestion(course_key: str | None = None):
    """
    Main ingestion process:
    1. Reads TXT and PDF documents from course directories.
    2. Extracts slide text and uses Gemini Vision to describe diagrams/figures.
    3. Uses persistent JSON cache for Vision API calls to avoid redundant costs/limits.
    4. Chunks text into semantic blocks.
    5. Computes dense embeddings with SentenceTransformer and stores in ChromaDB.
    6. Tokenizes chunks and builds BM25 sparse index (.pkl).
    """
    print("Φόρτωση μοντέλου Sentence-Transformer...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Load persistent vision cache from disk
    vision_cache = load_vision_cache()
    print(f"Φορτώθηκαν {len(vision_cache)} αποθηκευμένες περιγραφές εικόνων από το cache.")

    if course_key is None:
        courses_to_process = COURSES.items()
    else:
        if course_key not in COURSES:
            print(f"Σφάλμα: Άγνωστο μάθημα '{course_key}'")
            print(f"Διαθέσιμα μαθήματα: {', '.join(COURSES.keys())}")
            return
        courses_to_process = [(course_key, COURSES[course_key])]

    for course_key, course_info in courses_to_process:
        print(f"\n=== Επεξεργασία μαθήματος: {course_info['name']} ===")

        data_dir        = course_info["path"]
        collection_name = course_info["collection"]
        bm25_path       = course_info["bm25_index"]

        # Reset existing vector collection for fresh ingestion
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Διαγράφηκαν τα παλιά δεδομένα από '{collection_name}'")
        except:
            pass

        collection = chroma_client.get_or_create_collection(name=collection_name)

        chunks   = []
        metadata = []
        ids      = []
        chunk_id = 0

        # 1. Process TXT documents
        print(f"Επεξεργασία TXT αρχείων από {data_dir}...")
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

        # 2. Process PDF documents and extract multimodal image content
        print(f"Επεξεργασία PDF αρχείων από {data_dir}...")
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
                            xref       = img[0]
                            image_data = doc.extract_image(xref)

                            w = image_data.get("width",  0)
                            h = image_data.get("height", 0)
                            if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
                                continue
                            if w * h < 150000:
                                continue

                            image_bytes = image_data["image"]
                            img_hash    = hashlib.md5(image_bytes).hexdigest()

                            if img_hash in vision_cache:
                                vision_text = vision_cache[img_hash]
                            else:
                                print(f"\n  Εικόνα {image_index+1} σελίδας {page_num+1} ({w}x{h}px) - αποστολή στο Vision LLM...")
                                vision_text = describe_image_with_vision_llm(image_bytes, page_num=page_num+1)
                                if vision_text and not vision_text.startswith("[Σφάλμα") and not vision_text.startswith("[Αποτυχία"):
                                    vision_cache[img_hash] = vision_text
                                    save_vision_cache(vision_cache)

                            if len(vision_text.strip()) > 20 and not vision_text.startswith("[Σφάλμα") and not vision_text.startswith("[Αποτυχία"):
                                text += f"\n[VISION_EXTRACT - Εικόνα {image_index+1}]: {vision_text}"

                        except Exception as e:
                            print(f"  Σφάλμα εξαγωγής εικόνας {image_index} σελίδας {page_num}: {e}")
                            continue

                if text:
                    page_texts.append((page_num, text))

            # Merge text into chunks meeting minimum word length
            buffer_text       = ""
            buffer_start_page = 0
            for page_num, text in page_texts:
                if not buffer_text:
                    buffer_text       = text
                    buffer_start_page = page_num
                elif len(buffer_text.split()) < MIN_CHUNK_WORDS:
                    buffer_text += " " + text
                else:
                    if len(buffer_text.split()) >= MIN_CHUNK_WORDS:
                        chunks.append(buffer_text)
                        metadata.append({"source": os.path.basename(filepath),
                                         "page": buffer_start_page, "type": "pdf"})
                        ids.append(f"chunk_{chunk_id}")
                        chunk_id += 1
                    buffer_text       = text
                    buffer_start_page = page_num

            if buffer_text and len(buffer_text.split()) >= MIN_CHUNK_WORDS:
                chunks.append(buffer_text)
                metadata.append({"source": os.path.basename(filepath),
                                 "page": buffer_start_page, "type": "pdf"})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

        if not chunks:
            print(f"Δεν βρέθηκαν έγγραφα στο φάκελο {data_dir}. Παράλειψη...")
            continue

        # 3. Store Dense Embeddings in ChromaDB
        print(f"Αποθήκευση {len(chunks)} chunks στην ChromaDB...")
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
        collection.add(documents=chunks, embeddings=embeddings,
                       metadatas=metadata, ids=ids)

        # 4. Build and Save Sparse BM25 Index
        print("Χτίσιμο του ευρετηρίου BM25...")
        tokenized_corpus = [tokenize_greek(doc) for doc in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        if os.path.exists(bm25_path):
            os.remove(bm25_path)

        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": chunks,
                         "metadata": metadata, "ids": ids}, f)

        print(f"Το μάθημα '{course_info['name']}' επεξεργάστηκε επιτυχώς!")

    print("\n=== Η διαδικασία Ingestion ολοκληρώθηκε επιτυχώς! ===")


