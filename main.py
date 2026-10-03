"""
CLI Interface for Multi-Course RAG System.
Provides command-line arguments to trigger ingestion or start an interactive Q&A session with evaluation logging.
"""

import argparse
import sys
import warnings
import json
import os
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import COURSES
from src.ingest import run_ingestion
from src.retriever import Retriever
from src.llm import generate_response

def show_available_courses():
    """Prints the list of supported courses configured in the system."""
    print("\nΔιαθέσιμα μαθήματα:")
    for key, info in COURSES.items():
        print(f"  {key}: {info['name']}")

def save_queries_to_json(queries: list[dict], filename: str = "rag_evaluation_data.json"):
    """
    Appends interactive QA session data (queries, taxonomy, retrieved contexts, and answers)
    to a JSON dataset for RAG evaluation.
    """
    if not queries:
        return

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    existing_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        except Exception as e:
            print(f"\nΣφάλμα κατά την ανάγνωση του αρχείου {filename}: {e}")
            existing_data = []

    existing_data.extend(queries)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"\nΟι ερωτήσεις αποθηκεύτηκαν επιτυχώς στο αρχείο: {file_path}")
    except Exception as e:
        print(f"\nΣφάλμα κατά την αποθήκευση στο αρχείο {filename}: {e}")

def chat(course_key: str = "databases", session_queries: list[dict] | None = None):
    """
    Runs the interactive CLI chat loop for a selected course.
    Handles user queries, hybrid document retrieval, Gemini response generation, and evaluation logging.
    """
    course_info = COURSES.get(course_key)
    if not course_info:
        print(f"Σφάλμα: Άγνωστο μάθημα '{course_key}'")
        show_available_courses()
        return None

    try:
        retriever = Retriever(course_key=course_key)
    except Exception as e:
        print(f"Σφάλμα κατά την φόρτωση του retriever: {e}")
        print("Βεβαιωθείτε ότι έχετε τρέξει το ingestion πρώτα (για να δημιουργηθούν τα indexes).")
        print("Χρήση: python main.py --course <course_name> --ingest")
        return None

    print(f"Το σύστημα για '{course_info['name']}' είναι έτοιμο!")
    print("Πληκτρολογήστε 'exit', 'quit' ή 'courses' για αλλαγή μαθήματος.")

    while True:
        try:
            query = input("\nΕρώτηση: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "exit_abrupt"

        if not query:
            continue

        if query.lower() in ['exit', 'quit']:
            return "exit_graceful"

        if query.lower() == 'courses':
            show_available_courses()
            try:
                selected = input("Επιλέξτε μάθημα (databases/parallel-distributed): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "exit_abrupt"
            if selected in COURSES:
                return selected
            elif selected in ['exit', 'quit']:
                return "exit_graceful"
            else:
                print(f"Άγνωστο μάθημα: {selected}")
            continue

        # Execute hybrid retrieval
        context_texts, sources_with_pages = retriever.retrieve(query)

        if not context_texts:
            print("Δεν βρέθηκαν αρκετά σχετικά έγγραφα για να απαντηθεί η ερώτηση.")
            continue

        # Generate response using LLM
        response_text, error = generate_response(query, context_texts, course_info['name'])

        if error:
            print("\n[Σφάλμα κατά την επικοινωνία με το Gemini API]")
            print(f"Λεπτομέρειες: {error}")
        else:
            print("\n--- Απάντηση ---")
            print(response_text)

            # Display source files and specific page citations
            sources_display = []
            for source_name, pages in sorted(sources_with_pages.items()):
                if pages:
                    pages_str = ', '.join(str(p) for p in sorted(pages))
                    sources_display.append(f"{source_name} (σελ. {pages_str})")
                else:
                    sources_display.append(source_name)

            print(f"\n[Πηγές: {', '.join(sources_display)}]")

            if session_queries is not None:
                session_queries.append({
                    "query": query,
                    "retrieved_context": "\n\n".join(context_texts),
                    "generated_answer": response_text
                })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Course Local RAG Project CLI")
    parser.add_argument("--ingest", action="store_true", help="Εκτέλεση της διαδικασίας ingestion για το επιλεγμένο μάθημα")
    parser.add_argument("--course", type=str, required=True,
                       help=f"Επιλογή μαθήματος (υποχρεωτικό - διαθέσιμα: {', '.join(COURSES.keys())})")
    args = parser.parse_args()

    if args.ingest:
        run_ingestion(course_key=args.course)
    else:
        current_course = args.course
        session_queries = []
        graceful_exit = False

        while current_course and current_course not in ["exit_graceful", "exit_abrupt"]:
            result = chat(course_key=current_course, session_queries=session_queries)
            if result in COURSES:
                current_course = result
            elif result == "exit_graceful":
                graceful_exit = True
                current_course = "exit_graceful"
            else:
                current_course = "exit_abrupt"

        if graceful_exit:
            save_queries_to_json(session_queries)

