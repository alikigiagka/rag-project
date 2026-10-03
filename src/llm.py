"""
LLM Generation Module.
Interfaces with Google Gemini API to generate grounded answers based on retrieved RAG context.
"""

import os
from google import genai
from google.genai import types
from src.config import GEMINI_MODEL_NAME, GEMINI_API_KEY

def generate_response(query: str, context_texts: list[str], course_name: str) -> tuple[str | None, str | None]:
    """
    Generates a natural language response in Greek using Google Gemini.
    Enforces a strict grounding policy to prevent hallucinations by restricting answers 
    solely to the retrieved educational context.

    Returns:
        tuple: (response_text, error_message)
    """
    prompt = f"""Είσαι βοηθός για το μάθημα '{course_name}'.
    Απάντησε ΜΟΝΟ βασισμένος στο παρακάτω κείμενο από τις διαφάνειες και τα εκπαιδευτικά υλικά.
    Αν η απάντηση δεν βρίσκεται στο υλικό, πες: Δεν βρίσκω αυτή την πληροφορία στο διαθέσιμο υλικό.
    ΜΗΝ προσθέτεις πληροφορίες που δεν υπάρχουν στο παριεχόμενο κείμενο.
    Απάντησε στα Ελληνικά με πλήρεις προτάσεις.\n\nΕρώτηση: {query}\n\nΠληροφορίες:\n""" + "\n".join(context_texts)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,  # Zero temperature for deterministic, grounded output
            )
        )
        return response.text, None
    except Exception as e:
        return None, str(e)

