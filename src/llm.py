"""LLM Generation Module.

Interfaces with the Google Gemini API to generate grounded natural language answers
based on context chunks retrieved by the hybrid search pipeline.
"""

from google import genai
from google.genai import types
from src.config import GEMINI_MODEL_NAME, GEMINI_API_KEY


def generate_response(query: str, context_texts: list[str], course_name: str) -> tuple[str | None, str | None]:
    """Generates a natural language response grounded in retrieved course context.

    Constructs a prompt enforcing zero-temperature response generation restricted
    strictly to the provided educational material to prevent model hallucinations.

    Args:
        query (str): User natural language question.
        context_texts (list[str]): List of relevant document chunks retrieved by hybrid search.
        course_name (str): Human-readable name of the target educational course.

    Returns:
        tuple[str | None, str | None]: A tuple containing:
            - response_text (str | None): Generated answer text in Greek, or None on failure.
            - error_message (str | None): Error detail string if API invocation failed, else None.
    """
    prompt = f"""Είσαι βοηθός για το μάθημα '{course_name}'.
    Απάντησε ΜΟΝΟ βασισμένος στο παρακάτω κείμενο από τις διαφάνειες και τα εκπαιδευτικά υλικά.
    Αν η απάντηση δεν βρίσκεται στο υλικό, πες: Δεν βρίσκω αυτή την πληροφορία στο διαθέσιμο υλικό.
    ΜΗΝ προσθέτεις πληροφορίες που δεν υπάρχουν στο περιεχόμενο του κειμένου.
    Απάντησε στα Ελληνικά με πλήρεις προτάσεις.\n\nΕρώτηση: {query}\n\nΠληροφορίες:\n""" + "\n".join(context_texts)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                # Zero temperature guarantees deterministic output strictly grounded in context
                temperature=0,
            )
        )
        return response.text, None
    except Exception as e:
        return None, str(e)


