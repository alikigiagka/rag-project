import json
import os

def migrate():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_evaluation_data.json")
    if not os.path.exists(file_path):
        print("Το αρχείο rag_evaluation_data.json δεν βρέθηκε.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Σφάλμα κατά την ανάγνωση: {e}")
        return

    if not isinstance(data, list):
        print("Μη αναμενόμενη δομή αρχείου (δεν είναι λίστα).")
        return

    updated_count = 0
    for item in data:
        if "question_type" not in item:
            item["question_type"] = "Factual Recall"
            updated_count += 1

    if updated_count > 0:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Ενημερώθηκαν επιτυχώς {updated_count} ερωτήσεις στο {file_path}.")
        except Exception as e:
            print(f"Σφάλμα κατά την αποθήκευση: {e}")
    else:
        print("Δεν χρειάστηκε να ενημερωθεί καμία ερώτηση (όλες έχουν ήδη question_type).")

if __name__ == "__main__":
    migrate()
