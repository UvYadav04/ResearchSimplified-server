import re
from SafeExecution.safeExecution import safeExecution
@safeExecution
def is_noise(text: str) -> bool:
    if not text:
        return True

    text = text.strip()
    words = text.split()
    n = len(words)

    # 🔥 Step 1: Long text → keep
    if n > 12:
        return False
    # print(paragraph)
    lower = text.lower()

    # --- 1. Metadata / identifiers ---
    if re.search(r"(arxiv|doi|issn|isbn|www\.|http|https)", lower):
        return True

    # --- 2. Conference / publisher / copyright ---
    if re.search(
        r"(conference|journal|proceedings|published|copyright|all rights reserved)",
        lower,
    ):
        return True

    # --- 3. Section headings like "2 RELATED WORK", "3.1 METHOD"
    if re.match(r"^\d+(\.\d+)*\s+[A-Z\s\-\(\)]+$", text):
        return True

    # --- 4. Mostly uppercase text (common in headers)
    upper_words = sum(1 for w in words if w.isupper())
    if n > 0 and (upper_words / n) > 0.6:
        return True

    # --- 5. Page numbers / standalone numbers
    if re.match(r"^\d+$", text):
        return True

    if re.match(r"^(page|p\.)\s*\d+$", lower):
        return True

    # --- 6. Dates
    if re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b", text):  # 3 Jun 2021
        return True

    # --- 7. Broken spacing (like "V ISION T RANSFORMER")
    if re.search(r"(?:\b[A-Z]\s+){2,}[A-Z]\b", text):
        return True

    # --- 8. Too short + no sentence structure
    if n < 6 and not re.search(r"[.!?]", text):
        return True

    # --- 9. Mostly symbols / garbage
    if re.match(r"^[^\w\s]+$", text):
        return True

    # --- 10. Table-like fragments
    if "|" in text or "\t" in text:
        return True

    # --- 11. Figure / table references
    if re.match(r"^(figure|fig\.?|table)\s*\d+", lower):
        return True

    # --- 12. Emails
    if re.search(r"\S+@\S+", text):
        return True

    # --- 13. Affiliations (heuristic)
    if re.search(r"(university|institute|department|lab)", lower) and n < 10:
        return True

    # --- 14. Single word junk (except useful ones)
    if n == 1:
        if lower not in ["introduction", "abstract", "conclusion", "method"]:
            return True

    return False
