# preprocess_only.py
"""
Extract the “clean” noun-phrases from a raw ingredient line
so you can inspect what eventually reaches is_vegan / is_keto.

Run:
    python preprocess_only.py "2 cups chopped walnuts"
    python preprocess_only.py --file sample_ings.txt
"""

import re, unicodedata, sys, argparse, spacy
from pathlib import Path
from typing import List

# ───────────────────────────── spaCy ──────────────────────────────
NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

# units & “prep‐words” copied from your snippet
_UNIT_RE = re.compile(
    r"\b\d+[½¼¾⅓⅔/.\-]*\s*("
    r"cups?|tbsp|tablespoons?|tsp|teaspoons?|pounds?|lbs?|oz|ounces?|"
    r"cloves?|slices?|pinch|heads?|cans?|sticks?|fluid\s+ounces?"
    r")\b", re.I)
PREP_WORDS = {
    "chopped","minced","shredded","sliced","diced","melted","ground",
    "peeled","deveined","fresh","trimmed","grated","crushed","divided",
    "optional","to","taste"
}

SEG_SPLIT_RE = re.compile(
    r""",
      |;|
      \s+\band\b\s+|
      \s+[–-]\s+|
      (?=\d+\s?(?:cups?|tablespoons?|tbsp|tsp|teaspoons?|pounds?|lbs?|oz|ounces?))
    """,
    re.VERBOSE | re.I,
)

def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

def extract_phrases(raw: str) -> List[str]:
    """Return a **deduped list** of lower-cased noun phrases."""
    phrases: list[str] = []

    for seg in SEG_SPLIT_RE.split(raw):
        seg = seg.strip()
        if not seg:
            continue

        # 1️⃣ lowercase, ASCII-only, remove qty + units + bare numbers
        seg = _ascii(seg.lower())
        seg = _UNIT_RE.sub(" ", seg)
        seg = re.sub(r"\b\d+([./]\d+)?\b", " ", seg)
        seg = re.sub(r"[^\w\s]", " ", seg)

        # 2️⃣ drop preparation words
        tokens = [tok for tok in seg.split() if tok not in PREP_WORDS]
        cleaned = " ".join(tokens).strip()
        if not cleaned:
            continue

        # 3️⃣ spaCy noun-chunks
        doc = NLP(cleaned)
        for chunk in doc.noun_chunks:
            phrase = re.sub(r"\d+", "", chunk.text).strip()
            if phrase:
                phrases.append(phrase)

        # fallback if spaCy found nothing
        if not list(doc.noun_chunks):
            phrases.append(cleaned)

    # 4️⃣ final dedupe + keep only non-empty
    return sorted({p for p in phrases if p})

# ───────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("line", nargs="?", help="single ingredient line")
    ap.add_argument("--file", help="txt file with one ingredient per line")
    args = ap.parse_args()

    if args.line:
        lines = [args.line]
    elif args.file:
        lines = Path(args.file).read_text().splitlines()
    else:
        sys.exit("Provide a line or --file")

    for raw in lines:
        print(f"\nRAW  : {raw}")
        print(f"PHRASES → {extract_phrases(raw)}")
