import ast, re, sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Set

import pandas as pd
import spacy

NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

STOP_PHRASES = {
    "optional", "to", "taste", "needed", "as", "or", "more", "divided",
}
UNIT_RE = re.compile(
    r"\b\d+[½¼¾⅓⅔/.\-]*\s*(?:cups?|tbsp|tablespoons?|tsp|teaspoons?|"
    r"pounds?|lbs?|oz|ounces?|cloves?|slices?|pinch|heads?|cans?|sticks?"
    r"|fluid\s+ounces?)\b",
    flags=re.IGNORECASE,
)
SEG_SPLIT = re.compile(r",|;|\s+\band\b\s+|\s+[–-]\s+", flags=re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(?:[./]\d+)?\b")

def clean_segment(seg: str) -> str | None:
    """Return a cleaned ingredient phrase or None if trash."""
    seg = seg.lower()
    seg = UNIT_RE.sub(" ", seg)
    seg = NUM_RE.sub(" ", seg)
    seg = re.sub(r"[^\w\s]", " ", seg)
    seg = " ".join(t for t in seg.split() if t not in STOP_PHRASES)
    if not seg.strip():
        return None

    doc = NLP(seg)
    words: List[str] = []

    for chunk in doc.noun_chunks:
        # keep adjectives directly to the left of the noun
        parts = [
            tok.lemma_ for tok in chunk
            if tok.pos_ in {"NOUN", "ADJ"} and len(tok.lemma_) > 1
        ]
        if parts:
            words.append(" ".join(parts))

    return words[0] if words else None

def extract_phrases(raw: str) -> List[str]:
    phrases = []
    for seg in SEG_SPLIT.split(raw):
        seg = seg.strip()
        if not seg:
            continue
        cleaned = clean_segment(seg)
        if cleaned:
            phrases.append(cleaned)
    return phrases

def build_list(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    if isinstance(df["ingredients"].iloc[0], str):
        df["ingredients"] = df["ingredients"].apply(ast.literal_eval)

    unique: Set[str] = set()
    for ing_list in df["ingredients"]:
        for raw in ing_list:
            unique.update(extract_phrases(raw))

    pd.DataFrame(sorted(unique), columns=["ingredient"]).to_csv(output_csv, index=False)
    print(f"Wrote {len(unique):,} clean ingredients → {output_csv}")

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--input_csv", default="/usr/src/data/ground_truth_sample.csv")
    ap.add_argument("--output_csv", default="unique_ingredients.csv")
    args = ap.parse_args()
    build_list(args.input_csv, args.output_csv)
