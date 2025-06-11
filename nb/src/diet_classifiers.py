import ast, re, sys
from argparse import ArgumentParser
from time import time
from typing import List

import pandas as pd
import spacy

try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional in Argmax image
    def classification_report(*_):
        print("sklearn not installed – skipping detailed report.")


# ─────────────────────────────────────────
# 1. spaCy model  (keep the parser enabled!)
# ─────────────────────────────────────────
NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

_UNIT_RE = re.compile(
    r"\b\d+[½¼¾⅓⅔/.\-]*\s*("
    r"cups?|tbsp|tablespoons?|tsp|teaspoons?|pounds?|lbs?|oz|ounces?|"
    r"cloves?|slices?|pinch|heads?|cans?|sticks?|fluid\s+ounces?"
    r")\b"
)
PREP_WORDS = {
    "chopped", "minced", "shredded", "sliced", "diced", "melted", "ground",
    "peeled", "deveined", "fresh", "trimmed", "grated", "crushed", "divided",
    "optional", "to", "taste"
}
SEG_SPLIT_RE = re.compile(
    r"""
    ,|;|                                     # commas / semicolons
    \s+\band\b\s+|                           # " and "
    \s+[–-]\s+|                              # dash / en-dash
    (?=\d+\s?(?:cups?|tablespoons?|tbsp|tsp|teaspoons?|pounds?|lbs?|oz|ounces?))
""",
    re.VERBOSE | re.IGNORECASE,
)

def extract_phrases(raw: str) -> list[str]:
    """Return clean noun-phrases, with **all digits removed**, for one line."""
    phrases_out: list[str] = []

    for seg in SEG_SPLIT_RE.split(raw):
        seg = seg.strip()
        if not seg:
            continue

        # 1️⃣ lowercase, drop qty + units and bare numbers
        seg = seg.lower()
        seg = _UNIT_RE.sub(" ", seg)
        seg = re.sub(r"\b\d+([./]\d+)?\b", " ", seg)
        seg = re.sub(r"[^\w\s]", " ", seg)

        # 2️⃣ kill preparation words
        tokens = [t for t in seg.split() if t not in PREP_WORDS]
        cleaned = " ".join(tokens).strip()
        if not cleaned:
            continue

        # 3️⃣ remove digits *inside* tokens (e.g. "vodka3" → "vodka")
        cleaned = re.sub(r"\d+", "", cleaned)

        # 4️⃣ noun-chunks
        doc = NLP(cleaned)
        for chunk in doc.noun_chunks:
            phrase = re.sub(r"\d+", "", chunk.text).strip()
            if phrase:
                phrases_out.append(phrase)

        # Fallback if no chunks found
        if not doc.noun_chunks:
            phrases_out.append(cleaned)

    # final de-dupe
    phrases_out = list({p for p in phrases_out if p})

    print(f"\nRAW: {raw}\n → phrases: {phrases_out}")

    return phrases_out

# ─────────────────────────────────────────
# 2. rule dictionaries  (extend as desired)
# ─────────────────────────────────────────
NON_VEGAN = {
    "egg", "eggs", "milk", "butter", "cheese", "yogurt", "cream",
    "beef", "chicken", "pork", "lamb", "bacon", "sausage",
    "fish", "anchovy", "gelatin", "lard", "honey"
}

HIGH_CARB = {
    "sugar", "flour", "bread", "rice", "pasta", "noodles", "potato",
    "corn", "beans", "lentils", "oats", "banana", "apple", "dates",
    "syrup", "honey"
}


# ─────────────────────────────────────────
# 3. ingredient-level checks
# ─────────────────────────────────────────
def is_ingredient_vegan(ing: str) -> bool:
    return not any(term in phrase for phrase in extract_phrases(ing)
                                  for term  in NON_VEGAN)


def is_ingredient_keto(ing: str) -> bool:
    return not any(term in phrase for phrase in extract_phrases(ing)
                                  for term  in HIGH_CARB)


# ─────────────────────────────────────────
# 4. recipe-level helpers
# ─────────────────────────────────────────
def is_vegan(ings: List[str]) -> bool:
    return all(is_ingredient_vegan(x) for x in ings)


def is_keto(ings: List[str]) -> bool:
    return all(is_ingredient_keto(x) for x in ings)


# ─────────────────────────────────────────
# 5. command-line evaluation
# ─────────────────────────────────────────
def main(args):
    df = pd.read_csv(args.ground_truth)
    # parse stringified lists
    if isinstance(df["ingredients"].iloc[0], str):
        df["ingredients"] = df["ingredients"].apply(ast.literal_eval)

    t0 = time()
    df["keto_pred"]  = df["ingredients"].apply(is_keto)
    df["vegan_pred"] = df["ingredients"].apply(is_vegan)
    t1 = time()

    print("=== Keto ===")
    print(classification_report(df["keto"],  df["keto_pred"]))
    print("=== Vegan ===")
    print(classification_report(df["vegan"], df["vegan_pred"]))
    print(f"Completed in {t1 - t0:.2f} s")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--ground_truth",
                   default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(p.parse_args()))