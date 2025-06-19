#!/usr/bin/env python3
# web/src/diet_classifiers.py

from __future__ import annotations
import logging, re, string, unicodedata
from pathlib import Path
from typing import List, Any

import joblib
import spacy

# ──────────────── logging & spaCy ─────────────────
log = logging.getLogger("diet")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())
NLP = spacy.blank("en")
KEEP_POS = {"NOUN", "PROPN"}

# ─────────── vegan‐ and keto‐whitelists / blacklists ───────────────
VEGAN_BLOCK = {
    "egg", "cheese", "butter", "cream", "yogurt",
    "honey", "anchovy", "chicken", "beef", "shrimp", "sausage"
}
PLANT_BASES = {
    "almond", "cashew", "soy", "oat", "rice", "coconut",
    "hemp", "macadamia", "peanut", "walnut", "hazelnut"
}

KETO_ALLOW = {
    "avocado", "bacon", "olive oil", "egg", "butter", "cream", "cheese",
    "almond", "coconut", "stevia", "erythritol", "mushroom", "pesto",
    "cilantro", "shallot", "asparagus", "artichoke", "bay leaf", "sherry",
    "lemon", "ginger root", "green bean", "spinach", "vinegar", "olive",
    "walnut", "caper", "lime juice", "salmon fillet", "flank steak",
    "anchovy fillet", "vodka", "liqueur", "ice", "pepper sauce",
    "sweetener", "yogurt", "bean"
}
KETO_BLOCK = {
    "sugar", "flour", "rice", "pasta", "bread", "potato", "corn",
    "oats", "lentils", "banana", "apple", "orange", "carrot",
    "honey", "jam", "cereal", "cooking spray"
}

# ──────────────── quantity/unit stripping ─────────────────
FRACTIONS = "¼½¾⅓⅔⅛⅜⅝⅞"
NUM      = rf"\d+(?:[./]\d+)?|[{FRACTIONS}]"
UNITS    = [
    "cup","cups","tbsp","tablespoon","tablespoons","tbs","tsp","teaspoon","teaspoons",
    "pound","pounds","lb","lbs","ounce","ounces","oz",
    "gram","grams","kg","kilogram","kilograms","ml","l","liter","liters",
    "pint","pints","pt","quart","quarts","qt","gal","gallon","gallons",
    "inch","inches"
]
UNITS_RE    = re.compile(rf"\b(?:{'|'.join(map(re.escape, UNITS))})\b", re.I)
PUNCT_TR    = str.maketrans({p: " " for p in string.punctuation if p != "-"})
REMOVE_RE   = re.compile(
    r"\b(?:fresh|large|medium|small|whole|boneless|skinless|washed|peeled|"
    r"pitted|chopped|diced|sliced|shredded|grated|minced|crushed|trimmed|"
    r"cleaned|piece|pieces|slice|slices|pinch|dash|taste|optional|bite|"
    r"head|bay|leaf)\b", re.I
)
LEADING_QTY = re.compile(rf"^\s*{NUM}\s*(?:{'|'.join(map(re.escape, UNITS))})?\s*", re.I)

def _fast_scrub(raw: str) -> str:
    # drop any "(...inch...)" parentheticals first
    txt = re.sub(r"\([^)]*inch[^)]*\)", " ", raw, flags=re.I)
    txt = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode().lower()
    txt = UNITS_RE.sub(" ", txt)
    txt = re.sub(NUM, " ", txt)
    txt = txt.translate(PUNCT_TR)
    txt = REMOVE_RE.sub(" ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def _to_key_phrases(scrub: str) -> str:
    buff, out = [], []
    for tok in NLP(scrub):
        if tok.pos_ in KEEP_POS:
            buff.append(tok.lemma_)
        else:
            if 0 < len(buff) <= 3:
                out.append(" ".join(buff))
            buff = []
    if 0 < len(buff) <= 3:
        out.append(" ".join(buff))
    return " ".join(out) if out else scrub

def preprocess(raw: str) -> str:
    return _to_key_phrases(_fast_scrub(raw))

# ─────────── load the pickled SVMs once ─────────────────
def _load_models() -> tuple[Any,Any]:
    for root in (Path.cwd(),
                 Path(__file__).resolve().parent,
                 Path(__file__).resolve().parent.parent):
        k = root/"nutrition-ml"/"keto_svm.joblib"
        v = root/"nutrition-ml"/"vegan_svm.joblib"
        if k.exists() and v.exists():
            log.info("Loading SVMs from %s", k.parent)
            return joblib.load(k), joblib.load(v)
    log.warning("SVM pickles not found – defaulting to always‐False")
    return None, None

KETO_MODEL, VEGAN_MODEL = _load_models()

# ─────────── public API for the Flask app ───────────────────
def is_ingredient_keto(ing: str) -> bool:
    clean = preprocess(ing)
    toks  = clean.split()
    # 1) explicit keto‐safe items
    if any(tok in clean for tok in KETO_ALLOW):
        return True
    # 2) obvious high‐carb blockers
    if any(tok in clean for tok in KETO_BLOCK):
        return False
    # 3) fallback to SVM
    if KETO_MODEL is None:
        return False
    return bool(KETO_MODEL.predict([clean])[0])

def is_ingredient_vegan(ing: str) -> bool:
    clean = preprocess(ing)
    toks  = clean.split()
    # 1) plant‐based first words
    if toks and toks[0] in PLANT_BASES:
        return True
    # 2) block known animal products
    if any(tok in toks for tok in VEGAN_BLOCK):
        return False
    # 3) fallback to SVM
    if VEGAN_MODEL is None:
        return False
    return bool(VEGAN_MODEL.predict([clean])[0])

def is_keto(ingredients: List[str]) -> bool:
    return all(is_ingredient_keto(ing) for ing in ingredients)

def is_vegan(ingredients: List[str]) -> bool:
    return all(is_ingredient_vegan(ing) for ing in ingredients)
