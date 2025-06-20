#!/usr/bin/env python3
# web/src/diet_classifiers.py

import logging, re, string, unicodedata
from typing import List
import joblib
import spacy
from pathlib import Path

# ─── logging ─────────────────────────────────────────────────────────
log = logging.getLogger("diet")
log.setLevel(logging.DEBUG)      # Switch to INFO to silence
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
log.addHandler(h)

# ─── load spaCy (only tokenizer & tagger) ────────────────────────────
log.info("Loading spaCy…")
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
KEEP_POS = {"NOUN", "PROPN"}

# ─── vegan & keto hard rules ─────────────────────────────────────────
VEGAN_BLOCK = {
    "egg","cheese","butter","cream","yogurt","honey",
    "anchovy","chicken","beef","shrimp","sausage"
}
PLANT_BASES = {
    "almond","cashew","soy","oat","rice","coconut",
    "hemp","macadamia","peanut","walnut","hazelnut"
}

KETO_BLOCK = {
    "sugar","flour","rice","pasta","bread","potato","corn",
    "oats","lentils","banana","apple","orange","carrot","honey",
    "jam","cereal","cooking spray"
}
KETO_ALLOW = {
    "avocado","bacon","olive oil","egg","butter","cream","cheese",
    "almond","coconut","stevia","erythritol","mushroom","pesto",
    "cilantro","shallot","asparagus","artichoke","bay leaf","sherry",
    "lemon","ginger","green bean","spinach","vinegar","olive",
    "walnut","caper","lime juice","salmon fillet","flank steak",
    "anchovy fillet","vodka","liqueur","ice","yogurt","bean"
}

# ─── load SVMs if present ────────────────────────────────────────────
_base = Path(__file__).resolve().parent               # .../web
_models = _base / "nutrition-ml"                       # .../web/nutrition-ml
_keto_pkl = _models / "keto_svm.joblib"
_vegan_pkl = _models / "vegan_svm.joblib"

try:
    KETO_MODEL  = joblib.load(str(_keto_pkl))
    VEGAN_MODEL = joblib.load(str(_vegan_pkl))
    log.info(f"✅ Loaded SVMs from {_models}/")
except Exception as e:
    KETO_MODEL = VEGAN_MODEL = None
    log.warning(f"⚠️  Could not load SVMs from {_models}/ ({e}) — falling back to rules only")

# ─── quantity/unit stripping & punctuation removal ───────────────────
FRACTIONS = "¼½¾⅓⅔⅛⅜⅝⅞"
NUM       = rf"\d+(?:[./]\d+)?|[{FRACTIONS}]"
UNITS_RE  = re.compile(
    r"\b(?:cups?|tbsp|tablespoons?|tbs|tsp|teaspoons?|"
    r"lbs?|pounds?|oz|ounces?|grams?|kg|kilograms?|ml|l|liters?|"
    r"pints?|pt|quarts?|qt|gal|gallons?|inch(?:es)?)\b", re.I
)
PUNCT_TR  = str.maketrans({p: " " for p in string.punctuation if p != "-"})
REMOVE_RE = re.compile(
    r"\b(?:fresh|large|medium|small|whole|boneless|skinless|"
    r"washed|peeled|pitted|chopped|diced|sliced|shredded|"
    r"grated|minced|crushed|trimmed|cleaned|piece|pieces|slice|"
    r"slices|pinch|dash|taste|optional|bite|head|bay|leaf)\b",
    re.I
)

def _fast_scrub(raw: str) -> str:
    # remove "(…inch…)" then normalize
    t = re.sub(r"\([^)]*inch[^)]*\)", " ", raw, flags=re.I)
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode().lower()
    t = UNITS_RE.sub(" ", t)
    t = re.sub(NUM, " ", t)
    t = t.translate(PUNCT_TR)
    t = REMOVE_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def _to_key_phrases(text: str) -> str:
    buff, out = [], []
    for tok in NLP(text):
        if tok.pos_ in KEEP_POS:
            buff.append(tok.lemma_.lower())
        else:
            if 0 < len(buff) <= 3:
                out.append(" ".join(buff))
            buff = []
    if 0 < len(buff) <= 3:
        out.append(" ".join(buff))
    return " ".join(out) if out else text

def preprocess(raw: str) -> str:
    """Clean + extract noun‐phrase keys."""
    return _to_key_phrases(_fast_scrub(raw))

# ─── public API ─────────────────────────────────────────────────────

def is_ingredient_keto(raw: str) -> bool:
    norm = preprocess(raw)
    toks = norm.split()

    # BLOCK → ALLOW → SVM
    if any(t in KETO_BLOCK for t in toks):
        log.debug(f"[KETO-BLOCK] '{raw}' → False")
        return False

    if any(t in KETO_ALLOW for t in toks):
        log.debug(f"[KETO-ALLOW] '{raw}' → True")
        return True

    if KETO_MODEL:
        p = bool(KETO_MODEL.predict([norm])[0])
        log.debug(f"[KETO-SVM] '{raw}' → {p}")
        return p

    log.debug(f"[KETO-DEFAULT] '{raw}' → False")
    return False

def is_ingredient_vegan(raw: str) -> bool:
    norm = preprocess(raw)
    toks = norm.split()

    if toks and toks[0] in PLANT_BASES:
        log.debug(f"[VEGAN-PLANT] '{raw}' → True")
        return True

    if any(t in VEGAN_BLOCK for t in toks):
        log.debug(f"[VEGAN-BLOCK] '{raw}' → False")
        return False

    if VEGAN_MODEL:
        p = bool(VEGAN_MODEL.predict([norm])[0])
        log.debug(f"[VEGAN-SVM] '{raw}' → {p}")
        return p
    else:
        log.warning(f"[VEGAN-DEFAULT] '{raw}' → False (no model)")
    return False

def is_keto(ings: List[str]) -> bool:
    return all(is_ingredient_keto(i) for i in ings)

def is_vegan(ings: List[str]) -> bool:
    return all(is_ingredient_vegan(i) for i in ings)
