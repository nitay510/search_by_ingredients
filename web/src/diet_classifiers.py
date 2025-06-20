
import re
import string
import unicodedata
from functools import lru_cache
from typing import List
from pathlib import Path

import joblib
import spacy

# ─── load spaCy (only tokenizer & tagger for speed) ─────────────────
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
KEEP_POS = {"NOUN", "PROPN"}

# ─── hard lists for rules ───────────────────────────────────────────
VEGAN_BLOCK = {
    "egg","cheese","butter","cream","yogurt","honey",
    "anchovy","chicken","beef","shrimp","sausage",
    "fish","pork","lamb","goat","duck","turkey","bacon",
    "milk","gelatin","honeycomb","caviar","oyster",
    "clam","crab","lobster","scallop","shrimp",
    "squid","octopus","mollusk","shellfish","scallop",
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

# ─── try loading SVM models if available ───────────────────────────
_base   = Path(__file__).resolve().parent
_models = _base / "nutrition-ml"
try:
    KETO_MODEL  = joblib.load(str(_models / "keto_svm.joblib"))
    VEGAN_MODEL = joblib.load(str(_models / "vegan_svm.joblib"))
except Exception:
    KETO_MODEL = VEGAN_MODEL = None

# ─── regex for removing quantities, units, punctuation ──────────────
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
    """Lowercase, drop numbers/units/punctuation/descriptors."""
    t = re.sub(r"\([^)]*inch[^)]*\)", " ", raw, flags=re.I)
    t = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode().lower()
    t = UNITS_RE.sub(" ", t)
    t = re.sub(NUM, " ", t)
    t = t.translate(PUNCT_TR)
    t = REMOVE_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def _to_key_phrases(text: str) -> str:
    """Keep only short noun‐lemma chunks (1–3 words)."""
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

@lru_cache(maxsize=8192)
def preprocess(raw: str) -> str:
    """Full clean + noun‐phrase extraction, cached for speed."""
    return _to_key_phrases(_fast_scrub(raw))

# ─── public functions ────────────────────────────────────────────────

def is_ingredient_keto(raw: str) -> bool:
    """True if ingredient is keto‐friendly."""
    norm = preprocess(raw)
    # 1) any blocked term? → not keto
    if any(blk in norm for blk in KETO_BLOCK):
        return False
    # 2) any allowed term? → keto
    if any(ok in norm for ok in KETO_ALLOW):
        return True
    # 3) SVM fallback if available
    if KETO_MODEL:
        return bool(KETO_MODEL.predict([norm])[0])
    # 4) default to False
    return False

def is_ingredient_vegan(raw: str) -> bool:
    """True if ingredient is vegan."""
    norm = preprocess(raw)
    toks = norm.split()
    # 1) plant‐base in first position → vegan (e.g. "almond milk")
    if toks and toks[0] in PLANT_BASES:
        return True
    # 2) block list → not vegan
    if any(blk in toks for blk in VEGAN_BLOCK):
        return False
    # 3) SVM fallback if available
    if VEGAN_MODEL:
        return bool(VEGAN_MODEL.predict([norm])[0])
    # 4) default to False
    return False

def is_keto(ings: List[str]) -> bool:
    """True if all ingredients are keto‐friendly."""
    return all(is_ingredient_keto(i) for i in ings)

def is_vegan(ings: List[str]) -> bool:
    """True if all ingredients are vegan."""
    return all(is_ingredient_vegan(i) for i in ings)
