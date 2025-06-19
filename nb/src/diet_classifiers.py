#!/usr/bin/env python3
# diet_classifiers.py

from __future__ import annotations
import ast, logging, re, string, sys, unicodedata
from pathlib import Path
from time import time
from typing import List, Tuple, Any

import joblib, pandas as pd, spacy
from sklearn.metrics import classification_report, confusion_matrix

# ─── logging / spaCy ─────────────────────────
log = logging.getLogger("diet")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(stream=sys.stderr))

log.info("Loading spaCy …")
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
KEEP_POS = {"NOUN", "PROPN"}

# ─── debug toggle ──────────────────────────────
DEBUG = True
def dbg(msg: str):
    if DEBUG:
        print(msg, file=sys.stderr)

# ─── vegan block / plant-base ──────────────────
VEGAN_BLOCK = {
    "egg", "cheese", "butter", "cream", "yogurt",
    "honey", "anchovy", "chicken", "beef", "shrimp", "sausage"
}
PLANT_BASES = {
    "almond", "cashew", "soy", "oat", "rice", "coconut",
    "hemp", "macadamia", "peanut", "walnut", "hazelnut"
}

# ─── keto allow / block ────────────────────────
KETO_ALLOW = {
    "avocado", "bacon", "olive oil", "egg", "butter", "cream", "cheese",
    "almond", "coconut", "stevia", "erythritol",
    "mushroom", "pesto", "cilantro", "shallot", "asparagus",
    "artichoke", "bay leaf", "sherry", "lemon", "ginger root",
    "green bean", "spinach", "vinegar", "olive", "walnut", "caper",
    "lime juice", "salmon fillet", "flank steak", "anchovy fillet",
    "vodka", "liqueur", "ice", "pepper sauce", "sweetener", "yogurt", "bean"
}
KETO_BLOCK = {
    "sugar","flour","rice","pasta","bread","potato","corn","oats","beans",
    "lentils","banana","apple","orange","carrot","honey","jam","cereal",
    "cooking spray"
}

# ─── regex setup ───────────────────────────────
FRACTIONS   = "¼½¾⅓⅔⅛⅜⅝⅞"
NUM         = rf"\d+(?:[./]\d+)?|[{FRACTIONS}]"
UNITS       = (
    "cups?|cup|tbsp|tablespoons?|tbs|tsp|teaspoons?|"
    "lbs?|pounds?|oz|ounces?|grams?|kg|kilograms?|ml|l|liters?|"
    "pints?|pt|quarts?|qt|gal|gallons?|inch(?:es)?"
)
UNITS_RE    = re.compile(rf"\b(?:{UNITS})\b", re.I)
SPLIT_QTY   = re.compile(rf"(?<!^)(?={NUM})")
LEADING_QTY = re.compile(rf"^\s*{NUM}\s*(?:{UNITS})?\s*", re.I)
PUNCT_TR    = str.maketrans({p: " " for p in string.punctuation if p != "-"})

# ─── extra adjectives / descriptors to drop ──
REMOVE_WORDS = (
    "fresh|large|medium|small|whole|boneless|skinless|washed|peeled|"
    "pitted|chopped|diced|sliced|shredded|grated|minced|crushed|trimmed|"
    "cleaned|piece|pieces|slice|slices|pinch|dash|taste|optional|bite|head|bay|leaf"
)
REMOVE_RE = re.compile(rf"\b(?:{REMOVE_WORDS})\b", re.I)

# ─── preprocessing ─────────────────────────────
def _fast_scrub(raw: str) -> str:
    # drop parenthetical inches
    txt = re.sub(r"\([^)]*inch[^)]*\)", " ", raw, flags=re.I)
    # normalize & lower
    txt = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode().lower()
    txt = UNITS_RE.sub(" ", txt)          # strip units (incl. inch)
    txt = re.sub(NUM, " ", txt)           # strip numbers/fractions
    txt = txt.translate(PUNCT_TR)         # drop punctuation
    txt = REMOVE_RE.sub(" ", txt)         # drop descriptors
    return re.sub(r"\s+"," ", txt).strip()

def _to_key_phrases(scrub: str) -> str:
    buff, out = [], []
    for tok in NLP(scrub):
        if tok.pos_ in KEEP_POS:
            buff.append(tok.lemma_.lower())
        else:
            if 0 < len(buff) <= 3:
                out.append(" ".join(buff))
            buff = []
    if 0 < len(buff) <= 3:
        out.append(" ".join(buff))
    return " ".join(out) if out else scrub

def preprocess(raw: str) -> str:
    cleaned = _to_key_phrases(_fast_scrub(raw))
    dbg(f"    CLEAN: '{raw}' → '{cleaned}'")
    return cleaned

# ─── parse a CSV ingredient cell ─────────────────
SINGLE_QUOTED = re.compile(r"'([^']+)'")
def to_list(cell) -> List[str]:
    if isinstance(cell, list): return cell
    if not isinstance(cell, str): return []
    txt = cell.strip()
    # python literal
    if txt.startswith("[") and txt.endswith("]"):
        hits = SINGLE_QUOTED.findall(txt)
        if hits: return [h.strip() for h in hits]
        if "," in txt:
            try: return [str(x).strip() for x in ast.literal_eval(txt)]
            except: pass
    # fallback split
    pieces = SPLIT_QTY.sub("\n", txt)
    parts = [LEADING_QTY.sub("", p).strip()
             for p in re.split(r"[\n,]+", pieces)]
    return [p for p in parts if p]

# ─── load the SVMs ──────────────────────────────
def _load_models() -> Tuple[Any,Any]:
    for root in (Path.cwd(),
                 Path(__file__).resolve().parent,
                 Path(__file__).resolve().parent.parent):
        k = root/"nutrition-ml"/"keto_svm.joblib"
        v = root/"nutrition-ml"/"vegan_svm.joblib"
        if k.exists() and v.exists():
            log.info("Loading pickles from %s", k.parent)
            return joblib.load(k), joblib.load(v)
    log.warning("‼ pickles not found – defaulting to False")
    return None, None

KETO_MODEL, VEGAN_MODEL = _load_models()

# ─── per-ingredient wrappers ─────────────────────
def is_ingredient_keto(raw: str) -> bool:
    dbg(f"[ING] RAW: '{raw}'")
    clean = preprocess(raw)
    # allow-list?
    if any(kw in clean for kw in KETO_ALLOW):
        dbg("     → ALLOW")
        return True
    # block-list?
    if any(kw in clean for kw in KETO_BLOCK):
        dbg("     → BLOCK")
        return False
    # fallback SVM
    pred = False if KETO_MODEL is None else bool(KETO_MODEL.predict([clean])[0])
    dbg(f"     → SVM: {pred}")
    return pred

def is_ingredient_vegan(raw: str) -> bool:
    dbg(f"[ING] RAW: '{raw}'")
    clean = preprocess(raw)
    toks  = clean.split()
    if toks and toks[0] in PLANT_BASES:
        dbg("     → PLANT")
        return True
    if any(tok in toks for tok in VEGAN_BLOCK):
        dbg("     → BLOCK")
        return False
    pred = False if VEGAN_MODEL is None else bool(VEGAN_MODEL.predict([clean])[0])
    dbg(f"     → SVM: {pred}")
    return pred

# ─── recipe-level helpers ────────────────────────
def is_keto(field) -> bool:
    return all(is_ingredient_keto(ing) for ing in to_list(field))

def is_vegan(field) -> bool:
    return all(is_ingredient_vegan(ing) for ing in to_list(field))

# ─── CLI ─────────────────────────────────────────
def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth",
                    default="/usr/src/data/ground_truth_sample.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.ground_truth)
    log.info("Evaluating on %s rows …", f"{len(df):,}")

    t0 = time()
    df["keto_pred"]  = df["ingredients"].apply(is_keto)
    df["vegan_pred"] = df["ingredients"].apply(is_vegan)
    log.info("Done in %.2fs", time() - t0)

    # report errors
    for idx, row in df[df["keto_pred"] != df["keto"]].iterrows():
        print(f"[KETO ERROR] row={idx} exp={row['keto']} got={row['keto_pred']} ing={to_list(row['ingredients'])}",
              file=sys.stderr)
    for idx, row in df[df["vegan_pred"] != df["vegan"]].iterrows():
        print(f"[VEGAN ERROR] row={idx} exp={row['vegan']} got={row['vegan_pred']} ing={to_list(row['ingredients'])}",
              file=sys.stderr)

    print("=== Keto ===")
    print(classification_report(df["keto"],  df["keto_pred"]))
    print(confusion_matrix(df["keto"],  df["keto_pred"]), "\n")
    print("=== Vegan ===")
    print(classification_report(df["vegan"], df["vegan_pred"]))
    print(confusion_matrix(df["vegan"], df["vegan_pred"]))

if __name__ == "__main__":
    _main()
