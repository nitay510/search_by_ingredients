#!/usr/bin/env python3
# diet_classifiers.py

from __future__ import annotations
import ast, logging, re, string, sys, unicodedata
from pathlib import Path
from time import time
from typing import List, Tuple, Any

import joblib, pandas as pd, spacy
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────── logging / spaCy ───────────────────────────────
log = logging.getLogger("diet")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(stream=sys.stderr))

log.info("Loading spaCy …")
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
KEEP_POS = {"NOUN", "PROPN"}

# ─────────────────── vegan block‐list & plant‐base whitelist ────────────────
VEGAN_BLOCK = {
    "egg", "cheese", "butter", "cream", "yogurt",
    "honey", "anchovy", "chicken", "beef", "shrimp", "sausage"
}
PLANT_BASES = {
    "almond", "cashew", "soy", "oat", "rice", "coconut",
    "hemp", "macadamia", "peanut", "walnut", "hazelnut"
}

# ─────────────────── keto must‐yes / must‐no lists ─────────────────────────
KETO_ALLOW = {
    "avocado", "bacon", "olive oil", "egg", "butter", "cream", "cheese",
    "almond", "coconut", "stevia", "erythritol"
}
KETO_BLOCK = {
    "sugar","flour","rice","pasta","bread","potato","corn","oats","beans",
    "lentils","banana","apple","orange","carrot","honey","jam","cereal"
}

# ───────────────────────── quantity regexes ──────────────────────────
FRACTIONS   = "¼½¾⅓⅔⅛⅜⅝⅞"
NUM         = rf"\d+(?:[./]\d+)?|[{FRACTIONS}]"
UNITS       = (
    "cups?|cup|tbsp|tablespoons?|tbs|tsp|teaspoons?|"
    "lbs?|pounds?|oz|ounces?|grams?|kg|kilograms?|ml|l|liters?|"
    "pints?|pt|quarts?|qt|gal|gallons?"
)
UNITS_RE    = re.compile(rf"\b(?:{UNITS})\b", re.I)
SPLIT_QTY   = re.compile(rf"(?<!^)(?={NUM})")
LEADING_QTY = re.compile(rf"^\s*{NUM}\s*(?:{UNITS})?\s*", re.I)
PUNCT_TR    = str.maketrans({p: " " for p in string.punctuation if p != "-"})

# ───────────────────────── preprocessing ─────────────────────────────
def _fast_scrub(raw: str) -> str:
    txt = unicodedata.normalize("NFKD", raw).encode("ascii","ignore").decode()
    txt = txt.lower()
    txt = UNITS_RE.sub(" ", txt)
    txt = re.sub(NUM, " ", txt)
    txt = txt.translate(PUNCT_TR)
    return re.sub(r"\s+"," ", txt).strip()

def _to_key_phrases(scrubbed: str) -> str:
    buff, out = [], []
    for tok in NLP(scrubbed):
        if tok.pos_ in KEEP_POS:
            buff.append(tok.lemma_.lower())
        else:
            if 0 < len(buff) <= 3:
                out.append(" ".join(buff))
            buff = []
    if 0 < len(buff) <= 3:
        out.append(" ".join(buff))
    return " ".join(out) if out else scrubbed

def preprocess(raw: str) -> str:
    return _to_key_phrases(_fast_scrub(raw))

# ───────────────── ingredient‐cell → list[str] ───────────────────────
SINGLE_QUOTED = re.compile(r"'([^']+)'")
def to_list(cell) -> List[str]:
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str):
        return []
    txt = cell.strip()
    # 1) Python‐literal list?
    if txt.startswith("[") and txt.endswith("]") and "," in txt:
        try:
            return [str(x).strip() for x in ast.literal_eval(txt)]
        except Exception:
            pass
    # 2) numpy‐style one‐liner?
    hits = SINGLE_QUOTED.findall(txt)
    if hits:
        return hits
    # 3) fallback heuristic
    pieces = SPLIT_QTY.sub("\n", txt)
    parts = [LEADING_QTY.sub("", p).strip()
             for p in re.split(r"[\n,]+", pieces)]
    return [p for p in parts if p]

# ───────────────── load your SVMs ───────────────────────────────────
def _load_models() -> Tuple[Any,Any]:
    for root in (Path.cwd(),
                 Path(__file__).resolve().parent,
                 Path(__file__).resolve().parent.parent):
        k = root/"nutrition-ml"/"keto_svm.joblib"
        v = root/"nutrition-ml"/"vegan_svm.joblib"
        if k.exists() and v.exists():
            log.info("Loading pickles from %s", k.parent)
            return joblib.load(k), joblib.load(v)
    log.warning("‼  pickles not found – defaulting to False")
    return None, None

KETO_MODEL, VEGAN_MODEL = _load_models()

# ─────────────────── per‐ingredient wrappers ─────────────────────────
def is_ingredient_keto(raw: str) -> bool:
    clean = preprocess(raw)
    toks  = clean.split()
    # 1) explicit allowed fats & proteins
    if any(tok in KETO_ALLOW for tok in toks):
        return True
    # 2) obvious carb blockers
    if any(tok in KETO_BLOCK for tok in toks):
        return False
    # 3) fallback to SVM
    if KETO_MODEL is None:
        return False
    return bool(KETO_MODEL.predict([clean])[0])
def is_ingredient_vegan(raw: str) -> bool:
    clean = preprocess(raw)
    toks  = clean.split()
    if toks and toks[0] in PLANT_BASES:
        return True
    if any(tok in VEGAN_BLOCK for tok in toks):
        return False
    if VEGAN_MODEL is None:
        return False
    return bool(VEGAN_MODEL.predict([clean])[0])

# ─────────────────── recipe‐level helpers (short‐circuit) ────────────
def is_keto(field) -> bool:
    return all(map(is_ingredient_keto, to_list(field)))

def is_vegan(field) -> bool:
    return all(map(is_ingredient_vegan, to_list(field)))

# ────────────────────────────── CLI ────────────────────────────────
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
    elapsed = time() - t0
    log.info("Done in %.2fs", elapsed)

    # ── debug: print every wrong keto prediction
    for idx, row in df[df["keto_pred"] != df["keto"]].iterrows():
        ing = to_list(row["ingredients"])
        print(f"[KETO ERROR] row={idx}  expected={row['keto']}  got={row['keto_pred']}",
              f"ingredients={ing}", file=sys.stderr)

    # ── debug: print every wrong vegan prediction
    for idx, row in df[df["vegan_pred"] != df["vegan"]].iterrows():
        ing = to_list(row["ingredients"])
        print(f"[VEGAN ERROR] row={idx}  expected={row['vegan']}  got={row['vegan_pred']}",
              f"ingredients={ing}", file=sys.stderr)

    # ── finally, the metrics
    print("=== Keto ===")
    print(classification_report(df["keto"],  df["keto_pred"]))
    print(confusion_matrix(df["keto"],  df["keto_pred"]), "\n")

    print("=== Vegan ===")
    print(classification_report(df["vegan"], df["vegan_pred"]))
    print(confusion_matrix(df["vegan"], df["vegan_pred"]))

if __name__ == "__main__":
    _main()
