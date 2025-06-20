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
NLP = spacy.blank("en")
KEEP_POS = {"NOUN", "PROPN"}

# ─────────────────── vegan / keto globals ───────────────────────────
VEGAN_BLOCK = {"egg","cheese","butter","cream","yogurt","honey",
               "anchovy","chicken","beef","shrimp","sausage"}
PLANT_BASES = {"almond","cashew","soy","oat","rice","coconut",
               "hemp","macadamia","peanut","walnut","hazelnut"}

KETO_ALLOW = {"avocado","bacon","olive oil","egg","butter","cream",
              "cheese","almond","coconut","stevia","erythritol",
              "mushroom","pesto","cilantro","shallot","asparagus",
              "artichoke","bay leaf","sherry","lemon","ginger root",
              "green bean","spinach","vinegar","olive","walnut","caper",
              "lime juice","salmon fillet","flank steak","anchovy fillet",
              "vodka","liqueur","ice","pepper sauce","sweetener","yogurt","bean"}
KETO_BLOCK = {"sugar","flour","rice","pasta","bread","potato","corn",
              "oats","lentils","banana","apple","orange","carrot","honey",
              "jam","cereal","cooking spray"}

# these will be populated in main()
_KETO_MAP: dict[str,bool]  = {}
_VEGAN_MAP: dict[str,bool] = {}

# ───────────────────────── quantity regexes ──────────────────────────
FRACTIONS   = "¼½¾⅓⅔⅛⅜⅝⅞"
NUM         = rf"\d+(?:[./]\d+)?|[{FRACTIONS}]"
UNITS_LIST  = [
    "cup","cups","tbsp","tablespoon","tablespoons","tbs","tsp","teaspoon","teaspoons",
    "pound","pounds","lb","lbs","ounce","ounces","oz",
    "gram","grams","kg","kilogram","kilograms","ml","l","liter","liters",
    "pint","pints","pt","quart","quarts","qt","gal","gallon","gallons","inch","inches"
]
UNITS_RE    = re.compile(rf"\b(?:{'|'.join(map(re.escape, UNITS_LIST))})\b", re.I)
SPLIT_QTY   = re.compile(rf"(?<!^)(?={NUM})")
LEADING_QTY = re.compile(rf"^\s*{NUM}\s*(?:{'|'.join(map(re.escape, UNITS_LIST))})?\s*", re.I)
PUNCT_TR    = str.maketrans({p: " " for p in string.punctuation if p != "-"})
REMOVE_RE   = re.compile(r"\b(?:fresh|large|medium|small|whole|boneless|skinless|washed|peeled|pitted|chopped|diced|sliced|shredded|grated|minced|crushed|trimmed|cleaned|piece|pieces|slice|slices|pinch|dash|taste|optional|bite|head|bay|leaf)\b", re.I)

# ────────────────────────── preprocessing ────────────────────────────
def _fast_scrub(raw: str) -> str:
    txt = re.sub(r"\([^)]*inch[^)]*\)", " ", raw, flags=re.I)
    txt = unicodedata.normalize("NFKD", txt).encode("ascii","ignore").decode().lower()
    txt = UNITS_RE.sub(" ", txt)
    txt = re.sub(NUM, " ", txt)
    txt = txt.translate(PUNCT_TR)
    txt = REMOVE_RE.sub(" ", txt)
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
    return _to_key_phrases(_fast_scrub(raw))

# ────────────────── explode CSV / numpy cells ────────────────────────
SINGLE_QUOTED = re.compile(r"'([^']+)'"")]
def to_list(cell) -> List[str]:
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str):
        return []
    txt = cell.strip()
    if txt.startswith("[") and txt.endswith("]"):
        hits = SINGLE_QUOTED.findall(txt)
        if hits:
            return [h.strip() for h in hits]
        if "," in txt:
            try:
                return [str(x).strip() for x in ast.literal_eval(txt)]
            except:
                pass
    pieces = SPLIT_QTY.sub("\n", txt)
    parts  = [LEADING_QTY.sub("", p).strip()
              for p in re.split(r"[\n,]+", pieces)]
    return [p for p in parts if p]

# ─────────────────── load pre‐trained SVMs ───────────────────────────
def _load_models() -> Tuple[Any,Any]:
    for root in (Path.cwd(),
                 Path(__file__).resolve().parent,
                 Path(__file__).resolve().parent.parent):
        k = root/"nutrition-ml"/"keto_svm.joblib"
        v = root/"nutrition-ml"/"vegan_svm.joblib"
        if k.exists() and v.exists():
            log.info("Loading pickles from %s", k.parent)
            return joblib.load(k), joblib.load(v)
    log.warning("‼ pickles missing – default to False")
    return None, None

KETO_MODEL, VEGAN_MODEL = _load_models()

# ─────────────────── wrapper that uses our global maps ────────────────
def is_ingredient_keto(raw: str) -> bool:
    return _KETO_MAP.get(raw, False)

def is_ingredient_vegan(raw: str) -> bool:
    return _VEGAN_MAP.get(raw, False)

def is_keto(ings: List[str]) -> bool:
    return all(is_ingredient_keto(i) for i in ings)

def is_vegan(ings: List[str]) -> bool:
    return all(is_ingredient_vegan(i) for i in ings)

# ────────────────────────────── main ─────────────────────────────────
def main(args):
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    try:
        start_time = time()
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(
            is_vegan)

        end_time = time()
    except Exception as e:
        print(f"Error: {e}")
        return -1

    print("===Keto===")
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred']))
    print("===Vegan===")
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time} seconds ==")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str,
                        default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))
