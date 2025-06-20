#!/usr/bin/env python3
# diet_classifiers.py

import ast
import json
import sys
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from time import time
from typing import List

import joblib
import pandas as pd
import re, string, unicodedata
import spacy

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    def classification_report(y, y_pred):
        print('sklearn is not installed, skipping classification report')
    def confusion_matrix(y, y_pred):
        return []

# ─── load spaCy (tokenizer+tagger only) ─────────────────────────────
NLP = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
KEEP_POS = {'NOUN', 'PROPN'}

# ─── rule lists ────────────────────────────────────────────────────
VEGAN_BLOCK = {
    'egg','cheese','butter','cream','yogurt','honey',
    'anchovy','chicken','beef','shrimp','sausage'
}
PLANT_BASES = {
    'almond','cashew','soy','oat','rice','coconut',
    'hemp','macadamia','peanut','walnut','hazelnut'
}

KETO_BLOCK = {
    'sugar','flour','rice','pasta','bread','potato','corn',
    'oats','lentils','banana','apple','orange','carrot',
    'honey','jam','cereal','cooking spray'
}
KETO_ALLOW = {
    'avocado','bacon','olive oil','egg','butter','cream','cheese',
    'almond','coconut','stevia','erythritol','mushroom','pesto',
    'cilantro','shallot','asparagus','artichoke','bay leaf','sherry',
    'lemon','ginger root','green bean','spinach','vinegar','olive',
    'walnut','caper','lime juice','salmon fillet','flank steak',
    'anchovy fillet','vodka','liqueur','ice','yogurt','bean'
}

# ─── load SVMs if present ───────────────────────────────────────────
_base   = Path(__file__).resolve().parent
_models = _base / 'nutrition-ml'
try:
    KETO_MODEL  = joblib.load(str(_models / 'keto_svm.joblib'))
    VEGAN_MODEL = joblib.load(str(_models / 'vegan_svm.joblib'))
except Exception:
    KETO_MODEL = VEGAN_MODEL = None

# ─── regex setup ────────────────────────────────────────────────────
FRACTIONS = '¼½¾⅓⅔⅛⅜⅝⅞'
NUM       = rf'\d+(?:[./]\d+)?|[{FRACTIONS}]'
UNITS     = (
    'cups?|cup|tbsp|tablespoons?|tbs|tsp|teaspoons?|'
    'lbs?|pounds?|oz|ounces?|grams?|kg|kilograms?|ml|l|liters?|'
    'pints?|pt|quarts?|qt|gal|gallons?|inch(?:es)?'
)
UNITS_RE  = re.compile(rf'\b(?:{UNITS})\b', re.I)
PUNCT_TR  = str.maketrans({p:' ' for p in string.punctuation if p!='-'})
REMOVE_RE = re.compile(
    r'\b(?:fresh|large|medium|small|whole|boneless|skinless|'
    r'washed|peeled|pitted|chopped|diced|sliced|shredded|'
    r'grated|minced|crushed|trimmed|cleaned|piece|pieces|slice|'
    r'slices|pinch|dash|taste|optional|bite|head|bay|leaf)\b',
    re.I
)

def _fast_scrub(raw: str) -> str:
    t = re.sub(r'\([^)]*inch[^)]*\)', ' ', raw, flags=re.I)
    t = unicodedata.normalize('NFKD', t).encode('ascii','ignore').decode().lower()
    t = UNITS_RE.sub(' ', t)
    t = re.sub(NUM, ' ', t)
    t = t.translate(PUNCT_TR)
    t = REMOVE_RE.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def _to_key_phrases(text: str) -> str:
    buff, out = [], []
    for tok in NLP(text):
        if tok.pos_ in KEEP_POS:
            buff.append(tok.lemma_.lower())
        else:
            if 0 < len(buff) <= 3:
                out.append(' '.join(buff))
            buff = []
    if 0 < len(buff) <= 3:
        out.append(' '.join(buff))
    return ' '.join(out) if out else text

@lru_cache(maxsize=8192)
def preprocess(raw: str) -> str:
    """Clean + noun‐phrase extraction, cached for speed."""
    return _to_key_phrases(_fast_scrub(raw))

# ─── mimic web’s to_list fallback ─────────────────────────────────
SPLIT_QTY   = re.compile(rf'(?<!^)(?={NUM})')
LEADING_QTY = re.compile(rf'^\s*{NUM}\s*(?:{UNITS})?\s*', re.I)
SINGLE_QUOTED = re.compile(r"'([^']+)'")

def to_list(cell) -> List[str]:
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str):
        return []
    txt = cell.strip()
    if txt.startswith('[') and txt.endswith(']'):
        hits = SINGLE_QUOTED.findall(txt)
        if hits:
            return [h.strip() for h in hits]
        if ',' in txt:
            try:
                return [str(x).strip() for x in ast.literal_eval(txt)]
            except:
                pass
    pieces = SPLIT_QTY.sub('\n', txt)
    parts  = [LEADING_QTY.sub('', p).strip() for p in re.split(r'[\n,]+', pieces)]
    return [p for p in parts if p]

# ─── per‐ingredient tests ──────────────────────────────────────────
def is_ingredient_keto(raw: str) -> bool:
    norm = preprocess(raw)
    if any(b in norm for b in KETO_BLOCK):
        return False
    if any(a in norm for a in KETO_ALLOW):
        return True
    if KETO_MODEL:
        return bool(KETO_MODEL.predict([norm])[0])
    return False

def is_ingredient_vegan(raw: str) -> bool:
    norm = preprocess(raw)
    toks = norm.split()
    if toks and toks[0] in PLANT_BASES:
        return True
    if any(b in toks for b in VEGAN_BLOCK):
        return False
    if VEGAN_MODEL:
        return bool(VEGAN_MODEL.predict([norm])[0])
    return False

def is_keto(ings: List[str]) -> bool:
    return all(is_ingredient_keto(i) for i in ings)

def is_vegan(ings: List[str]) -> bool:
    return all(is_ingredient_vegan(i) for i in ings)

# ─── CLI evaluation harness ────────────────────────────────────────
def main(args):
    df = pd.read_csv(args.ground_truth, index_col=None)

    # parse exactly as the web app does
    df['_ings'] = df['ingredients'].apply(to_list)

    # ── Pre-warm our preprocess cache ───────────────────────────
    unique_raws = {ing for cell in df['_ings'] for ing in cell}
    for raw in unique_raws:
        preprocess(raw)

    # ── run classification with plain list comprehensions ────────
    start = time()
    keto_preds  = [is_keto(cell)  for cell in df['_ings']]
    vegan_preds = [is_vegan(cell) for cell in df['_ings']]
    end = time()

    print('=== Keto ===')
    print(classification_report(df['keto'], keto_preds))
    print(confusion_matrix(df['keto'], keto_preds), '\n')

    print('=== Vegan ===')
    print(classification_report(df['vegan'], vegan_preds))
    print(confusion_matrix(df['vegan'], vegan_preds), '\n')

    print(f'== Time: {end-start:.2f}s ==')
    return 0

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--ground_truth',
        default='/usr/src/data/ground_truth_sample.csv',
        help="CSV with ['ingredients','keto','vegan']"
    )
    sys.exit(main(parser.parse_args()))
