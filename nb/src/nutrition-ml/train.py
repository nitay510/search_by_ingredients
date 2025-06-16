#!/usr/bin/env python
"""
fast_train.py
─────────────
Train two light-weight text models:

  ▸ keto_svm.joblib
  ▸ vegan_svm.joblib

Input  : data/keto.csv   (ingredient,keto)
         data/vegan.csv  (ingredient,vegan)
Output : saved models ready for inference.

Runtime on 4 600 rows  ≈ 10–20 s on a laptop.
"""

from pathlib import Path
import re, unicodedata, time, joblib, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

DATA_DIR = Path("data")
TASKS    = [("keto",  DATA_DIR / "keto.csv"),
            ("vegan", DATA_DIR / "vegan.csv")]

def normalise(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt.casefold())
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def load(csv: Path, label_col: str):
    df = (pd.read_csv(csv, dtype=str)
            .drop_duplicates("ingredient")       # de-dupe
            .dropna(subset=["ingredient", label_col]))
    df["ingredient"] = df["ingredient"].map(normalise)
    y  = (df[label_col]
            .str.replace(r"\.0$", "", regex=True)   # “0.0” → “0”
            .astype(int))
    return df["ingredient"], y

def train_one(csv: Path, label: str, dst: str):
    print(f"\n─── {label.upper()} ───")
    X, y = load(csv, label)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2,
                                  ngram_range=(1,2),
                                  sublinear_tf=True)),
        ("clf",  LinearSVC())
    ])

    grid = {"clf__C": [0.1, 1, 10]}

    gs = GridSearchCV(pipe, grid, cv=5,
                      n_jobs=-1, scoring="f1_macro")

    t0 = time.time()
    gs.fit(X_tr, y_tr)
    print(f"search  {time.time()-t0:,.1f}s  best C={gs.best_params_['clf__C']}")

    print(classification_report(y_te, gs.predict(X_te)))
    joblib.dump(gs.best_estimator_, dst)
    print("✓ saved →", dst)

if __name__ == "__main__":
    for label, csv in TASKS:
        train_one(csv, label, f"{label}_svm.joblib")
