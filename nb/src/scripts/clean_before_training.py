#!/usr/bin/env python3
"""
prep_for_training.py
────────────────────
Create *deduplicated, noise-reduced* CSVs for two separate
classification tasks: keto and vegan.

INPUTS  : ingredients_keto_fixed.csv   (orig_index, ingredient, keto,  reason)
          ingredients_vegan_fixed.csv  (orig_index, ingredient, vegan, reason)

OUTPUTS : train_keto.csv   (ingredient, keto)
          train_vegan.csv  (ingredient, vegan)
"""

from pathlib import Path
import re
import pandas as pd

KETO_SRC  = Path("ingredients_keto.csv")
VEGAN_SRC = Path("ingredients_vegan.csv")

# ---------- helper: text normaliser ----------
_spaces = re.compile(r"\s+")
def clean(text: str) -> str:
    """lower-case, strip, collapse internal whitespace"""
    return _spaces.sub(" ", str(text).strip().lower())
# ---------------------------------------------

# load
keto  = pd.read_csv(KETO_SRC , dtype=str)
vegan = pd.read_csv(VEGAN_SRC, dtype=str)

# normalise & de-dup
keto["ingredient"]  = keto["ingredient"].map(clean)
vegan["ingredient"] = vegan["ingredient"].map(clean)

keto  = keto.drop_duplicates("ingredient", keep="first")
vegan = vegan.drop_duplicates("ingredient", keep="first")

# keep only task columns
keto_out  = keto[["ingredient", "keto"]].copy()
vegan_out = vegan[["ingredient", "vegan"]].copy()

# write
keto_out .to_csv("train_keto.csv" ,  index=False)
vegan_out.to_csv("train_vegan.csv", index=False)

print(f"✔  keto rows : {len(keto_out)}")
print(f"✔  vegan rows: {len(vegan_out)}")
