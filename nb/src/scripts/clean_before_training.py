#!/usr/bin/env python
"""
Clean the ingredient CSV in preparation for model-training.

• removes obvious non-food rows (pan, blade, inches, etc.)
• normalises text (ASCII, lower-case, single-spaced, singular form)
• drops duplicates
• quick sanity flags for vegan / keto
• writes:
    clean_ingredients_full.csv      – everything we keep
    clean_ingredients_flagged.csv   – rows where auto vegan/keto ≠ original
"""
import re
import unicodedata
import inflect
import pathlib
import pandas as pd

RAW_CSV = "gpt_labeled.csv"        # <-- change if your file is named differently
OUT_DIR = pathlib.Path(".")

##############################################################################
# 1.  read
##############################################################################
df = pd.read_csv(RAW_CSV)

##############################################################################
# 2.  filter out non-edible/tool words (non-capturing groups so pandas is happy)
##############################################################################
NON_FOOD = re.compile(
    r"\b(?:"
    r"blade|adjustableblade|attachment|pan|wok|mold|mould|styrofoam|plank|rack|"
    r"plate|bowl|ramekin|pans?|spoon|skewer|strainer|foil|wrap|liner|"
    r"wire|sheet|tray|pot|kilogram|gram|liter|minutes?|hours?|seconds?|"
    r"inch(?:es)?|centimeters?|milliliters?|class coupe|crafts? pencil|"
    r"notes (?:roast|salt|sausage)|batches icing|transfer letters"
    r")\b",
    flags=re.I,
)
df = df.loc[~df["ingredient"].str.contains(NON_FOOD, regex=True, na=False)]

##############################################################################
# 3. normalise + dedupe
##############################################################################
inflector = inflect.engine()

def ascii_slug(text: str) -> str:
    """→ ascii lower-case words, single space."""
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # drop punctuation
    text = re.sub(r"\s+", " ", text).strip()
    # singular-ise every token
    tokens = [inflector.singular_noun(tok) or tok for tok in text.split()]
    return " ".join(tokens)

df["ingredient_norm"] = df["ingredient"].fillna("").astype(str).apply(ascii_slug)
df = df.drop_duplicates("ingredient_norm").reset_index(drop=True)

##############################################################################
# 4. vegan / keto heuristics
##############################################################################
VEGAN_BAD = re.compile(
    r"beef|pork|ham|bacon|lamb|veal|turkey|chicken|duck|goat|"
    r"fish|salmon|tuna|anchov|sardine|shellfish|shrimp|prawn|crab|lobster|"
    r"sausage|prosciutto|chorizo|gelatin\b|egg\b|honey\b|cheese|milk|butter|cream|yogurt",
    flags=re.I,
)
KETO_BAD = re.compile(
    r"sugar|syrup|honey|molasses|maple|agave|rice\b|corn\b|oats?\b|wheat|"
    r"barley|millet|quinoa|bread|pasta|noodle|flour|grain|bean|lentil|chickpea|pea\b|"
    r"potato|yam|cassava|taro|banana|apple|pear|orange|apricot|pineapple|mango|grape",
    flags=re.I,
)

df["vegan_auto"] = (~df["ingredient_norm"].str.contains(VEGAN_BAD)).astype(int)
df["keto_auto"]  = (~df["ingredient_norm"].str.contains(KETO_BAD )).astype(int)
flagged = df.loc[(df.vegan != df.vegan_auto) | (df.keto != df.keto_auto)]

##############################################################################
# 5. write
##############################################################################
OUT_DIR.mkdir(exist_ok=True, parents=True)
df.to_csv(OUT_DIR / "clean_ingredients_full.csv",   index=False)
flagged.to_csv(OUT_DIR / "clean_ingredients_flagged.csv", index=False)

print(
    f"✅ Cleaned: kept {len(df):,} unique ingredients • "
    f"flagged {len(flagged):,} for manual review"
)
