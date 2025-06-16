#!/usr/bin/env python3
"""
Clean + merge two ingredient files and auto-label VEGAN / KETO flags.

Usage
-----
    python merge_and_label.py          # use BOTH csvs
    python merge_and_label.py --skip-candidates    # only the clean 5 500 list
"""
import re, argparse, pandas as pd, pathlib, csv

# ------------------------- command-line switches
argp = argparse.ArgumentParser()
argp.add_argument("--skip-candidates", action="store_true",
                  help="ignore candidates_raw.csv entirely")
args = argp.parse_args()

# ------------------------- helpers
HERE = pathlib.Path(__file__).absolute().parent
def load_words(fname):
    return set(w.strip() for w in open(HERE / "rules" / fname, encoding="utf-8"))

ANIMAL  = load_words("animal_words.txt")
HI_CARB = load_words("high_carb_words.txt")

def tidy(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^\w\s%-]", " ", txt)     # keep letters, digits, % - _
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def vegan(word: str) -> int:
    return 0 if any(a in word for a in ANIMAL) else 1

def keto(word: str) -> int:
    if any(a in word for a in ANIMAL):
        return 1
    if any(c in word for c in HI_CARB):
        return 0
    if re.search(r"(oil|butter|almond|walnut|pecan|avocado|hazelnut|fat)", word):
        return 1
    return 0

def read_csv(path) -> pd.Series:
    """Return a Series of raw strings from first column called *ingredient* or index 0"""
    df = pd.read_csv(path)
    col = "ingredient" if "ingredient" in df.columns else df.columns[0]
    return df[col]

# ------------------------- load HIGH-quality list
clean = read_csv("clean_ingredients.csv").apply(tidy)
print(f"[info] clean_master.csv   : {len(clean)} rows")

frames = [pd.DataFrame({"ingredient": clean, "quality": "high"})]

# ------------------------- optionally load candidates
if not args.skip_candidates:
    cand_raw = read_csv("candidates_raw.csv")
    # quick filter: drop rows shorter than 3 chars or containing {error,duplicate,exxx,…}
    cand = (
        cand_raw[~cand_raw.str.contains(r"\berror|\bduplicate|\bexxx|\d{5,}", case=False, na=False)]
        .apply(tidy)
        .pipe(lambda s: s[s.str.len() >= 3])
        .drop_duplicates()
    )
    print(f"[info] candidates_raw.csv: {len(cand)} rows after auto-scrub")
    frames.append(pd.DataFrame({"ingredient": cand, "quality": "medium"}))

# ------------------------- merge, dedupe, label
all_ing = (
    pd.concat(frames, ignore_index=True)
      .drop_duplicates(subset="ingredient")
      .reset_index(drop=True)
)
all_ing["vegan"] = all_ing["ingredient"].apply(vegan)
all_ing["keto"]  = all_ing["ingredient"].apply(keto)
all_ing["notes"] = ""     # room for manual tweaks

out = "ingredients_final.csv"
all_ing.to_csv(out, index=False)
print(f"[✓] wrote {out}  ({len(all_ing)} unique rows)")
print(all_ing.head(10).to_markdown(index=False))   # sneak peek
