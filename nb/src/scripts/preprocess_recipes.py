#!/usr/bin/env python3
"""
  $ python clean_ingredients_v2.py --src recipes.csv --col Ingredients \
                                   --max-words 3 --debug
"""
from __future__ import annotations
import argparse, re, unicodedata, string, sys
from pathlib import Path
import pandas as pd, spacy

# ────────────────────────────────────────────────────────────────────
# 1.  ultra-fast regex scrub
# ────────────────────────────────────────────────────────────────────
FRACTIONS = "¼½¾⅓⅔⅛⅜⅝⅞"
NUMBERS   = r"\d+(?:\.\d+)?(?:/\d+)?"
UNITS     = r"""
    cups?|c|tbsp|tablespoons?|tbs|tsp|teaspoons?|lbs?|pounds?|oz|ounces?|
    grams?|g|kg|kilograms?|ml|l|liters?|litres?|pints?|pt|quarts?|qt|gal|gallons?
"""
EQUIP  = r"""
    bowl|skillet|pan|pot|sheet|tray|knife|foil|jar|glass|loaf|bundt|
    springform|gratin|processor|ricer|thermometer|spatula|stone|rack|
    slicer|tongs?|bag|box|package|pouch|cooker|blender|torch
"""
# words & patterns that never belong to an ingredient
FILLER = r"""
    (?:at\s+)?room\s+temperature|additional|accompaniments?|optional|about|roughly|
    plus|divided|for\s+serving|serve|servings?|to\s+taste|needed?|prepared|
    large|small|medium|extra|jumbo|mini|fresh|freshly|frozen|raw|cooked|dry|dried|
    skin(?:less|-on)?|bone(?:less|-in)?|lean|whole|halves?|quarters?|trimmed|
    chopped|minced|sliced|diced|ground|peeled|seeded|pitted|grated|shredded|
    julienned|matchsticks?|strips?|chunks?|cubes?|pieces?|wedges?|bias|crosswise|
    lengthwise|diagonal|inch(?:es|long|wide|thick|-thick|diameter|square)?|cm|mm|
    colour|colored|ripe|unripe|cold|warm|hot
"""

RE_NUM   = re.compile(rf"(?:{NUMBERS}|[{FRACTIONS}])")
RE_UNIT  = re.compile(rf"\b(?:{UNITS})\b",   re.I|re.X)
RE_EQUIP = re.compile(rf"\b(?:{EQUIP})\b",   re.I|re.X)
RE_FILL  = re.compile(rf"(?:{FILLER})",      re.I|re.X)
PUNCT_TR = str.maketrans("", "", string.punctuation)  # delete punctuation
CELL_SPLIT = re.compile(r"'([^']+)'")                # split "['a','b']"

def fast_scrub(txt: str) -> str:
    """Single rapid regex pass – no spaCy yet."""
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = txt.lower()
    txt = re.sub(r"\([^)]*\)", " ", txt)          # kill (...) comments
    txt = RE_NUM.sub(" ", txt)
    txt = RE_UNIT.sub(" ", txt)
    txt = RE_FILL.sub(" ", txt)
    txt = txt.translate(PUNCT_TR)
    txt = re.sub(r"\s+", " ", txt).strip()
    # filter out equipment lines that occasionally survive (e.g. “mixing bowl”)
    return "" if (not txt or RE_EQUIP.search(txt)) else txt

# ────────────────────────────────────────────────────────────────────
# 2.  spaCy – keep *runs* of NOUN / PROPN (incl. multi-word)
# ────────────────────────────────────────────────────────────────────
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
KEEP   = {"NOUN", "PROPN"}
STOP_NOUNS = {"package", "packages", "equipment", "recipe"}  # easy to extend

def spacy_pass(phrases: list[str], max_words: int) -> list[str]:
    out, seen = [], set()
    for doc in NLP.pipe(phrases, batch_size=512):
        buff = []
        for tok in doc:
            if tok.pos_ in KEEP and tok.lemma_ not in STOP_NOUNS:
                buff.append(tok.lemma_)
            else:
                if 0 < len(buff) <= max_words:
                    key = " ".join(buff)
                    if key not in seen:
                        out.append(key);  seen.add(key)
                buff = []
        # flush tail
        if 0 < len(buff) <= max_words:
            key = " ".join(buff)
            if key not in seen:
                out.append(key);  seen.add(key)
    return out

# ────────────────────────────────────────────────────────────────────
def each_cell(series: pd.Series):
    """Yield every raw ingredient string from the source column."""
    for cell in series:
        if not isinstance(cell, str):
            continue
        if cell.startswith("[") and "'" in cell:          # list-style cells
            yield from CELL_SPLIT.findall(cell)
        else:
            yield from re.split(r"[\n,]+", cell)

def main(src: Path, col: str, out: Path, max_words: int, debug: bool):
    df = pd.read_csv(src)
    coarse = {c for raw in each_cell(df[col]) if (c := fast_scrub(raw))}
    if debug:
        print(f"⚙️  after regex pass: {len(coarse):,} unique strings", file=sys.stderr)

    fine = spacy_pass(sorted(coarse), max_words)
    pd.DataFrame(fine, columns=["ingredient"]).to_csv(out, index=False)
    print(f"✅ {len(fine):,} clean ingredients (≤{max_words} words) → {out}")

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",       default="Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
    ap.add_argument("--col",       default="Ingredients")
    ap.add_argument("--out",       default="clean_ingredients_v2.csv")
    ap.add_argument("--max-words", type=int, default=3, help="keep phrases of ≤ N tokens")
    ap.add_argument("--debug",     action="store_true")
    args = ap.parse_args()

    main(Path(args.src), args.col, Path(args.out), args.max_words, args.debug)
