#!/usr/bin/env python
"""
  $ python clean_ingredients.py --src recipes.csv --col Ingredients \
                                --max-words 2 --debug
"""
from __future__ import annotations
import argparse, re, unicodedata, string, sys
from pathlib import Path
import pandas as pd, spacy

# ------------------------------------------------------------------- #
# fast regex scrub                                                     #
# ------------------------------------------------------------------- #
FRACTIONS = "¼½¾⅓⅔⅛⅜⅝⅞"
UNITS = r"""cups?|tbsp|tbs|tablespoons?|tsp|teaspoons?|lbs?|pounds?|oz|ounces?|
            grams?|g|kg|ml|l|liters?|litres?|pints?|pt|quarts?|qt|gal|gallons?"""
EQUIP  = r"""bowl|skillet|pan|pot|sheet|tray|knife|foil|jar|glass|loaf|bundt|
             springform|gratin|processor|ricer|thermometer|spatula|stone|rack|
             slicer|tongs|bag|box|package|pouch"""

# words that do NOT belong to an ingredient – add/remove at will
FILLER = r"""
    additional|accompaniments?|optional|about|roughly|plus|divided|serve|serving|
    taste|needed?|room|temperature|whole|large|small|medium|extra|jumbo|mini|
    fresh|freshly|frozen|raw|cooked|dry|dried|lean|
    bias|crosswise|lengthwise|diagonal|julienned?|matchsticks?|sticks?|strips?|
    chunks?|cubes?|dice[ds]?|pieces?|slabs?|ribbons?|rounds?|wedges?|halves?|
    quarters?|tips?|tops?|ends?|cores?|ribs?|leaves?|leaf|sprigs?|stems?|stalks?|
    hearts?|spears?|florets?|buds?|flowers?|bulb|bulbs|skins?|pits?|seeds?|
    root|roots|
    inch(?:es|long|wide|thick|thickslice|thickslices|diameter|square)?|cm|mm|
    shredded|minced|chopped|sliced|ground|peeled|seeded|pitted|trimmed|
    style|-style|quality|bestquality|bestoffryer|crisply
"""

NUM_WORDS  = re.compile(rf"\b\w*[0-9{FRACTIONS}]+\w*\b")
UNIT_WORD  = re.compile(rf"\b({UNITS})\b",   re.I)
EQUIP_RE   = re.compile(rf"\b({EQUIP})\b",   re.I)
FILLER_RE  = re.compile(rf"\b({FILLER})\b",  re.I)
PUNCT_MAP  = str.maketrans("", "", string.punctuation)
CELL_SPLIT = re.compile(r"'([^']+)'")

def coarse_scrub(txt: str) -> str:
    """Very fast, regex-only scrub to get rid of obvious junk."""
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = txt.lower()
    txt = re.sub(r"\([^)]*\)", " ", txt)          # kill (...) comments
    txt = NUM_WORDS.sub(" ", txt)                 # strip 4, 2-3/4, etc.
    txt = UNIT_WORD.sub(" ", txt)                 # strip ‘cup’, ‘lbs’, …
    txt = FILLER_RE.sub(" ", txt)                 # strip filler words
    txt = txt.translate(PUNCT_MAP)                # drop all punctuation
    txt = re.sub(r"\s+", " ", txt).strip()
    return "" if (not txt or EQUIP_RE.search(txt)) else txt

# ------------------------------------------------------------------- #
# spaCy pass (POS filter + noun stop-list + intra-phrase de-dup)       #
# ------------------------------------------------------------------- #
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
KEEP_POS   = {"NOUN", "PROPN"}
STOP_NOUNS = {
    # not really ingredients – added after manual inspection
    "slicer","baster","fryer","cooker","recipe","equipment",
    "package","packages"
}

def spacy_filter(phrases: list[str]) -> list[str]:
    out: list[str] = []
    for doc in NLP.pipe(phrases, batch_size=256):
        toks: list[str] = []
        last = None
        for t in doc:
            if t.pos_ in KEEP_POS and t.lemma_ not in STOP_NOUNS:
                # de-dup adjacent identical tokens (almonds almonds → almonds)
                if t.text != last:
                    toks.append(t.text)
                    last = t.text
        if toks:
            # second de-dup pass across the *whole* phrase
            dedup = []
            for tok in toks:
                if tok not in dedup:
                    dedup.append(tok)
            out.append(" ".join(dedup))
    # global order-preserving uniq
    return list(dict.fromkeys(out))

# ------------------------------------------------------------------- #
def cell_iterator(series: pd.Series):
    """Yield every raw ingredient string from a DataFrame column."""
    for cell in series:
        if not isinstance(cell, str):
            continue
        if cell.startswith("[") and "'" in cell:      # e.g. "['a','b']"
            yield from CELL_SPLIT.findall(cell)
        else:
            yield from re.split(r"[\n,]+", cell)

def main(src: Path, col: str, out: Path, max_words: int, debug: bool):
    df = pd.read_csv(src)
    coarse = {c for raw in cell_iterator(df[col]) if (c := coarse_scrub(raw))}
    if debug:
        print(f"🛠  coarse uniques: {len(coarse):,}", file=sys.stderr)

    fine = spacy_filter(sorted(coarse))
    fine = {x for x in fine if len(x.split()) <= max_words}

    pd.DataFrame(sorted(fine), columns=["ingredient"]).to_csv(out, index=False)
    print(f"✅ {len(fine):,} ingredients (≤ {max_words} words) written → {out}")

# ------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  default="Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
    ap.add_argument("--col",  default="Ingredients")
    ap.add_argument("--out",  default="clean_ingredients.csv")
    ap.add_argument("--max-words", type=int, default=2,
                    help="keep ingredients whose token count ≤ this (default 2)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    main(Path(args.src), args.col, Path(args.out), args.max_words, args.debug)
