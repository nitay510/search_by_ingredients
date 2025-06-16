#!/usr/bin/env python
"""
label_vegan_json.py
───────────────────
Label each ingredient VEGAN / NOT-vegan and save a short reason.

INPUT  : ingredients_cleaned.csv   (orig_index, ingredient)
OUTPUT : ingredients_vegan.csv     (orig_index, ingredient, vegan, reason)

• Robust JSON parsing – no “Unexpected CSV shape”.
• Tiny batches (default 10).
• Interrupt- & resume-safe.
"""

from __future__ import annotations
import os, sys, time, json, textwrap
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import openai

# ─────────────── CONFIG ───────────────
SRC_FILE    = "ingredients_cleaned.csv"
OUT_FILE    = "ingredients_vegan.csv"
MODEL       = "gpt-4o"            # or any model you prefer
BATCH_SIZE  = 10                  # tiny batches
MAX_RETRIES = 3
# ──────────────────────────────────────

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("❌  Please export OPENAI_API_KEY first.")

client = openai.OpenAI()

# 1 ── load source
src = pd.read_csv(SRC_FILE, dtype=str)
src["ingredient"] = src["ingredient"].fillna("").str.strip()

# 2 ── resume awareness
done_idx: set[int] = set()
if Path(OUT_FILE).is_file() and Path(OUT_FILE).stat().st_size:
    done_df = pd.read_csv(OUT_FILE, dtype=str)
    if {"orig_index", "vegan", "reason"} <= set(done_df.columns):
        done_idx = set(done_df["orig_index"].astype(int))
        print(f"✔  {len(done_idx)} already labeled – resuming…")

todo_df = src[~src.index.isin(done_idx)]
if todo_df.empty:
    print("✓ Everything already labeled.")
    sys.exit()

print(f"🚀  Need to label {len(todo_df)} rows in batches of {BATCH_SIZE}")

# 3 ── GPT helpers
SYSTEM_MSG = "You are a concise nutrition assistant."

def make_prompt(batch: pd.Series) -> str:
    items = "\n".join(f"- {ing}" for ing in batch)
    return textwrap.dedent(f"""
        For each ingredient decide:
        • vegan : 1 = 100 % plant-based (no meat, fish, dairy, eggs, honey, gelatin, etc.),
                  0 = any animal-derived ingredient
        • reason : SUPER-SHORT explanation (a few words)

        Reply with **VALID JSON ONLY** – an array of objects
        having exactly the keys: ingredient, vegan, reason

        Example:
        [
          {{"ingredient":"almond flour","vegan":1,"reason":"plant-based nut flour"}},
          ...
        ]

        Ingredients:
        {items}
    """).strip()

import re, json

def parse_json(raw: str) -> list[dict]:
    """
    Find the first [...] block in `raw` and json-load it.
    Strips ``` fences and any chatty text.
    Raises ValueError if nothing that looks like JSON is found.
    """
    raw = raw.strip()

    # remove ``` fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        # keep text inside the first fence block
        raw = parts[1] if len(parts) > 2 else "\n".join(parts[1:])

    # locate the first JSON-looking chunk
    match = re.search(r"\[[\s\S]*?\]", raw)
    if not match:
        raise ValueError("no JSON array found in reply")

    return json.loads(match.group(0))


def single_fallback(ing: str) -> dict:
    prompt = make_prompt(pd.Series([ing]))
    msgs = [{"role":"system","content":SYSTEM_MSG},
            {"role":"user","content":prompt}]
    for _ in range(MAX_RETRIES):
        try:
            txt = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0
            ).choices[0].message.content
            rec = parse_json(txt)[0]
            if {"ingredient","vegan","reason"} <= rec.keys():
                return rec
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"GPT failed on single item: {ing}")

def ask_gpt(batch: pd.Series) -> list[dict]:
    prompt = make_prompt(batch)
    msgs = [{"role":"system","content":SYSTEM_MSG},
            {"role":"user","content":prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            txt = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0
            ).choices[0].message.content
            data = parse_json(txt)
            if (isinstance(data, list) and
                all({"ingredient","vegan","reason"} <= d.keys() for d in data)):
                return data
            raise ValueError("Bad JSON shape")
        except Exception as e:
            wait = 2**attempt
            print(f"⚠️ GPT error ({e}) – retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("GPT failed 3× for batch")

# 4 ── run
header_exists = Path(OUT_FILE).is_file() and Path(OUT_FILE).stat().st_size > 0
with open(OUT_FILE, "a", newline="", encoding="utf-8") as sink:
    if not header_exists:
        sink.write("orig_index,ingredient,vegan,reason\n")

    for start in tqdm(range(0, len(todo_df), BATCH_SIZE), unit="batch"):
        batch_df  = todo_df.iloc[start:start+BATCH_SIZE]
        batch_ing = batch_df["ingredient"]

        # main GPT call
        try:
            records = ask_gpt(batch_ing)
        except RuntimeError:
            # fall back to single-item calls
            records = [single_fallback(ing) for ing in batch_ing]

        rec_df = pd.DataFrame(records)
        merged = (
            batch_df.reset_index()[["index", "ingredient"]]
            .merge(rec_df, on="ingredient", how="left")
        )

        # handle any stragglers
        missing = merged[merged["vegan"].isna()]["ingredient"].tolist()
        if missing:
            extra = [single_fallback(x) for x in missing]
            merged.update(pd.DataFrame(extra))

        merged.rename(columns={"index": "orig_index"}, inplace=True)
        merged.to_csv(sink, header=False, index=False)

print(f"✓ Vegan labels written to {OUT_FILE}")
