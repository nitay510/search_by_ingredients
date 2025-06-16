#!/usr/bin/env python
"""
label_keto_json.py
──────────────────
Label each ingredient as keto / not-keto and keep a SHORT reason.

INPUT  : ingredients_cleaned.csv   (orig_index, ingredient)
OUTPUT : ingredients_keto.csv      (orig_index, ingredient, keto, reason)

The script is:
• idempotent   – you can stop & restart any time;
• robust       – extracts the first JSON array even if GPT adds chatter
                 or ``` fences;
• cautious     – retries whole batches, then individual items.
"""

from __future__ import annotations
import os, sys, time, json, re, textwrap, io
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import openai

# ──────── CONFIG ────────
SRC_FILE    = "ingredients_cleaned.csv"
OUT_FILE    = "ingredients_keto.csv"
MODEL       = "gpt-4o"
BATCH_SIZE  = 10          # keep small for maximal reliability
MAX_RETRIES = 3
# ────────────────────────

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("❌  Please export OPENAI_API_KEY before running.")

client = openai.OpenAI()

# ──────── helpers ────────
def smart_json(raw: str) -> list[dict]:
    """
    Return the first JSON array in `raw`.

    • Strips ``` fences and any leading text.
    • Raises ValueError if no JSON array is found.
    """
    txt = raw.strip()
    if txt.startswith("```"):
        # take the largest chunk inside the fences
        txt = max(txt.split("```"), key=len).strip()

    match = re.search(r"\[[\s\S]*?]", txt)   # non-greedy up to first ']'
    if not match:
        raise ValueError("no JSON array found")
    return json.loads(match.group(0))

def prompt_for(batch: pd.Series) -> str:
    items = "\n".join(f"- {ing}" for ing in batch)
    return textwrap.dedent(f"""
        For each ingredient below decide:
        • keto : 1 = naturally ≤ 5 g net-carbs / 100 g **or** used only as a spice, otherwise 0
        • reason : a few words

        Reply with **ONLY** a JSON array like:
        [
          {{"ingredient":"almond flour","keto":1,"reason":"low-carb nut flour"}},
          …
        ]

        Ingredients:
        {items}
    """).strip()

def gpt_call(messages: list[dict]) -> str:
    return client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0
    ).choices[0].message.content

def label_one(ingredient: str) -> dict:
    """Fallback: ask GPT about a single ingredient."""
    for _ in range(MAX_RETRIES):
        try:
            txt = gpt_call([
                {"role":"system","content":"You are a concise nutrition assistant."},
                {"role":"user"  ,"content":prompt_for(pd.Series([ingredient]))}
            ])
            rec = smart_json(txt)[0]
            if {"ingredient","keto","reason"} <= rec.keys():
                return rec
        except Exception as e:
            time.sleep(1)
    raise RuntimeError(f"💥  could not label: {ingredient}")

def label_batch(batch: pd.Series) -> list[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            txt = gpt_call([
                {"role":"system","content":"You are a concise nutrition assistant."},
                {"role":"user"  ,"content":prompt_for(batch)}
            ])
            data = smart_json(txt)
            # basic shape check
            if all({"ingredient","keto","reason"} <= d.keys() for d in data):
                return data
            raise ValueError("bad keys")
        except Exception as e:
            wait = 2 ** attempt
            print(f"⚠️  GPT batch error ({e}) – retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("GPT failed 3× for a batch")

# ──────── main ────────
src = pd.read_csv(SRC_FILE, dtype=str).fillna("")
src["ingredient"] = src["ingredient"].str.strip()

# figure out what’s already done
done_idx: set[int] = set()
if Path(OUT_FILE).is_file() and Path(OUT_FILE).stat().st_size:
    done_df = pd.read_csv(OUT_FILE, dtype=str)
    if "orig_index" in done_df.columns:
        done_idx = set(done_df["orig_index"].astype(int))
        print(f"✔  {len(done_idx)} rows already labeled – resuming…")

todo_df = src[~src.index.isin(done_idx)]
total   = len(todo_df)
if total == 0:
    print("✓ Nothing left to do.")
    sys.exit()

print(f"🚀  Need to label {total} rows (batches of {BATCH_SIZE})")

header_written = Path(OUT_FILE).is_file() and Path(OUT_FILE).stat().st_size
with open(OUT_FILE, "a", newline="") as sink:
    if not header_written:
        sink.write("orig_index,ingredient,keto,reason\n")

    for start in tqdm(range(0, total, BATCH_SIZE), unit="batch"):
        batch_df  = todo_df.iloc[start:start+BATCH_SIZE]
        batch_ing = batch_df["ingredient"]

        try:
            records = label_batch(batch_ing)
        except RuntimeError:
            # fall back to single-item calls
            records = [label_one(ing) for ing in batch_ing]

        rec_df = pd.DataFrame(records)
        merged = (
            batch_df.reset_index()           # keep original idx
                    .rename(columns={"index":"orig_index"})
                    .merge(rec_df, on="ingredient", how="left")
        )

        # any missing? resolve one-by-one
        for miss in merged[merged["keto"].isna()]["ingredient"]:
            merged.loc[merged["ingredient"] == miss, ["keto","reason"]] = \
                pd.Series(label_one(miss))[["keto","reason"]].values

        merged.to_csv(sink, header=False, index=False)

print(f"✓ Keto labels written to {OUT_FILE}")
