#!/usr/bin/env python
"""
relabel_flagged.py
──────────────────
Send ONLY the rows in `clean_ingredients_flagged.csv`
to GPT-3.5-Turbo and save the new labels to
`clean_ingredients_flagged_labeled.csv`.

The flagged file must have at least one column called  ingredient
(other columns are ignored).

Safe to interrupt and resume – completed rows are skipped automatically.
"""

from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import openai

# ──────────────── CONFIG ─────────────────
SRC_FILE   = "clean_ingredients_flagged.csv"
OUT_FILE   = "clean_ingredients_flagged_labeled.csv"
MODEL      = "gpt-3.5-turbo-0125"
BATCH_SIZE = 5                # five ingredients per call
# ─────────────────────────────────────────

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("❌  Please export OPENAI_API_KEY first.")

client = openai.OpenAI()

# 1.  Load flagged rows
flagged = pd.read_csv(SRC_FILE, dtype=str)
flagged["ingredient"] = flagged["ingredient"].fillna("").str.strip()

# 2.  Resume-aware skip list
done = set()
if Path(OUT_FILE).exists() and Path(OUT_FILE).stat().st_size:
    done_df = pd.read_csv(OUT_FILE, dtype=str)
    done = set(done_df["ingredient"])
    print(f"✔  {len(done)} rows already labeled – resuming…")

todo = flagged[~flagged["ingredient"].isin(done)]["ingredient"].tolist()
if not todo:
    print("✓ Nothing left to do.")
    sys.exit()

print(f"🚀  Still need GPT for {len(todo)} ingredients")

SYSTEM_MSG = "You are a concise nutrition-labeling assistant."

def prompt(batch: list[str]) -> str:
    items = "\n".join(f"{i+1}. {x}" for i, x in enumerate(batch))
    return f"""\
For each item decide:
• vegan — 1 = vegan-safe, 0 = not vegan
• keto  — 1 = ≤5 g net-carb per 100 g **or** used only in trace spice amounts, else 0
Output ONLY csv. Columns: ingredient,vegan,keto
{items}
"""

def call_gpt(batch: list[str]) -> pd.DataFrame:
    msg = [{"role":"system","content":SYSTEM_MSG},
           {"role":"user",  "content":prompt(batch)}]
    for attempt in range(3):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=msg,
                temperature=0
            )
            csv_text = res.choices[0].message.content.strip()
            df = pd.read_csv(pd.compat.StringIO(csv_text), dtype=str)
            if set(df.columns) != {"ingredient","vegan","keto"}:
                raise ValueError("Bad columns")
            if len(df) != len(batch):
                raise ValueError("Row count mismatch")
            return df
        except Exception as e:
            wait = 2**attempt
            print(f"⚠️  GPT error ({e}) – retrying in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    sys.exit("GPT failed three times – aborting")

# 3.  Process
columns = ["ingredient","vegan","keto"]
Path(OUT_FILE).touch()
with open(OUT_FILE, "a") as sink:
    if sink.tell() == 0:
        sink.write(",".join(columns)+"\n")   # header once

    for i in tqdm(range(0, len(todo), BATCH_SIZE), unit="batch"):
        batch = todo[i:i+BATCH_SIZE]
        out_df = call_gpt(batch)
        out_df.to_csv(sink, header=False, index=False)

print(f"✓ All done – new labels written to {OUT_FILE}")
