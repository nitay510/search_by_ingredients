from __future__ import annotations
import os, sys, time, io, re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import openai

# ─────────── CONFIG ────────────
SRC_FILE   = "gpt_labeled.csv"
OUT_FILE   = "ingredients_cleaned.csv"
MODEL      = "gpt-4o"
BATCH_SIZE = 50                 # send 50 numbered lines each call
SYSTEM_MSG = "You are a culinary data-cleaning assistant."
# ───────────────────────────────

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("❌  Please export OPENAI_API_KEY first.")

client = openai.OpenAI()

# 1️⃣  Load source file
src = pd.read_csv(SRC_FILE, dtype=str)
if "ingredient" not in src.columns:
    sys.exit("❌  'ingredient' column missing in source file")

src["ingredient"] = src["ingredient"].fillna("").str.strip()

# 2️⃣  Determine which rows are already done
done_idx: set[int] = set()
if Path(OUT_FILE).is_file() and Path(OUT_FILE).stat().st_size:
    done_df = pd.read_csv(OUT_FILE, dtype=str)
    done_idx = set(done_df["orig_index"].astype(int))
    print(f"✔  {len(done_idx)} rows already cleaned – resuming…")

todo_idx = [i for i in range(len(src)) if i not in done_idx]
if not todo_idx:
    print("✓ Nothing left to do.")
    sys.exit()

print(f"🚀  Sending {len(todo_idx)} rows to GPT in batches of {BATCH_SIZE}")

# ---------- helpers --------------------------------------------------


def build_prompt(batch: list[str]) -> str:
    """Number each line starting at 1 and build the prompt text"""
    numbered = "\n".join(f"{n}. {txt}" for n, txt in enumerate(batch, 1))
    return (
        "Below is a numbered list of texts. "
        "For each line decide whether it is a food ingredient.\n"
        "• If YES: output exactly the same text (unchanged).\n"
        "• If NO: replace it with a plausible food ingredient.\n\n"
        "Return **CSV only** with TWO columns: id,ingredient. "
        "`id` is the original number. Do not add extra columns.\n\n"
        + numbered
    )


def call_gpt(batch: list[str]) -> list[str]:
    """Send one batch, parse reply, return cleaned list (same length)"""
    msg = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": build_prompt(batch)},
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=msg,
            ).choices[0].message.content

            # strip markdown fences if any
            resp = re.sub(r"^```.*?\n|```$", "", resp, flags=re.S).strip()
            df   = pd.read_csv(io.StringIO(resp), dtype=str)

            df.columns = [c.strip().lower() for c in df.columns]
            if {"id", "ingredient"} - set(df.columns):
                raise ValueError("Missing 'id' or 'ingredient' column")

            df["id"] = pd.to_numeric(df["id"], errors="coerce") - 1  # 0-based
            cleaned  = [""] * len(batch)

            for row in df.itertuples(index=False):
                idx = int(row.id)
                if 0 <= idx < len(batch):
                    cleaned[idx] = row.ingredient.strip()

            # any slot GPT skipped → keep original text
            for i, txt in enumerate(cleaned):
                if not txt:
                    cleaned[i] = batch[i]

            return cleaned

        except Exception as e:
            wait = 2 ** attempt
            print(f"⚠️ GPT error ({e}) – retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError("GPT failed 3× on one batch")


# 3️⃣  Open output file once (append mode)
header_written = Path(OUT_FILE).is_file() and Path(OUT_FILE).stat().st_size
with open(OUT_FILE, "a", newline="") as sink:
    if not header_written:
        sink.write("orig_index,ingredient\n")

    # process batches in order
    for start in tqdm(range(0, len(todo_idx), BATCH_SIZE), unit="batch"):
        batch_idx   = todo_idx[start : start + BATCH_SIZE]
        batch_lines = src.loc[batch_idx, "ingredient"].tolist()

        cleaned = call_gpt(batch_lines)

        for idx, txt in zip(batch_idx, cleaned):
            sink.write(f"{idx},{txt}\n")

print(f"✅  Finished – cleaned list saved to {OUT_FILE}")