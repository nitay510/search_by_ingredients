#!/usr/bin/env python3
"""
label_with_gpt.py  –  version for openai-python ≥ 1.0
"""

import csv, json, os, sys, time, pathlib, logging
from collections import OrderedDict
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI          # ← NEW import

# ---------- CONFIG ----------------------------------------------------------

CSV_IN              = "clean_ingredients.csv"
CSV_OUT             = "gpt_labeled.csv"
CHECKPOINT_FILE     = "checkpoint.json"
MODEL               = "gpt-3.5-turbo"
TEMPERATURE         = 0.2
BATCH_SIZE          = 5
MAX_RETRIES         = 3
SLEEP_BETWEEN_CALLS = 0.7          # seconds

# ---------- LOGGING ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler("label_with_gpt.log"),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ---------- PROMPT ----------------------------------------------------------

SYSTEM_MSG = """\
You are a culinary data assistant.

TASK 1 – For EACH valid food-ingredient phrase I give you, output
  { "ingredient": <string>, "vegan": 1/0, "keto": 1/0 }
Omit any item that is not an ingredient.

TASK 2 – Invent FIVE more, similar ingredients and label them.

Return one JSON **array only** – no markdown, no commentary.
"""

def pack_user(batch: List[str]) -> str:
    joined = "\n".join(f"- {w}" for w in batch)
    return f"Here are {len(batch)} candidate ingredient names:\n{joined}"

# ---------- FILE HELPERS ----------------------------------------------------

def read_column(csv_path: str, col: str) -> List[str]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        return [row[col].strip() for row in rdr if row[col].strip()]

def append_rows(csv_path: str, rows: List[Dict[str, str]]) -> None:
    exists = pathlib.Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ingredient", "vegan", "keto", "source"])
        if not exists:
            w.writeheader()
        w.writerows(rows)

def save_ckpt(done: int, seen: set):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
        json.dump({"done": done, "seen": list(seen)}, fh)

def load_ckpt():
    if not pathlib.Path(CHECKPOINT_FILE).exists():
        return 0, set()
    with open(CHECKPOINT_FILE, encoding="utf-8") as fh:
        d = json.load(fh)
        return d.get("done", 0), set(d.get("seen", []))

# ---------- GPT WRAPPER -----------------------------------------------------

def call_chat(client: OpenAI, batch: List[str]) -> List[Dict]:
    user_msg = pack_user(batch)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model       = MODEL,
                temperature = TEMPERATURE,
                messages = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": user_msg}
                ]
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)

            if (isinstance(data, list)
                    and all(set(x) == {"ingredient", "vegan", "keto"} for x in data)):
                return data

            raise ValueError("Unexpected JSON schema")

        except Exception as e:
            log.warning("Batch failed (attempt %d/%d): %s", attempt, MAX_RETRIES, e)
            time.sleep(1.5 * attempt)

    raise RuntimeError("GPT call failed 3× in a row")

# ---------- MAIN LOOP -------------------------------------------------------

def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)   # ← NEW client object

    ingredients = read_column(CSV_IN, "ingredient")
    done, seen  = load_ckpt()
    remaining   = [w for w in OrderedDict.fromkeys(ingredients) if w not in seen]

    if not remaining:
        log.info("All ingredients already processed.")
        return

    log.info("Starting… %d of %d ingredients left", len(remaining), len(ingredients))

    pbar = tqdm(total=len(remaining), desc="Batches", unit="ing")

    try:
        while remaining:
            batch, remaining = remaining[:BATCH_SIZE], remaining[BATCH_SIZE:]

            gpt_rows = call_chat(client, batch)

            out = []
            for row in gpt_rows:
                ing = row["ingredient"].strip()
                if ing in seen:
                    continue
                seen.add(ing)
                out.append({
                    "ingredient": ing,
                    "vegan": int(row["vegan"]),
                    "keto":  int(row["keto"]),
                    "source": "original" if ing in batch else "gpt_extra"
                })

            append_rows(CSV_OUT, out)
            done += len(batch)
            save_ckpt(done, seen)

            pbar.update(len(batch))
            time.sleep(SLEEP_BETWEEN_CALLS)

    except KeyboardInterrupt:
        log.warning("Interrupted by user – progress saved.")
    finally:
        save_ckpt(done, seen)
        pbar.close()
        log.info("✅  Finished – processed %d original ingredients, %d unique total.",
                 done, len(seen))

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
