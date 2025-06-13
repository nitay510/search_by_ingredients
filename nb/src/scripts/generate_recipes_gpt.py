import os
import json
import time
import random
from pathlib import Path

import openai

# Load your API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# How many recipes to generate per label
RECIPES_PER_LABEL = 300
LABELS = ["vegan", "keto", "regular"]
OUT_PATH = Path("gpt_recipes.json")


def generate_prompt(label: str) -> str:
    return f"""Create a unique {label} recipe.
Return a JSON with:
- title: short name of the dish
- ingredients: a list of 5-15 ingredients (only the raw ingredient phrases, no steps or instructions)
Only output the JSON.
"""


def call_gpt(label: str) -> dict | None:
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a recipe generator."},
                {"role": "user", "content": generate_prompt(label)},
            ],
            temperature=0.8,
        )
        content = response.choices[0].message.content
        recipe = json.loads(content)
        recipe["label"] = label
        return recipe
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    all_recipes = []
    seen_titles = set()

    for label in LABELS:
        count = 0
        while count < RECIPES_PER_LABEL:
            print(f"[{label}] Generating recipe {count + 1}/{RECIPES_PER_LABEL}...")
            recipe = call_gpt(label)

            if recipe and isinstance(recipe.get("ingredients"), list):
                title = recipe["title"].strip().lower()
                if title not in seen_titles:
                    all_recipes.append(recipe)
                    seen_titles.add(title)
                    count += 1
                else:
                    print(f"Duplicate title skipped: {title}")
            else:
                print("Bad format, skipping...")

            time.sleep(random.uniform(1.5, 2.5))  # avoid rate limit

    with open(OUT_PATH, "w") as f:
        json.dump(all_recipes, f, indent=2)

    print(f"\n✅ Saved {len(all_recipes)} recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
