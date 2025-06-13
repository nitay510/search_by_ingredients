import json
import re
import pandas as pd
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

# Regex patterns for cleaning
UNIT_RE = re.compile(r"\b\d+[½¼¾⅓⅔/.\-]*\s*(?:cups?|tbsp|tablespoons?|tsp|teaspoons?|"
                     r"pounds?|lbs?|oz|ounces?|cloves?|slices?|pinch|heads?|cans?|sticks?|"
                     r"fluid\s+ounces?|grams?|g|kg|liters?|ml|cm)\b", flags=re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(?:[./]\d+)?\b")

# Load JSON file
with open("generated_recipes.json", "r") as f:
    recipes = json.load(f)

def clean_ingredient(ingredient: str) -> str | None:
    """Clean and lemmatize a single ingredient line."""
    ingredient = UNIT_RE.sub(" ", ingredient.lower())
    ingredient = NUM_RE.sub(" ", ingredient)
    ingredient = re.sub(r"[^\w\s]", " ", ingredient)
    ingredient = " ".join(ingredient.split())

    doc = nlp(ingredient)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "ADJ"}
    ]
    return " ".join(tokens).strip() if tokens else None

def preprocess_recipe(recipe):
    label = recipe.get("label", "").strip().lower()
    text = recipe.get("text", "")

    # Extract ingredients section
    match = re.search(r"Ingredients:\s*(.*?)\n\n(?:Preparation|Instructions):", text, re.DOTALL)
    raw_ingredients = match.group(1).strip() if match else ""
    raw_lines = [line.strip("- ").strip() for line in raw_ingredients.split("\n") if line.strip()]

    # Clean and filter
    cleaned_ingredients = [clean_ingredient(line) for line in raw_lines]
    cleaned_ingredients = [ing for ing in cleaned_ingredients if ing]

    return {"label": label, "ingredients": cleaned_ingredients}

# Apply preprocessing
processed = [preprocess_recipe(r) for r in recipes if "text" in r and "label" in r]

# Create DataFrame
df = pd.DataFrame(processed)

# Save to CSV
df.to_csv("preprocessed_recipes.csv", index=False)
print(f"Saved {len(df)} recipes to preprocessed_recipes.csv")
