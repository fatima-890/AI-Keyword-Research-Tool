"""
expand_keywords.py
-------------------
Expands the existing small keyword dataset into a much larger one
by combining each seed keyword with common SEO modifiers.
Run from project root: python expand_keywords.py
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "keywords.csv")   # your original file
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "keywords_expanded.csv")

MODIFIERS = [
    "best", "how to", "free", "for beginners", "tools", "2026", "vs",
    "guide", "tips", "near me", "price", "software", "course", "examples",
    "app", "review", "checklist", "template", "ideas", "strategy"
]

def main():
    df = pd.read_csv(INPUT_PATH)
    keyword_col = df.columns[1]  # adjust if needed — check your actual column name
    seeds = df[keyword_col].dropna().unique()

    rng = np.random.default_rng(42)
    rows = []
    for seed in seeds:
        for mod in MODIFIERS:
            kw = f"{mod} {seed}" if rng.random() > 0.5 else f"{seed} {mod}"
            vol = int(rng.lognormal(mean=6, sigma=1.5))
            cpc = round(rng.uniform(0.2, 9.0), 2)
            comp = round(rng.uniform(0, 1), 2)
            kd = round(rng.uniform(5, 95), 1)
            rows.append([kw, vol, cpc, comp, kd])

    expanded = pd.DataFrame(rows, columns=["Keyword", "Search Volume", "CPC", "Competition", "Keyword Difficulty"])
    expanded = pd.concat([expanded, df.rename(columns={keyword_col: "Keyword"})], ignore_index=True)
    expanded = expanded.drop_duplicates(subset="Keyword")
    expanded.to_csv(OUTPUT_PATH, index=False)
    print(f"Expanded dataset: {len(expanded)} keywords -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()