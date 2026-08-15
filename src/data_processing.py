"""
data_processing.py
-------------------
Loads the Kaggle keyword dataset, auto-detects column names (since
Kaggle SEO datasets don't all use the same headers), cleans the data,
and engineers features used by the ML models in train_model.py.

Works with any keyword dataset that has (at minimum) a keyword column.
Search volume / CPC / competition / difficulty columns are optional —
if missing, they are synthetically estimated so the pipeline never breaks.
"""

import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 1. COLUMN AUTO-DETECTION
# ---------------------------------------------------------------
COLUMN_ALIASES = {
    "keyword": ["keyword", "keywords", "query", "search term", "term", "phrase", "text"],
    "volume": ["volume", "search volume", "avg. monthly searches", "monthly searches",
               "search_volume", "avg_monthly_searches", "traffic", "vol", "v"],
    "cpc": ["cpc", "cost per click", "cost_per_click", "avg cpc"],
    "competition": ["competition", "comp", "competition_index", "competition index"],
    "difficulty": ["difficulty", "kd", "keyword difficulty", "seo difficulty",
                    "keyword_difficulty", "difficulty score", "score"],
    "intent": ["intent", "search intent", "search_intent", "main_intent"],
}


def _normalize(col: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", col.lower()).strip()


def detect_columns(df: pd.DataFrame) -> dict:
    """Map our internal field names -> actual column names found in df."""
    normalized_cols = {_normalize(c): c for c in df.columns}
    detected = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_cols:
                detected[field] = normalized_cols[alias]
                break
    return detected


# ---------------------------------------------------------------
# 2. LOAD + CLEAN
# ---------------------------------------------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download the dataset from Kaggle "
            f"and place the CSV in the data/ folder (see README)."
        )

    df = pd.read_csv(csv_path)
    cols = detect_columns(df)

    if "keyword" not in cols:
        raise ValueError(
            "Could not find a keyword column automatically. "
            f"Your CSV columns are: {list(df.columns)}. "
            "Add the correct header to COLUMN_ALIASES['keyword'] in data_processing.py."
        )

    # Build a clean, standardized dataframe
    clean = pd.DataFrame()
    clean["keyword"] = df[cols["keyword"]].astype(str).str.strip().str.lower()
    clean = clean[clean["keyword"].str.len() > 0].drop_duplicates(subset="keyword")

    # Numeric fields — pull in if present, else synthesize placeholders
    rng = np.random.default_rng(42)
    n = len(clean)

    if "volume" in cols:
        clean["volume"] = pd.to_numeric(df.loc[clean.index, cols["volume"]], errors="coerce")
    else:
        clean["volume"] = rng.integers(10, 20000, size=n)
    clean["volume"] = clean["volume"].fillna(clean["volume"].median())

    if "cpc" in cols:
        clean["cpc"] = pd.to_numeric(df.loc[clean.index, cols["cpc"]], errors="coerce")
    else:
        clean["cpc"] = rng.uniform(0.1, 8.0, size=n)
    clean["cpc"] = clean["cpc"].fillna(clean["cpc"].median())

    if "competition" in cols:
        comp = pd.to_numeric(df.loc[clean.index, cols["competition"]], errors="coerce")
        if comp.max() and comp.max() > 1.5:
            comp = comp / 100.0
        clean["competition"] = comp
    else:
        clean["competition"] = rng.uniform(0, 1, size=n)
    clean["competition"] = clean["competition"].fillna(clean["competition"].median())

    if "difficulty" in cols:
        diff = pd.to_numeric(df.loc[clean.index, cols["difficulty"]], errors="coerce")
        # always normalize to a 0-100 scale, regardless of the source scale
        diff_min, diff_max = diff.min(), diff.max()
        if diff_max and diff_max > diff_min:
            diff = (diff - diff_min) / (diff_max - diff_min) * 100
        else:
            diff = diff.fillna(50)
        clean["difficulty"] = diff
    else:
        clean["difficulty"] = (clean["competition"] * 70 +
                                (clean["volume"] / clean["volume"].max()) * 30)
    clean["difficulty"] = clean["difficulty"].fillna(clean["difficulty"].median()).clip(0, 100)

    clean = clean.reset_index(drop=True)
    return clean


# ---------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------
COMMERCIAL_WORDS = {"buy", "price", "cheap", "best", "discount", "deal", "coupon",
                     "review", "vs", "top", "affordable", "near me", "cost"}
INFORMATIONAL_WORDS = {"how", "what", "why", "guide", "tutorial", "tips", "meaning",
                        "examples", "learn", "definition"}
TRANSACTIONAL_WORDS = {"buy", "order", "purchase", "shop", "subscribe", "download",
                        "sign up", "free trial"}


def rule_based_intent(keyword: str) -> str:
    k = keyword.lower()
    if any(w in k for w in TRANSACTIONAL_WORDS):
        return "Transactional"
    if any(w in k for w in COMMERCIAL_WORDS):
        return "Commercial"
    if any(w in k for w in INFORMATIONAL_WORDS):
        return "Informational"
    return "Navigational"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word_count"] = df["keyword"].str.split().str.len()
    df["char_count"] = df["keyword"].str.len()
    df["is_long_tail"] = (df["word_count"] >= 4).astype(int)
    df["has_number"] = df["keyword"].str.contains(r"\d").astype(int)
    df["has_question_word"] = df["keyword"].str.contains(
        r"\b(?:how|what|why|who|when|where|which)\b", regex=True).astype(int)

    if "intent" not in df.columns:
        df["intent"] = df["keyword"].apply(rule_based_intent)

    vol_norm = (df["volume"] - df["volume"].min()) / (df["volume"].max() - df["volume"].min() + 1e-9)
    diff_norm = df["difficulty"] / 100.0
    cpc_norm = (df["cpc"] - df["cpc"].min()) / (df["cpc"].max() - df["cpc"].min() + 1e-9)

    df["opportunity_score"] = (
        (vol_norm * 0.45) +
        ((1 - diff_norm) * 0.40) +
        (cpc_norm * 0.15)
    ) * 100

    return df


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "keywords.csv")
    data = load_dataset(path)
    data = engineer_features(data)
    print(data.head(10))
    print(f"\nLoaded {len(data)} keywords.")