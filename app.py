"""
app.py
------
Streamlit dashboard for the AI Keyword Research Tool.
Run with:  python -m streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from rapidfuzz import process, fuzz

from src.data_processing import rule_based_intent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="AI Keyword Research Tool", page_icon="🔍", layout="wide")


@st.cache_data
def load_processed_data():
    path = os.path.join(MODELS_DIR, "processed_keywords.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    model = joblib.load(os.path.join(MODELS_DIR, "opportunity_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "opportunity_features.pkl"))
    pred_range = joblib.load(os.path.join(MODELS_DIR, "opportunity_pred_range.pkl"))
    return model, feature_cols, pred_range


def score_new_keyword(keyword: str, model, feature_cols, pred_range):
    word_count = len(keyword.split())
    char_count = len(keyword)
    is_long_tail = int(word_count >= 4)
    has_number = int(any(ch.isdigit() for ch in keyword))
    has_question_word = int(any(w in keyword.lower().split()
                                 for w in ["how", "what", "why", "who", "when", "where", "which"]))
    intent = rule_based_intent(keyword)

    row = {c: 0 for c in feature_cols}
    row["word_count"] = word_count
    row["char_count"] = char_count
    row["is_long_tail"] = is_long_tail
    row["has_number"] = has_number
    row["has_question_word"] = has_question_word
    intent_col = f"intent_{intent}"
    if intent_col in row:
        row[intent_col] = 1

    X = pd.DataFrame([row])[feature_cols]
    raw_score = model.predict(X)[0]

    # Rescale to full 0-100 range using the training prediction spread,
    # so scores aren't all compressed into a narrow band like 20-30
    pred_min, pred_max = pred_range
    if pred_max > pred_min:
        score = (raw_score - pred_min) / (pred_max - pred_min) * 100
    else:
        score = raw_score
    score = max(0, min(100, score))

    return round(score, 1), intent


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🔍 AI Keyword Research Tool")
st.caption("ML-powered keyword clustering, intent detection, and opportunity scoring")

df = load_processed_data()

if df is None:
    st.error(
        "No trained data found. Run these first from the project root:\n\n"
        "```\npython -m src.train_model\n```"
    )
    st.stop()

model, feature_cols, pred_range = load_models()

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧩 Topic Clusters", "🎯 Score a New Keyword"])

# --- TAB 1: Overview ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Keywords", f"{len(df):,}")
    col2.metric("Avg. Search Volume", f"{df['volume'].mean():,.0f}")
    col3.metric("Avg. Difficulty", f"{df['difficulty'].mean():.1f}")
    col4.metric("Topic Clusters", df["cluster"].nunique())

    st.subheader("Search & Filter Keywords")
    search_term = st.text_input("Fuzzy-search keywords (e.g. a topic or seed word)")

    view = df.copy()
    if search_term:
        matches = process.extract(search_term, df["keyword"], scorer=fuzz.WRatio, limit=200)
        matched_keywords = [m[0] for m in matches if m[1] > 60]
        view = df[df["keyword"].isin(matched_keywords)]

    st.dataframe(
        view[["keyword", "volume", "cpc", "difficulty", "intent", "opportunity_score", "cluster_label"]]
        .sort_values("opportunity_score", ascending=False),
        use_container_width=True,
        height=400,
    )

    fig = px.scatter(
        df, x="difficulty", y="volume", color="intent", size="opportunity_score",
        hover_data=["keyword"], title="Volume vs. Difficulty (bubble size = opportunity score)",
        log_y=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Clusters ---
with tab2:
    st.subheader("Auto-Generated Content Clusters")
    st.write("Each cluster groups semantically related keywords — use these as content pillar ideas.")

    cluster_summary = (
        df.groupby(["cluster", "cluster_label"])
        .agg(keywords=("keyword", "count"),
             avg_volume=("volume", "mean"),
             avg_difficulty=("difficulty", "mean"),
             avg_opportunity=("opportunity_score", "mean"))
        .reset_index()
        .sort_values("avg_opportunity", ascending=False)
    )
    st.dataframe(cluster_summary, use_container_width=True)

    chosen_cluster = st.selectbox(
        "View keywords in a cluster",
        options=cluster_summary["cluster"],
        format_func=lambda c: cluster_summary.loc[cluster_summary["cluster"] == c, "cluster_label"].values[0],
    )
    st.dataframe(
        df[df["cluster"] == chosen_cluster][["keyword", "volume", "difficulty", "opportunity_score"]]
        .sort_values("opportunity_score", ascending=False),
        use_container_width=True,
    )

# --- TAB 3: Score new keyword ---
with tab3:
    st.subheader("Predict Opportunity for a New Keyword")
    st.write("Type any keyword idea — the model estimates intent and an opportunity score (0-100) "
             "based on patterns learned from the dataset.")

    new_kw = st.text_input("Enter a keyword", placeholder="e.g. best ai seo tools for beginners")
    if st.button("Score Keyword") and new_kw.strip():
        score, intent = score_new_keyword(new_kw.strip(), model, feature_cols, pred_range)
        c1, c2 = st.columns(2)
        c1.metric("Predicted Opportunity Score", f"{score}/100")
        c2.metric("Detected Intent", intent)

        if score >= 70:
            st.success("High opportunity — strong candidate for content targeting.")
        elif score >= 45:
            st.info("Moderate opportunity — consider as a supporting/long-tail keyword.")
        else:
            st.warning("Lower opportunity — likely high competition or low relevance signal.")

st.divider()
st.caption("Built with scikit-learn, TF-IDF, KMeans, and RapidFuzz · Data: Kaggle SEO Keyword Research dataset")