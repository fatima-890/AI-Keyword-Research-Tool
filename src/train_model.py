"""
train_model.py
---------------
Trains two ML components on the keyword dataset:

1. TF-IDF + KMeans  -> groups keywords into semantic topic clusters
2. RandomForestRegressor -> predicts an "Opportunity Score" for a
   keyword using text + numeric features (works even for brand-new keywords)

Run: python -m src.train_model   (from project root)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from src.data_processing import load_dataset, engineer_features
except ImportError:
    from data_processing import load_dataset, engineer_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "keywords_expanded.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_CLUSTERS = 10


def train_clustering(df: pd.DataFrame):
    print("Training TF-IDF + KMeans keyword clustering...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X = tfidf.fit_transform(df["keyword"])

    k = min(N_CLUSTERS, max(2, len(df) // 20))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    terms = tfidf.get_feature_names_out()
    cluster_labels = {}
    for c in range(k):
        center = kmeans.cluster_centers_[c]
        top_idx = center.argsort()[-3:][::-1]
        cluster_labels[c] = ", ".join(terms[i] for i in top_idx)
    df["cluster_label"] = df["cluster"].map(cluster_labels)

    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    joblib.dump(cluster_labels, os.path.join(MODELS_DIR, "cluster_labels.pkl"))

    print(f"Created {k} keyword clusters.")
    return df


def train_opportunity_model(df: pd.DataFrame):
    print("Training Opportunity Score regressor...")

    features = ["word_count", "char_count", "is_long_tail", "has_number", "has_question_word"]
    X_numeric = df[features]
    y = df["opportunity_score"]

    intent_dummies = pd.get_dummies(df["intent"], prefix="intent")
    X = pd.concat([X_numeric, intent_dummies], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Opportunity model  ->  MAE: {mae:.2f}   R2: {r2:.3f}")

    # Capture the prediction range on ALL training data so the app can
    # rescale new predictions to use the full 0-100 range meaningfully
    all_preds = model.predict(X)
    pred_min, pred_max = float(all_preds.min()), float(all_preds.max())
    print(f"Prediction range on training data: {pred_min:.2f} - {pred_max:.2f}")

    joblib.dump(model, os.path.join(MODELS_DIR, "opportunity_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "opportunity_features.pkl"))
    joblib.dump((pred_min, pred_max), os.path.join(MODELS_DIR, "opportunity_pred_range.pkl"))

    return model, mae, r2


def main():
    # Clustering uses the larger expanded dataset for richer topic groups
    df_expanded = load_dataset(DATA_PATH)
    df_expanded = engineer_features(df_expanded)
    df_expanded = train_clustering(df_expanded)

    # Opportunity model trains on the original, real Kaggle data only —
    # synthetic expanded numbers would teach the model noise, not signal
    original_path = os.path.join(BASE_DIR, "..", "data", "keywords.csv")
    df_original = load_dataset(original_path)
    df_original = engineer_features(df_original)
    train_opportunity_model(df_original)

    df_expanded.to_csv(os.path.join(MODELS_DIR, "processed_keywords.csv"), index=False)
    print(f"\nProcessed dataset saved with {len(df_expanded)} keywords -> models/processed_keywords.csv")
    print("Opportunity model trained on original (real) dataset only.")
    print("All models saved to /models. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()