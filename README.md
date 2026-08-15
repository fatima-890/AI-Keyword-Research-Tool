# 🔍 AI Keyword Research Tool

An ML-powered keyword research tool that clusters SEO keywords into content topics, detects search intent, and predicts an **opportunity score** (0–100) for any keyword — including brand-new ones typed by the user — using scikit-learn and NLP.

---

## 📌 Project Description
>ML-powered SEO keyword research tool built with Python and scikit-learn. Uses TF-IDF + KMeans to auto-cluster keywords into content topics, RandomForest to predict an opportunity score for any keyword (even new, unseen ones), and rule-based NLP for search intent classification — all wrapped in an interactive Streamlit dashboard.

---

## 📖 About

I built this project to combine two things I actively work on — machine learning and SEO/content strategy — into one practical tool. As someone running an affiliate marketing brand and an AI tools blog, keyword research is something I do regularly, and this project automates part of that process using real ML techniques instead of just calling third-party SEO APIs.

The tool takes a raw keyword dataset and turns it into:
- **Content topic clusters** (so you know what pillar pages to build)
- **Search intent labels** (so you know what type of content to write)
- **An opportunity score for any keyword you type**, even one that was never in the original dataset

This is part of an ongoing series of ML/NLP portfolio projects I'm building to combine my AI/ML coursework with my background in SEO and content creation.

---

## ✨ Features
- **Topic Clustering** — TF-IDF + KMeans groups related keywords into content pillars automatically
- **Opportunity Scoring** — RandomForestRegressor scores any keyword on a 0–100 scale, rescaled to use the full range meaningfully, and works even for keywords not in the dataset
- **Search Intent Detection** — classifies keywords as Informational, Commercial, Transactional, or Navigational
- **Fuzzy Search** — RapidFuzz-powered keyword search across the dataset
- **Dataset Expansion** — an optional script expands a small seed dataset into a much larger one using SEO modifier combinations, for richer clustering
- **Interactive Dashboard** — Streamlit app with charts, a cluster explorer, and a live keyword scorer

---

## 📊 Dataset

This project uses the **[SEO Keyword Research dataset on Kaggle](https://www.kaggle.com/datasets/sheryshisingh/seo-keyword-research)**.

> ⚠️ Kaggle dataset column names vary between contributors and can change. This project **auto-detects** the keyword/volume/CPC/competition/difficulty columns (see `COLUMN_ALIASES` in `src/data_processing.py`). If your downloaded CSV uses different headers, just add them to the alias list — no other code changes needed. If a column is missing entirely, the pipeline estimates a reasonable placeholder so training never breaks.

**Optional dataset expansion:** the original Kaggle dataset used here had 118 unique keywords. To get richer topic clusters, `expand_keywords.py` combines each seed keyword with common SEO modifiers (best, how to, near me, etc. — a technique real keyword-research tools also use) to generate a larger set (~2,400+ keywords). The opportunity-score model is intentionally trained only on the **original real keywords**, since the expanded set's numeric metrics (volume, CPC, difficulty) are synthetically simulated and would teach the model noise rather than signal. This split is a deliberate design choice, documented here rather than hidden.

---

## 🗂 Project Structure

ai-keyword-research-tool/
├── data/
│ └── keywords.csv # place the downloaded Kaggle CSV here
├── src/
│ ├── init.py
│ ├── data_processing.py # loading, cleaning, feature engineering
│ └── train_model.py # clustering + opportunity score model
├── models/ # saved models (generated after training)
├── app.py # Streamlit dashboard
├── expand_keywords.py # optional dataset expansion script
├── requirements.txt
├── .gitignore
└── README.md


---

## ⚙️ Setup & Usage

### 1. Clone and install dependencies
```bash
git clone https://github.com/fatima-890/AI-Keyword-Research-Tool.git
cd ai-keyword-research-tool
pip install -r requirements.txt
```

### 2. Download the dataset
Download the CSV from the [Kaggle dataset page](https://www.kaggle.com/datasets/sheryshisingh/seo-keyword-research), rename it to `keywords.csv`, and place it inside the `data/` folder.

### 3. (Optional) Expand the dataset for richer clustering
```bash
python expand_keywords.py
```
This creates `data/keywords_expanded.csv` — a larger set combining seed keywords with common SEO modifiers.

### 4. Train the models
```bash
python -m src.train_model
```
This creates:
- `models/tfidf_vectorizer.pkl`, `models/kmeans_model.pkl` — clustering
- `models/opportunity_model.pkl`, `models/opportunity_pred_range.pkl` — opportunity scoring
- `models/processed_keywords.csv` — the full processed dataset

### 5. Launch the dashboard
```bash
python -m streamlit run app.py
```
(Using `python -m streamlit` avoids PATH issues on Windows/PowerShell.)

---

## 🧠 Model Notes
The opportunity score model predicts based on text-derived features (word count, question words, detected intent, etc.), since a brand-new keyword has no historical volume/CPC data to draw on. With only 118 real training examples, the model captures broad patterns but can occasionally produce less intuitive scores on edge-case keyword phrasing — a known limitation of small training sets. Predictions are rescaled to use the full 0–100 range so scores don't compress into a narrow band. The clustering component (TF-IDF + KMeans) is the stronger, more directly useful part of the tool for real content planning, and benefits from the larger expanded dataset.

---

## 🛠 Tech Stack
Python · pandas · scikit-learn (TF-IDF, KMeans, RandomForest) · Streamlit · Plotly · RapidFuzz · Pillow

---

## 🤝 Contributing
This is primarily a personal portfolio project, but suggestions and improvements are welcome:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes and commit (`git commit -m "Add your feature"`)
4. Push to your branch and open a Pull Request

Bug reports and feature ideas are also welcome via GitHub Issues.

---

## 💬 Support
If you run into any issues setting this up or have questions about how it works, feel free to open a GitHub Issue on this repository, or reach out via my GitHub profile below. I try to respond and help where I can, especially for anyone learning ML/SEO the way I am.

---

## 📄 License
This project is open source and available under the MIT License.

---

## 👤 Author
**Esha Fatima**
GitHub:github.com/fatima-890
Linkedin: www.linkedin.com/in/esha-fatima-bba9423bb