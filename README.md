# 🛡️ Credit Card Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A machine learning system that detects fraudulent credit card transactions with high precision — featuring a real-time Streamlit dashboard.**

[Live Demo](https://your-app.streamlit.app) · [Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [Report an Issue](https://github.com/yourname/credit-card-fraud-detection/issues)

</div>

---

## 📌 Overview

This project trains three ML classifiers on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate) and deploys a **Streamlit web app** for real-time fraud risk assessment.

### Why this problem is hard
- **Extreme class imbalance** — 99.83% legitimate vs. 0.17% fraud
- Standard accuracy is meaningless — we optimise for **AUPRC** and **Recall**
- Features V1–V28 are PCA-transformed by the card network (anonymised)

---

## 🏗️ Project Structure

```
credit-card-fraud-detection/
├── creditcard.csv              ← raw data (not in git — too large)
├── data/
│   └── processed/              ← scaled outputs (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory Data Analysis
│   └── 02_modelling.ipynb      ← Training, evaluation, comparison
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        ← sklearn ColumnTransformer pipeline
│   ├── train.py                ← trains & serialises all models
│   └── evaluate.py             ← AUPRC, F1, confusion matrix helpers
├── models/                     ← serialised .joblib files
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   └── random_forest.joblib
├── app.py                      ← Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🤖 Models

| Model | Approach | Imbalance Strategy |
|---|---|---|
| **Logistic Regression** | Linear baseline | SMOTE + `class_weight="balanced"` |
| **Decision Tree** | Rule-based | SMOTE + `class_weight="balanced"` |
| **Random Forest** | Ensemble | `class_weight="balanced"` |

### Preprocessing Pipeline
```
Time, Amount → RobustScaler ┐
V1–V28       → Passthrough  ┴→ SMOTE (train only) → Classifier
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourname/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 5. Train the models
```bash
python src/train.py
```
This will save three `.joblib` files in `models/`.

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Results

| Model | AUPRC | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| Random Forest | ~0.87 | ~0.97 | ~0.86 | ~0.84 | ~0.88 |
| Logistic Regression | ~0.72 | ~0.97 | ~0.74 | ~0.91 | ~0.62 |
| Decision Tree | ~0.74 | ~0.93 | ~0.80 | ~0.79 | ~0.82 |

> Results may vary slightly depending on random seed and SMOTE sampling.

---

## 🌐 Deploying to Streamlit Cloud

1. Push this repo to GitHub (without `creditcard.csv` — it's in `.gitignore`).
2. Upload `models/*.joblib` files (they are small enough for git).
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → connect your repo.
4. Set **Main file path** to `app.py`.
5. Click **Deploy** 🚀

> ⚠️ The CSV is too large for GitHub. The Streamlit app works with pre-trained `.joblib` models — no CSV needed at runtime.

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Class distribution, correlation heatmap, amount/time distributions |
| `02_modelling.ipynb` | Train all models, plot AUPRC curves, compare confusion matrices |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — models, pipelines, metrics
- **imbalanced-learn** — SMOTE oversampling
- **pandas / numpy** — data wrangling
- **Streamlit** — web application
- **joblib** — model serialisation
- **matplotlib / seaborn / plotly** — visualisation

---

## 📄 License

MIT © 2024 — feel free to use for learning and portfolio projects.
