"""
app.py — Credit Card Fraud Detection · Streamlit Application
=============================================================
Modes:
  1. Single Transaction  — manual slider / number inputs → instant prediction
  2. Batch Upload        — upload a CSV → predictions + downloadable results
  3. Model Insights      — feature importances, metrics comparison

Run locally:
  streamlit run app.py
"""

import io
import os
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield · Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* Root variables */
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --danger: #ef4444;
    --success: #10b981;
    --warning: #f59e0b;
    --bg-dark: #0f0f1a;
    --bg-card: #1a1a2e;
    --bg-card2: #16213e;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f0f1a 100%);
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e1e3a 0%, #1a1a2e 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.7);
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    font-weight: 600;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1.3;
}

/* Prediction badges */
.badge-fraud {
    display: inline-block;
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.6rem 1.8rem;
    border-radius: 999px;
    letter-spacing: 0.05em;
    box-shadow: 0 0 20px rgba(239,68,68,0.5);
}
.badge-legit {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #065f46);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.6rem 1.8rem;
    border-radius: 999px;
    letter-spacing: 0.05em;
    box-shadow: 0 0 20px rgba(16,185,129,0.5);
}

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.8rem;
    border-left: 4px solid #6366f1;
    padding-left: 0.75rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* Divider */
hr {
    border-color: rgba(99,102,241,0.15) !important;
}

.info-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #c7d2fe;
    font-size: 0.88rem;
    line-height: 1.6;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = ["Time"] + V_COLS + ["Amount"]

MODEL_INFO = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "icon": "📐",
        "desc": "Linear baseline — fast, interpretable, calibrated probabilities",
    },
    "decision_tree": {
        "label": "Decision Tree",
        "icon": "🌳",
        "desc": "Rule-based — fully explainable, great for regulatory compliance",
    },
    "random_forest": {
        "label": "Random Forest",
        "icon": "🌲",
        "desc": "Ensemble — highest performance, robust to outliers",
    },
}


# ── Helper functions ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_models() -> dict:
    """Load all serialised model pipelines from the models/ directory."""
    loaded = {}
    for key in MODEL_INFO:
        path = os.path.join(MODELS_DIR, f"{key}.joblib")
        if os.path.exists(path):
            loaded[key] = joblib.load(path)
    return loaded


def predict_single(model, input_dict: dict) -> tuple[int, float]:
    """Run single-transaction prediction. Returns (label, fraud_probability)."""
    df = pd.DataFrame([input_dict])[FEATURE_COLS]
    prob = model.predict_proba(df)[0, 1]
    label = int(prob >= 0.5)
    return label, prob


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run batch predictions. Adds 'Fraud_Probability' and 'Prediction' columns."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
        return df
    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df = df.copy()
    df["Fraud_Probability"] = np.round(probs, 4)
    df["Prediction"] = (probs >= 0.5).astype(int).map({0: "Legit ✅", 1: "FRAUD 🚨"})
    return df


def render_metric_cards(metrics: dict):
    """Render a row of styled metric cards."""
    cols = st.columns(5)
    items = [
        ("AUPRC", metrics.get("auprc", "—")),
        ("ROC-AUC", metrics.get("roc_auc", "—")),
        ("F1", metrics.get("f1", "—")),
        ("Recall", metrics.get("recall", "—")),
        ("Precision", metrics.get("precision", "—")),
    ]
    for col, (label, val) in zip(cols, items):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-header">
    <h1 class="hero-title">🛡️ FraudShield</h1>
    <p class="hero-subtitle">Credit Card Fraud Detection · ML-Powered Risk Assessment</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FraudShield")
    st.markdown("<small style='color:#94a3b8'>ML-Powered Fraud Detection</small>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Single Transaction", "📂 Batch Upload", "📊 Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    model_key = st.selectbox(
        "Active Model",
        options=list(MODEL_INFO.keys()),
        format_func=lambda k: f"{MODEL_INFO[k]['icon']} {MODEL_INFO[k]['label']}",
    )

    st.markdown("---")
    st.markdown(
        """<div class="info-box">
        <b>📋 Dataset</b><br>
        Kaggle Credit Card Fraud 2013<br>
        284,807 transactions · 0.17% fraud<br><br>
        <b>⚙️ Features</b><br>
        Time, V1–V28 (PCA), Amount
        </div>""",
        unsafe_allow_html=True,
    )

# ── Load models ────────────────────────────────────────────────────────────────
models = load_models()

if not models:
    st.warning(
        "⚠️ No trained models found in `models/`. "
        "Please run `python src/train.py` first.",
        icon="⚠️",
    )
    st.stop()

active_model = models[model_key]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Transaction
# ══════════════════════════════════════════════════════════════════════════════
if "Single" in page:
    st.markdown('<div class="section-header">🔍 Single Transaction Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        f"Using **{MODEL_INFO[model_key]['icon']} {MODEL_INFO[model_key]['label']}** — "
        f"<span style='color:#94a3b8'>{MODEL_INFO[model_key]['desc']}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown(
            "Adjust the sliders to describe a transaction. In real usage the V-features "
            "are PCA-transformed by the card network — for demo purposes you can use the "
            "default values or paste values from the raw dataset."
        )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Transaction Details**")
        time_val = st.number_input("Time (seconds since first transaction)", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)
        amount_val = st.number_input("Amount (USD)", min_value=0.0, max_value=30000.0, value=120.0, step=1.0)

    with col_right:
        st.markdown("**PCA Feature Presets**")
        preset = st.selectbox(
            "Load a preset",
            ["Custom (manual)", "Typical Legitimate", "Suspicious Pattern"],
        )

    # V-feature defaults
    if preset == "Typical Legitimate":
        v_defaults = {f"V{i}": np.random.normal(0, 0.5) for i in range(1, 29)}
    elif preset == "Suspicious Pattern":
        v_defaults = {
            "V1": -3.0, "V2": 2.5, "V3": -2.8, "V4": 3.1, "V5": -1.9,
            "V6": -1.2, "V7": -2.6, "V8": 0.7, "V9": -0.8, "V10": -3.5,
            "V11": 2.0, "V12": -3.8, "V13": 0.5, "V14": -4.2, "V15": 0.3,
            "V16": -2.1, "V17": -3.2, "V18": -1.8, "V19": 0.4, "V20": 0.6,
            "V21": 0.9, "V22": -0.3, "V23": 0.1, "V24": -0.5, "V25": 0.2,
            "V26": -0.1, "V27": 0.8, "V28": 0.4,
        }
    else:
        v_defaults = {f"V{i}": 0.0 for i in range(1, 29)}

    with st.expander("⚙️ PCA Features (V1–V28)", expanded=False):
        v_cols1, v_cols2 = st.columns(2)
        v_inputs = {}
        for i, v in enumerate(V_COLS):
            col = v_cols1 if i % 2 == 0 else v_cols2
            v_inputs[v] = col.number_input(
                v, value=float(round(v_defaults[v], 4)),
                min_value=-20.0, max_value=20.0, step=0.01, key=f"v_{v}"
            )

    st.markdown("")
    predict_btn = st.button("🔮 Predict Transaction", use_container_width=True)

    if predict_btn:
        input_data = {"Time": time_val, "Amount": amount_val, **v_inputs}
        with st.spinner("Analysing transaction …"):
            time.sleep(0.4)  # UX micro-delay
            label, prob = predict_single(active_model, input_data)

        st.markdown("---")
        result_col, prob_col = st.columns([2, 1])

        with result_col:
            if label == 1:
                st.markdown(
                    '<div style="text-align:center;padding:1.5rem">'
                    '<span class="badge-fraud">🚨 FRAUD DETECTED</span></div>',
                    unsafe_allow_html=True,
                )
                st.error(f"This transaction has been flagged as **fraudulent** with {prob*100:.1f}% confidence.")
            else:
                st.markdown(
                    '<div style="text-align:center;padding:1.5rem">'
                    '<span class="badge-legit">✅ LEGITIMATE</span></div>',
                    unsafe_allow_html=True,
                )
                st.success(f"This transaction appears **legitimate** (fraud probability: {prob*100:.1f}%).")

        with prob_col:
            st.metric("Fraud Probability", f"{prob*100:.2f}%")
            risk = "🔴 HIGH" if prob > 0.7 else "🟡 MEDIUM" if prob > 0.3 else "🟢 LOW"
            st.metric("Risk Level", risk)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Upload
# ══════════════════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown('<div class="section-header">📂 Batch Transaction Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f"Upload a CSV file with the same columns as the training data to score multiple transactions at once."
    )

    uploaded = st.file_uploader(
        "Upload CSV (must contain Time, V1–V28, Amount columns)",
        type=["csv"],
        help="The file should match the creditcard.csv schema.",
    )

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_raw):,} transactions loaded.** Previewing first 5 rows:")
        st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner("Running predictions …"):
                df_result = predict_batch(active_model, df_raw)

            fraud_count = (df_result["Prediction"] == "FRAUD 🚨").sum()
            legit_count = len(df_result) - fraud_count
            fraud_pct = fraud_count / len(df_result) * 100

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Transactions", f"{len(df_result):,}")
            m2.metric("Fraudulent", f"{fraud_count:,} ({fraud_pct:.2f}%)", delta=None)
            m3.metric("Legitimate", f"{legit_count:,}")

            st.markdown("### Results")
            # Highlight fraud rows
            def highlight_fraud(row):
                if row["Prediction"] == "FRAUD 🚨":
                    return ["background-color: rgba(239,68,68,0.15)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_result.style.apply(highlight_fraud, axis=1),
                use_container_width=True,
                height=400,
            )

            # Download button
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if fraud_count > 0:
                st.markdown("### 🚨 Fraud Transactions Detail")
                st.dataframe(
                    df_result[df_result["Prediction"] == "FRAUD 🚨"].sort_values(
                        "Fraud_Probability", ascending=False
                    ),
                    use_container_width=True,
                )
    else:
        st.info("👆 Upload a CSV file to get started. The file must contain columns: **Time**, **V1–V28**, and **Amount**.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
elif "Insights" in page:
    st.markdown('<div class="section-header">📊 Model Insights & Performance</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Model Overview", "🔬 Feature Importances"])

    with tab1:
        st.markdown("### Available Models")
        for key, info in MODEL_INFO.items():
            status = "✅ Loaded" if key in models else "❌ Not found"
            with st.container():
                st.markdown(
                    f"""
**{info['icon']} {info['label']}** &nbsp;·&nbsp; <span style='color:#94a3b8'>{status}</span>

{info['desc']}
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("---")

        st.markdown("### 📚 About the Dataset")
        st.markdown(
            """
| Property | Value |
|---|---|
| Source | Kaggle — ULB Credit Card Fraud Detection |
| Rows | 284,807 |
| Features | Time, V1–V28 (PCA), Amount |
| Target | Class (0 = Legit, 1 = Fraud) |
| Fraud rate | 0.172% (highly imbalanced) |
| Imbalance strategy | SMOTE + class_weight="balanced" |
| Preprocessing | RobustScaler on Time & Amount |
| Evaluation metric | AUPRC (primary), F1, Recall |
            """
        )

    with tab2:
        st.markdown("### Feature Importances")
        if model_key == "random_forest" and model_key in models:
            rf_model = models[model_key]
            classifier = rf_model.named_steps["classifier"]
            importances = classifier.feature_importances_

            # Get feature names after preprocessing
            preprocessor = rf_model.named_steps["preprocessor"]
            try:
                feat_names = preprocessor.get_feature_names_out()
            except Exception:
                feat_names = FEATURE_COLS

            imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=False).head(20)
            imp_df["Importance"] = imp_df["Importance"].round(4)

            st.bar_chart(imp_df.set_index("Feature")["Importance"])
            st.dataframe(imp_df, use_container_width=True)

        elif model_key == "logistic_regression" and model_key in models:
            lr_model = models[model_key]
            classifier = lr_model.named_steps["classifier"]
            coefs = classifier.coef_[0]

            preprocessor = lr_model.named_steps["preprocessor"]
            try:
                feat_names = preprocessor.get_feature_names_out()
            except Exception:
                feat_names = FEATURE_COLS

            coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
            coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()
            coef_df = coef_df.sort_values("Abs_Coef", ascending=False).head(20)

            st.bar_chart(coef_df.set_index("Feature")["Coefficient"])
            st.dataframe(coef_df.drop(columns="Abs_Coef"), use_container_width=True)

        else:
            st.info(
                "Feature importances are available for **Random Forest** and **Logistic Regression**. "
                "Decision Tree importances are similar to Random Forest."
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem'>"
    "FraudShield · Built with Streamlit & scikit-learn · "
    "Dataset: Kaggle Credit Card Fraud Detection (ULB)"
    "</div>",
    unsafe_allow_html=True,
)
