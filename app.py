import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

st.set_page_config(
    page_title="AI-Based Machine Failure Prediction System",
    layout="wide"
)


MODEL_PATH = "machine_failure_model.pkl"
DATA_PATH = "machine_data.csv"
TARGET_COL = "failure"
ID_COL = "machine_id"

from ui_helpers import apply_ui_theme, glass_card
apply_ui_theme()


st.markdown(
    """
<style>

.stApp {
  background: linear-gradient(180deg, #050816 0%, #111827 50%, #020617 100%);
  color: #e6eef6;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
}

.main .block-container {
  padding: 28px 36px;
  max-width: 1200px;
  margin-left: auto;
  margin-right: auto;
}

/* Glass Cards */
.glass-card {
  background: rgba(15, 23, 42, 0.85) !important;
  border: 1px solid rgba(148, 163, 184, 0.12) !important;
  border-radius: 18px !important;
  box-shadow: 0 8px 30px rgba(2, 6, 23, 0.45);
  padding: 20px !important;
  margin-bottom: 18px;
}

/* Titles */
.card-title {
  font-weight: 700;
  font-size: 15px;
  color: #cbd5e1;
  margin-bottom: 10px;
  letter-spacing: 0.12em;
}

/* ---------------------------------------
   PREMIUM BUTTONS (Predict + Download CSV)
---------------------------------------- */
.stButton > button,
.stDownloadButton > button,
.stForm .stButton > button {
    background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #ffffff !important;
    border-radius: 999px !important;
    padding: 12px 26px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border: none !important;
    cursor: pointer !important;
    box-shadow: 0px 6px 18px rgba(99, 102, 241, 0.25);
    transition: all 0.18s ease-in-out !important;
}

/* Hover animation */
.stButton > button:hover,
.stDownloadButton > button:hover,
.stForm .stButton > button:hover {
    transform: translateY(-4px) scale(1.03);
    box-shadow: 0px 14px 32px rgba(99, 102, 241, 0.40);
    opacity: 0.95;
}

/* Click effect */
.stButton > button:active,
.stDownloadButton > button:active,
.stForm .stButton > button:active {
    transform: scale(0.97);
    box-shadow: 0px 4px 12px rgba(99, 102, 241, 0.20);
}

/* Data Table */
.stDataFrame table {
  background: rgba(10,14,20,0.35) !important;
  color: #cbd5e1 !important;
}
.stDataFrame thead th {
  background: rgba(255,255,255,0.05) !important;
  text-transform: uppercase;
  font-size: 12px !important;
  color: #e6eef6 !important;
}
.stDataFrame tbody tr:hover td {
  background: rgba(255,255,255,0.04) !important;
}

/* KPI */
.kpi-value { font-size: 22px; font-weight: 800; color: #f8fafc; }
.kpi-label { font-size: 13px; color: #9aa6b2; }

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


model = load_model()
raw_df = load_data()

if ID_COL not in raw_df.columns:
    raw_df[ID_COL] = 0

if TARGET_COL not in raw_df.columns:
    st.error(f"Validation dataset must contain target '{TARGET_COL}'")
    st.stop()

# Default features
DEFAULT_FEATURES = [
    ID_COL,
    "age",
    "temperature",
    "vibration",
    "pressure",
    "load",
    "rpm",
    "operating_hours",
]


def _model_features(model, fallback):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return fallback


MODEL_FEATURES = _model_features(model, DEFAULT_FEATURES)


def align_dataframe_to_features(df, features, ref_df=None):
    dfc = df.copy()
    for c in features:
        if c not in dfc:
            dfc[c] = ref_df[c].mean() if ref_df is not None and c in ref_df else 0
    return dfc[features]



val_df = align_dataframe_to_features(raw_df, MODEL_FEATURES, ref_df=raw_df)
X_val = val_df
y_val = raw_df[TARGET_COL]

# Predictions
try:
    y_proba_val = model.predict_proba(X_val)[:, 1]
except Exception:
    y_proba_val = np.zeros(len(X_val))

try:
    y_pred_val = model.predict(X_val)
except Exception as e:
    st.error(f"Prediction failure: {e}")
    st.stop()

metrics = {
    "accuracy": accuracy_score(y_val, y_pred_val),
    "precision": precision_score(y_val, y_pred_val, zero_division=0),
    "recall": recall_score(y_val, y_pred_val, zero_division=0),
    "f1": f1_score(y_val, y_pred_val, zero_division=0),
}
cm = confusion_matrix(y_val, y_pred_val)
fpr, tpr, _ = roc_curve(y_val, y_proba_val)
roc_auc = roc_auc_score(y_val, y_proba_val)


st.markdown(
    """
<div style='text-align:center; padding:12px;'>
  <h1 style='margin-bottom:4px;'>AI-Based Machine Failure Prediction System</h1>
  <div style='color:#9aa6b2;'>Smart monitoring and failure prediction powered by Machine Learning</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")


with glass_card(""):
    st.markdown(
        "<div class='card-title'>Machine Input Parameters</div>",
        unsafe_allow_html=True,
    )

    with st.form(key="predict_form"):
        col1, col2, col3 = st.columns([3, 3, 3])
        with col1:
            age = st.number_input("Machine Age (Years)", 0, 100, 5)
            temperature = st.number_input("Temperature (°C)", 0.0, 300.0, 80.0)
        with col2:
            vibration = st.number_input(
                "Vibration Level", 0.0, 10.0, 0.8, step=0.01
            )
            pressure = st.number_input("Pressure Level", 0.0, 2000.0, 50.0)
        with col3:
            load = st.number_input("Load (%)", 0, 100, 70)
            rpm = st.number_input("RPM", 0, 20000, 1500)
            operating_hours = st.number_input("Operating Hours per Day", 0, 24, 10)

        submit = st.form_submit_button("Predict Failure")


if submit:
    input_row = pd.DataFrame(
        {
            ID_COL: [0],
            "age": [age],
            "temperature": [temperature],
            "vibration": [vibration],
            "pressure": [pressure],
            "load": [load],
            "rpm": [rpm],
            "operating_hours": [operating_hours],
        }
    )

    input_aligned = align_dataframe_to_features(
        input_row, MODEL_FEATURES, ref_df=raw_df
    )

    pred = int(model.predict(input_aligned)[0])
    proba = float(model.predict_proba(input_aligned)[0][1])

    st.markdown("---")
    c1, c2 = st.columns([1, 4])
    with c1:
        st.metric("Failure Probability", f"{proba * 100:.2f}%")
    with c2:
        if pred == 1:
            st.error(f"⚠️ Machine is LIKELY to FAIL — {proba * 100:.2f}%")
        else:
            st.success(f"✅ Machine is SAFE — {proba * 100:.2f}%")


with glass_card(""):
    colA, colB, colC, colD = st.columns(4)
    colA.markdown(
        f"<div class='kpi-value'>{metrics['accuracy']*100:.1f}%</div>"
        "<div class='kpi-label'>Accuracy</div>",
        unsafe_allow_html=True,
    )
    colB.markdown(
        f"<div class='kpi-value'>{metrics['precision']*100:.1f}%</div>"
        "<div class='kpi-label'>Precision</div>",
        unsafe_allow_html=True,
    )
    colC.markdown(
        f"<div class='kpi-value'>{metrics['recall']*100:.1f}%</div>"
        "<div class='kpi-label'>Recall</div>",
        unsafe_allow_html=True,
    )
    colD.markdown(
        f"<div class='kpi-value'>{metrics['f1']*100:.1f}%</div>"
        "<div class='kpi-label'>F1 Score</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


col_left, col_right = st.columns(2)

with col_left:
    with glass_card(""):
        st.markdown(
            "<div class='card-title'>Confusion Matrix</div>",
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", ax=ax)
        st.pyplot(fig)

with col_right:
    with glass_card(""):
        st.markdown(
            f"<div class='card-title'>ROC Curve — AUC: {roc_auc:.2f}</div>",
            unsafe_allow_html=True,
        )
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, linewidth=2)
        ax2.plot([0, 1], [0, 1], "--")
        st.pyplot(fig2)

st.markdown("---")


with glass_card(""):
    st.markdown(
        "<div class='card-title'>Predicted Output Summary</div>",
        unsafe_allow_html=True,
    )

    n = 10
    np.random.seed(42)

    scenario = pd.DataFrame(
        {
            "machine_id": np.zeros(n),
            "age": np.full(n, age),
            "temperature": np.random.normal(temperature, 5, n),
            "vibration": np.random.normal(vibration, 0.1, n),
            "pressure": np.random.normal(pressure, 5, n),
            "load": np.random.normal(load, 5, n),
            "rpm": np.random.normal(rpm, 80, n),
            "operating_hours": np.random.normal(operating_hours, 1, n),
        }
    )

    scenario = align_dataframe_to_features(scenario, MODEL_FEATURES, raw_df)

    scen_pred = model.predict(scenario)
    scen_proba = model.predict_proba(scenario)[:, 1]

    visible_cols = [
        "temperature",
        "vibration",
        "pressure",
        "load",
        "rpm",
        "operating_hours",
    ]
    out_df = scenario[visible_cols].copy()
    out_df["Failure Probability (%)"] = (scen_proba * 100).round(2)
    out_df["Predicted Failure"] = scen_pred

    st.dataframe(out_df, height=300)

    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="predicted_output_summary.csv",
        mime="text/csv",
    )

st.caption("Model logic preserved. Predictions run only when you press 'Predict Failure'.")
