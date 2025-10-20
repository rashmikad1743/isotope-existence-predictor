import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="centered", page_title="Isotope Existence Predictor")

# Custom CSS for styling
st.markdown("""
    <style>
    .main-card {
        background-color: #eaf3fa;
        border-radius: 12px;
        padding: 32px 24px 24px 24px;
        margin-top: 24px;
        margin-bottom: 24px;
        border: 1px solid #b3d1f1;
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #183153;
    }
    .prediction-label {
        font-size: 1.1rem;
        font-weight: 500;
        color: #183153;
    }
    .prediction-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #222;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.1rem;
        font-weight: 600;
        background-color: #1976d2;
        color: white;
        border-radius: 6px;
        padding: 0.75em 0;
    }
    .stTextInput>div>input {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
MODEL_PATH = "artifact/model.pkl"
with open(MODEL_PATH, "rb") as f:
    meta = pickle.load(f)
model = meta["model"]
feature_cols = meta["feature_columns"]
target_col = meta["target_col"]

st.markdown(
    '<h1 style="text-align:center;"><span style="font-size:2.2rem;">🔬 Isotope Existence Predictor</span></h1>',
    unsafe_allow_html=True
)

# Sidebar or main input form
with st.form("predict_form"):
    st.subheader("Enter Isotope Features")
    user_input = {}
    for col in feature_cols:
        # Try to guess numeric or text input
        if "number" in col.lower() or "count" in col.lower() or "energy" in col.lower() or "half" in col.lower() or "index" in col.lower():
            value = st.text_input(f"{col}:", value="")
            user_input[col] = value
        else:
            value = st.text_input(f"{col}:", value="")
            user_input[col] = value

    submitted = st.form_submit_button("🔍 Predict")

# Prediction logic
if submitted:
    # Convert input to DataFrame and handle numeric conversion
    input_df = pd.DataFrame([user_input])
    for col in feature_cols:
        try:
            input_df[col] = pd.to_numeric(input_df[col])
        except Exception:
            pass  # leave as is if not convertible

    # Predict
    try:
        pred = model.predict(input_df)[0]
        # Example: If you want to show "Exists" or "Does not exist" based on a threshold
        exists = "Exists" if pred > 50 else "Does not exist"
        st.markdown(
            f"""
            <div class="main-card">
                <span class="prediction-title">Prediction: <a style="color:#1976d2;font-weight:700;">{exists}</a></span>
                <br><br>
                <span class="prediction-label">🕒 Predicted {target_col}: <span class="prediction-value">{pred:.2e}</span></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Optionally, add CSV upload for batch predictions
with st.expander("📂 Batch Prediction (Upload CSV)"):
    uploaded = st.file_uploader("Upload CSV (must contain these feature columns):", type=["csv"])
    st.write("Required columns:", feature_cols)
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            X = df[feature_cols]
            preds = model.predict(X)
            out = df.copy()
            out["predicted_" + target_col] = preds
            st.success("Predictions ready")
            st.dataframe(out.head(200))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv, file_name="predictions.csv")