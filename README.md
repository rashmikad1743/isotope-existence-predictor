# Isotope Existence Predictor

Predicts isotope existence (binary) from provided features using a trained ML model and offers a Streamlit UI.

## Structure
- `train.py` — training script (produces `artifact/model.pkl`)
- `app.py` — Streamlit app to upload CSV and get predictions
- `requirements.txt`
- `artifact/model.pkl` — output after running `train.py`

## Quick start (local)
1. Put your dataset `final_corrected_isotope_data.csv` in project root.
2. Create venv & install:
