# ldh-stage-predictor
AI-driven breast cancer stage prediction using LDH isoenzymes and SHAP explainability

This Streamlit application predicts breast cancer stage based on LDH isoenzyme levels (LDHA, LDHB, LDHC, LDHD) using a trained RandomForestClassifier model. It integrates explainable AI (SHAP), multi-language support, and PDF report generation.

## Features

- LDH Expression Input
- Predictive Model (Random Forest)
- SHAP Explanation for each prediction
- Symptom Tracker, Genetic Risk Module
- Multilingual UI: English, Russian, Kazakh
- PDF Report Generator

## Files

- `ldh_stage_predictor_app.py` — Main Streamlit app
- `LDH_outputs/best_model.pkl` — Trained classifier
- `requirements.txt` — Dependencies

##  Example Usage

1. Input sample values:  
   - LDHA: `380`, LDHB: `450`, LDHC: `210`, LDHD: `310`
2. Model outputs: **Stage III** (High risk)
3. SHAP explains feature importance visually
4. Downloadable personalized PDF report

## Citation

If used in publications, please cite SHAP and Scikit-learn as appropriate.
