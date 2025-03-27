# ✅ Streamlit App: Cleaned & Structured with Full Navigation & Model Integration

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import shap

# ----------------- Load model and SHAP explainer -----------------
model = joblib.load("models/best_model.pkl")
feature_names = ["LDHA", "LDHB", "LDHC", "LDHD"]
explainer = shap.Explainer(model, feature_names=feature_names)

# ----------------- App Configuration -----------------
st.set_page_config(page_title="LDH Cancer Stage Predictor", layout="wide")

# Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe6f0;
        background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Pink_ribbon.svg/1200px-Pink_ribbon.svg.png");
        background-repeat: no-repeat;
        background-position: top right;
        background-size: 120px;
    }
    section[data-testid="stSidebar"] {
        background-image: url("https://i.pinimg.com/736x/8e/f3/06/8ef3068685673b71ed9cc91b2cb3a259.jpg");
        background-size: cover;
    }
    h1, h2, h3, .stMarkdown {
        color: #cc3366;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- LDHA Risk Level -----------------
def interpret_ldha_stage(ldha):
    if ldha < 250:
        return "Stage I", "Low risk"
    elif 250 <= ldha < 400:
        return "Stage II", "Moderate risk"
    elif 400 <= ldha < 600:
        return "Stage III", "High risk"
    elif ldha >= 600:
        return "Stage IV", "Very High risk"

# ----------------- Language & Pages -----------------
st.sidebar.title("🌐 Language / Тіл / Язык")
lang = st.sidebar.selectbox("Choose your language", ["English", "Қазақша", "Русский"])

st.sidebar.subheader("🪾 Navigation")
page = st.sidebar.radio("Go to", [
    "Patient Data", "LDH Risk Tool", "Genetic Module",
    "Symptom Tracker", "LDHA Guide", "Education & FAQ", "Report Generator"
])

# ----------------- Pages -----------------
if page == "Patient Data":
    st.title("👤 Patient Clinical History")
    st.session_state['name'] = st.text_input("Patient Name", value=st.session_state.get('name', ''))
    st.session_state['age'] = st.number_input("Age", 0, 120, value=st.session_state.get('age', 0))
    st.session_state['gender'] = st.selectbox("Gender", ["Female", "Male", "Other"], index=["Female", "Male", "Other"].index(st.session_state.get('gender', "Female")))
    history = st.multiselect("Any clinical history?", ["Previous breast cancer", "BRCA mutation", "Family history", "Radiation exposure"])
    st.success("✅ Data saved. Now go to 'LDH Risk Tool' from the sidebar")

elif page == "LDH Risk Tool":
    st.title("🦢 LDH Expression Input")
    st.markdown("Input LDH expression levels from recent test results:")

    st.session_state['ldha'] = st.number_input("LDHA (mU/mL)", 0.0, 1000.0, step=1.0, value=st.session_state.get('ldha', 0.0))
    st.session_state['ldhb'] = st.number_input("LDHB (mU/mL)", 0.0, 1000.0, step=1.0, value=st.session_state.get('ldhb', 0.0))
    st.session_state['ldhc'] = st.number_input("LDHC (mU/mL)", 0.0, 1000.0, step=1.0, value=st.session_state.get('ldhc', 0.0))
    st.session_state['ldhd'] = st.number_input("LDHD (mU/mL)", 0.0, 1000.0, step=1.0, value=st.session_state.get('ldhd', 0.0))

if st.button("🔍 Predict"):
    input_data = np.array([[st.session_state['ldha'], st.session_state['ldhb'],
                            st.session_state['ldhc'], st.session_state['ldhd']]])
    
    pred = model.predict(input_data)[0]
    stage_labels = {0: "Early Stage", 1: "Late Stage", 2: "Mid Stage"}
    st.success(f"Predicted Stage (Model): {stage_labels.get(pred, 'Unknown')}")

    ldha_stage, risk_label = interpret_ldha_stage(st.session_state['ldha'])
    st.info(f"LDHA-based Assessment: {ldha_stage} — {risk_label}")

    if pred == 1:
        st.warning("⚠️ Late-stage indicators detected. Contact an oncologist immediately.")
    elif pred == 2:
        st.warning("⚠️ Moderate risk. Schedule further clinical tests.")
    else:
        st.success("🟢 Low risk. Maintain regular check-ups.")

    # 🔍 SHAP Explanation
    st.subheader("🔍 Feature Contribution Explanation")
    shap_values = explainer(input_data)
    shap_values_array = shap_values.values[0].flatten()
    feature_count = shap_values_array.shape[0]

    # Dynamically handle feature name mismatch
    if len(feature_names) != feature_count:
        feature_names = [f"Feature {i+1}" for i in range(feature_count)]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values_array
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.session_state["shap_df"] = shap_df
    st.subheader("Top Feature Impact (SHAP values)")
    st.bar_chart(shap_df.set_index("Feature"))



elif page == "Genetic Module":
    st.title("🧬 Genetic & Risk Factor Information")

    if "age" not in st.session_state:
        st.session_state["age"] = 30
    if "family_history" not in st.session_state:
        st.session_state["family_history"] = "No"
    if "lifestyle" not in st.session_state:
        st.session_state["lifestyle"] = []

    st.session_state["age"] = st.number_input("Age", 0, 120, value=st.session_state["age"])
    st.session_state["family_history"] = st.radio(
        "Family History of Breast Cancer?", ["Yes", "No"], index=["Yes", "No"].index(st.session_state["family_history"])
    )
    st.session_state["lifestyle"] = st.multiselect(
        "Lifestyle Risk Factors", ["Alcohol use", "Smoking", "Obesity", "Lack of exercise"],
        default=st.session_state["lifestyle"]
    )


elif page == "Symptom Tracker":
    st.title("📋 Symptom Tracker")
    st.slider("Fatigue Level", 0, 10, 0)
    st.slider("Pain Level", 0, 10, 0)
    st.checkbox("Unintended Weight Loss")
    st.checkbox("Night Sweats")

elif page == "LDHA Guide":
    st.title("📊 Clinical LDHA Threshold Interpretation")
    st.markdown("""
    | LDHA Range (mU/mL) | Cancer Stage | Avg Tumor Size (cm) | p-value | Risk |
    |---------------------|--------------|----------------------|---------|------|
    | <250                | Stage I      | 2.1                  | 0.05    | Low   |
    | 250–400             | Stage II     | 2.8                  | 0.03    | Moderate |
    | 400–600             | Stage III    | 4.2                  | 0.01    | High  |
    | >600                | Stage IV     | 5.5                  | 0.001   | Very High |
    """)

elif page == "Education & FAQ":
    st.title("📘 LDH, Cancer & Biomarkers")
    st.markdown("""
    - **LDH (Lactate Dehydrogenase)**: Enzyme linked to tumor metabolism
    - **LDHA**: Indicates how advanced breast cancer might be
    - **BRCA mutation**: Hereditary gene risk
    - **Stage I–IV**: Represents progression severity

    **FAQs**
    - ❓ What is this tool? → A predictive model, not a clinical diagnosis
    - ⏳ When to use it? → Every 3–6 months post-treatment
    - ☎️ What to do if high? → Contact oncology professionals
    """)

elif page == "Report Generator":
    st.title("📄 Personalized Risk Report Generator")
    if st.button("📄 Download PDF Report"):
        name = st.session_state.get('name', 'Unknown')
        age = st.session_state.get('age', 'Unknown')
        gender = st.session_state.get('gender', 'Unknown')
        ldha = st.session_state.get('ldha', 0.0)
        ldhb = st.session_state.get('ldhb', 0.0)
        ldhc = st.session_state.get('ldhc', 0.0)
        ldhd = st.session_state.get('ldhd', 0.0)
        ldha_stage, risk_label = interpret_ldha_stage(ldha)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(51, 51, 51)
        pdf.cell(200, 10, txt="LDH Risk Assessment Report", ln=1, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=1)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=1)
        pdf.cell(200, 10, txt=f"Gender: {gender}", ln=1)
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"LDHA: {ldha} mU/mL", ln=1)
        pdf.cell(200, 10, txt=f"LDHB: {ldhb} mU/mL", ln=1)
        pdf.cell(200, 10, txt=f"LDHC: {ldhc} mU/mL", ln=1)
        pdf.cell(200, 10, txt=f"LDHD: {ldhd} mU/mL", ln=1)
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"LDHA Stage: {ldha_stage}", ln=1)
        pdf.cell(200, 10, txt=f"Risk Assessment: {risk_label}", ln=1)
        pdf.ln(5)
        pdf.ln(5)
        pdf.set_text_color(0, 0, 128)
        pdf.cell(200, 10, txt="SHAP Feature Importances:", ln=1)
        for row in st.session_state.get("shap_df", pd.DataFrame()).itertuples():
            pdf.cell(200, 10, txt=f"{row.Feature}: {row._2:.3f}", ln=1)
        pdf.set_text_color(150, 0, 0)
        pdf.multi_cell(0, 10, txt="Disclaimer: This report is not a clinical diagnosis. Always consult a medical professional.")

        filename = f"LDH_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        with open(filename, "rb") as f:
            st.download_button("📅 Download PDF", f.read(), file_name=filename)
