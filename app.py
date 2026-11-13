import joblib
import pandas as pd
import streamlit as st

import churn_utils  # noqa: F401  (the saved pipeline needs clean() from this module)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 820px; padding-top: 2rem;}
.result-card {padding: 1.2rem 1.4rem; border-radius: 14px; margin: 0.5rem 0 1rem 0;}
.result-high {background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.5);}
.result-low  {background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.5);}
.result-card h3 {margin: 0 0 .3rem 0;}
.result-card p  {margin: 0; opacity: .85;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("📉 Customer Churn Predictor")
st.write("Enter a telecom customer's details to estimate how likely they are to leave the company.")

YES_NO = ["No", "Yes"]
ADDON = ["No", "Yes", "No internet service"]

with st.form("customer_form"):
    st.subheader("👤 Customer profile")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    senior = c2.selectbox("Senior citizen", YES_NO)
    tenure = c3.number_input("Tenure (months)", 0, 100, 12)
    partner = c1.selectbox("Has partner", YES_NO)
    dependents = c2.selectbox("Has dependents", YES_NO)

    st.subheader("📞 Services")
    c1, c2 = st.columns(2)
    phone = c1.selectbox("Phone service", ["Yes", "No"])
    lines = c2.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = c1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    c1, c2, c3 = st.columns(3)
    security = c1.selectbox("Online security", ADDON)
    backup = c2.selectbox("Online backup", ADDON)
    device = c3.selectbox("Device protection", ADDON)
    tech = c1.selectbox("Tech support", ADDON)
    tv = c2.selectbox("Streaming TV", ADDON)
    movies = c3.selectbox("Streaming movies", ADDON)

    st.subheader("💳 Contract & billing")
    c1, c2 = st.columns(2)
    contract = c1.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    c1, c2 = st.columns(2)
    monthly = c1.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=0.5)
    total = c2.number_input("Total charges ($)", 0.0, 10000.0, 840.0, step=10.0,
                            help="Roughly monthly charges × tenure.")

    submitted = st.form_submit_button("Predict churn risk", type="primary",
                                      use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device,
        "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])

    churn = int(model.predict(row)[0]) == 1
    score = float(model.predict_proba(row)[0][1])

    if churn:
        st.markdown('<div class="result-card result-high"><h3>⚠️ Likely to churn</h3>'
                    '<p>This customer shows a high risk of leaving. Consider a retention offer.</p></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card result-low"><h3>✅ Likely to stay</h3>'
                    '<p>This customer shows a low risk of leaving.</p></div>',
                    unsafe_allow_html=True)

    st.metric("Churn risk score", f"{score:.0%}")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption("The score is the share of the 15 most similar customers (in the training data) "
               "who churned. The model was trained on SMOTE-balanced data, so treat it as a "
               "relative risk score rather than an exact probability.")

st.divider()
st.caption("Model: SMOTE + K-Nearest Neighbors (k=15, Manhattan distance) trained on the Telco Customer Churn dataset.")
