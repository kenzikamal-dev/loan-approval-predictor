import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# ---------------- LOAD MODEL & COLUMNS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(BASE_DIR, "models", "loan_model.pkl")
columns_path = os.path.join(BASE_DIR, "models", "columns.pkl")

model = joblib.load(model_path)
columns = joblib.load(columns_path)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval AI System",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🏦 Smart Loan Approval System")
st.write("AI-powered loan eligibility checker")

st.markdown("---")

# ---------------- INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

    married = st.selectbox("Married", ["No", "Yes"])

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

with col2:
    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0
    )

st.markdown("---")

# ---------------- INPUT DATA ----------------
input_dict = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Loan Status"):

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )

    # Predict
    prediction = model.predict(input_df)

    # Predict probability
    probability = model.predict_proba(input_df)

    approval_probability = probability[0][1] * 100
    rejection_probability = probability[0][0] * 100

    # ---------------- RESULTS ----------------
    st.markdown("---")

    st.subheader("📊 Prediction Results")

    # Progress bar
    st.progress(int(approval_probability))

    # Probability display
    st.write(
        f"✅ Approval Probability: "
        f"**{approval_probability:.2f}%**"
    )

    st.write(
        f"❌ Rejection Probability: "
        f"**{rejection_probability:.2f}%**"
    )

    # ---------------- RISK LEVEL ----------------
    if approval_probability >= 75:
        st.success("🟢 Low Risk Applicant")

    elif approval_probability >= 40:
        st.warning("🟡 Medium Risk Applicant")

    else:
        st.error("🔴 High Risk Applicant")

    st.markdown("---")

    # ---------------- EXPLAINABLE AI ----------------
    if prediction[0] == 1:

        st.success("🎉 Loan Approved")

        explanation = []

        if credit_history == 1.0:
            explanation.append(
                "✔ Strong credit history"
            )

        if applicant_income >= 5000:
            explanation.append(
                "✔ Good applicant income"
            )

        if coapplicant_income > 0:
            explanation.append(
                "✔ Additional coapplicant income support"
            )

        if loan_amount < 200:
            explanation.append(
                "✔ Reasonable loan amount"
            )

        if not explanation:
            explanation.append(
                "✔ Overall applicant profile is acceptable"
            )

        st.info(
            "AI Explanation:\n\n"
            + "\n".join(explanation)
        )

    else:

        st.error("❌ Loan Rejected")

        explanation = []

        if credit_history == 0.0:
            explanation.append(
                "❌ Poor credit history"
            )

        if applicant_income < 3000:
            explanation.append(
                "❌ Low applicant income"
            )

        if loan_amount > 300:
            explanation.append(
                "❌ High loan amount requested"
            )

        if coapplicant_income == 0:
            explanation.append(
                "❌ No coapplicant income support"
            )

        if not explanation:
            explanation.append(
                "❌ High overall applicant risk"
            )

        st.warning(
            "AI Explanation:\n\n"
            + "\n".join(explanation)
        )

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "Built with Machine Learning • "
    "Loan Approval AI System • "
    "Author: Kamalkenzi"
)