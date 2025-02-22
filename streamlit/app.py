import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved CatBoost model
with open("catboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit App
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction App 💰")
st.markdown("### Check your loan approval status instantly!")

# Sidebar inputs for user features
st.sidebar.header("📋 Enter Loan Applicant Details")

# Styling sidebar with sections
st.sidebar.markdown("---")
person_age = st.sidebar.number_input("📅 Person Age", min_value=18, max_value=100, step=1)
person_gender = st.sidebar.radio("👤 Person Gender", ["Male", "Female"], horizontal=True)
person_education = st.sidebar.selectbox("🎓 Highest Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.sidebar.number_input("💵 Annual Income ($)", min_value=0, step=1000)
person_emp_exp = st.sidebar.number_input("👔 Years of Employment Experience", min_value=0, max_value=50, step=1)
person_home_ownership = st.sidebar.selectbox("🏡 Home Ownership", ["Own", "Mortgage", "Rent", "Other"])
loan_amnt = st.sidebar.number_input("💰 Loan Amount Requested ($)", min_value=100, step=500)
loan_intent = st.sidebar.selectbox("🎯 Loan Intent", ["Education", "Medical", "Personal", "Home Improvement", "Debt Consolidation", "Business"])
loan_int_rate = st.sidebar.slider("📉 Loan Interest Rate (%)", min_value=0.0, max_value=30.0, step=0.1)
loan_percent_income = st.sidebar.slider("📊 Loan Amount as % of Income", min_value=0.0, max_value=1.0, step=0.01)
cb_person_cred_hist_length = st.sidebar.number_input("📜 Credit History Length (Years)", min_value=0, max_value=30, step=1)
credit_score = st.sidebar.number_input("💳 Credit Score", min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.sidebar.radio("📌 Previous Loan Defaults", ["No", "Yes"], horizontal=True)

st.sidebar.markdown("---")

# Encoding user inputs
person_gender = 1 if person_gender == "Male" else 0
previous_loan_defaults_on_file = 1 if previous_loan_defaults_on_file == "Yes" else 0
education_order = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "Doctorate": 5}
person_education = education_order[person_education]

# One-hot encoding for categorical variables
home_ownership_options = ["Mortgage", "Other", "Own", "Rent"]
loan_intent_options = ["Business", "Debt Consolidation", "Education", "Home Improvement", "Medical", "Personal"]

home_ownership_encoded = [1 if person_home_ownership == option else 0 for option in home_ownership_options[1:]]
loan_intent_encoded = [1 if loan_intent == option else 0 for option in loan_intent_options[1:]]

# Prepare input for model
features = [person_age, person_gender, person_education, person_income, person_emp_exp, loan_amnt, 
            loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, 
            previous_loan_defaults_on_file] + home_ownership_encoded + loan_intent_encoded

# Convert to NumPy array
input_data = np.array(features).reshape(1, -1)

# Centered Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Predict Loan Approval", key="predict_button", help="Click to check your loan approval status"):
        prediction = model.predict(input_data)
        result = "✅ Approved!" if prediction[0] == 1 else "❌ Rejected."
        st.markdown(f"## **{result}**")
        if prediction[0] == 1:
            st.success("🎉 Congratulations! Your loan is likely to be approved.")
        else:
            st.error("⚠️ Unfortunately, your loan application might be rejected.")
