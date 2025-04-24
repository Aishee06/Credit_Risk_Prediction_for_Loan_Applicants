import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

@st.cache_resource
def load_model_and_scaler():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

st.markdown("""
    <style>
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.2em;
        font-size: 16px;
    }
    .stSlider > div { color: #2E8B57; }
    .stSelectbox > div > div { color: #2E8B57; }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2E8B57;'>💳 Credit Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>🧠 Let data decide who gets credit!</p>",
    unsafe_allow_html=True
)
st.markdown("---")

st.header("📋 Enter Applicant Details")

def user_input():
    with st.expander("👤 Personal Information", expanded=True):
        age = st.slider("Age", 18, 75, 30)
        sex = st.selectbox("Gender", ["Female", "Male"])
        job = st.selectbox("Employment Type", [
            "0 - Unskilled and Non-Resident",
            "1 - Unskilled and Resident",
            "2 - Skilled",
            "3 - Highly Skilled"
        ])
        housing = st.selectbox("Housing Status", ["own", "rent", "free"])

    with st.expander("💳 Financial Information", expanded=True):
        saving_acc = st.selectbox("Savings Account", ["little", "moderate", "quite rich", "rich", "Not Applicable"])
        checking_acc = st.selectbox("Checking Account", ["little", "moderate", "rich", "Not Applicable"])
        credit_amount = st.number_input("Credit Amount (€)", 250, 20000, 1500)
        duration = st.slider("Loan Duration (Months)", 4, 72, 24)
        purpose = st.selectbox("Loan Purpose", [
            "radio/TV", "education", "furniture/equipment", "car",
            "business", "domestic appliance", "vacation", "repairs", "others"
        ])

    sex_val = 0 if sex == "Female" else 1
    job_val = int(job.split(" - ")[0])

    return pd.DataFrame([{
        "Age": age,
        "Sex": sex_val,
        "Job": job_val,
        "Housing": housing,
        "Saving accounts": saving_acc,
        "Checking account": checking_acc,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }])

input_df = user_input()

categorical_cols = ["Housing", "Saving accounts", "Checking account", "Purpose"]

def encode_input(df):
    mappings = {
        "Housing": {"own": 0, "free": 1, "rent": 2},
        "Saving accounts": {"little": 0, "moderate": 1, "quite rich": 2, "rich": 3, "Not Applicable": 4},
        "Checking account": {"little": 0, "moderate": 1, "rich": 2, "Not Applicable": 3},
        "Purpose": {
            "radio/TV": 0, "education": 1, "furniture/equipment": 2,
            "car": 3, "business": 4, "domestic appliance": 5,
            "repairs": 6, "vacation": 7, "others": 8
        }
    }
    for col in categorical_cols:
        df[col] = df[col].map(mappings[col])
    return df

input_df_encoded = encode_input(input_df)

st.markdown("---")
if st.button("🚀 Run Credit Risk Assessment"):
    with st.spinner("Analyzing credit profile..."):
        scaled_input = scaler.transform(input_df_encoded)
        prediction = model.predict(scaled_input)[0]

    st.subheader("📊 Prediction Outcome")
    if prediction == 0:
        st.markdown("<h3 style='color:green;'>✅ Good Credit Risk</h3>", unsafe_allow_html=True)
        st.success("They’re likely to repay on time. You can proceed with confidence! 💼")
    else:
        st.markdown("<h3 style='color:red;'>❌ Bad Credit Risk</h3>", unsafe_allow_html=True)
        st.warning("There may be repayment issues. Please consider further evaluation. 📉")

st.markdown("---")