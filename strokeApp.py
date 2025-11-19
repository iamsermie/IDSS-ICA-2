# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load('stroke_random_forest_model.pkl')

model = load_model()

# -----------------------------
# App Title & Description
# -----------------------------
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠")
st.title("🧠 Stroke Risk Predictions System")
st.markdown("""
This app predicts the risk of stroke based on patient information using a trained machine learning model.
Please fill in the details below.
""")

st.sidebar.header("Patient Information")

# -----------------------------
# Input Fields
# -----------------------------
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female", "Other"))
    age = st.sidebar.slider("Age", 1, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
    heart_disease = st.sidebar.selectbox("Heart Disease", ("No", "Yes"))
    ever_married = st.sidebar.selectbox("Ever Married", ("Yes", "No"))
    work_type = st.sidebar.selectbox("Work Type", 
        ("Private", "Self-employed", "Govt_job", "children", "Never_worked"))
    residence_type = st.sidebar.selectbox("Residence Type", ("Urban", "Rural"))
    avg_glucose_level = st.sidebar.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
    bmi = st.sidebar.slider("BMI", 10.0, 70.0, 25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", 
        ("never smoked", "formerly smoked", "smokes", "Unknown"))

    # Convert to numeric
    data = {
        'gender': 1 if gender in ["Male", "Other"] else 0,
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'ever_married': 1 if ever_married == "Yes" else 0,
        'Residence_type': 1 if residence_type == "Urban" else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        # One-hot encoding matches what we did during training (drop_first=True)
        'work_Never_worked': 1 if work_type == "Never_worked" else 0,
        'work_Private': 1 if work_type == "Private" else 0,
        'work_Self-employed': 1 if work_type == "Self-employed" else 0,
        'work_children': 1 if work_type == "children" else 0,
        'smoking_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
        'smoking_never smoked': 1 if smoking_status == "never smoked" else 0,
        'smoking_smokes': 1 if smoking_status == "smokes" else 0,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Stroke Risk"):
    with st.spinner("Analyzing patient data..."):
        prediction_proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    
    risk_level = "🟢 LOW" if prediction_proba < 0.2 else "🟡 MODERATE" if prediction_proba < 0.5 else "🔴 HIGH"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stroke Probability", f"{prediction_proba:.1%}")
    with col2:
        st.metric("Risk Level", risk_level)

    if prediction == 1:
        st.error("⚠️ HIGH RISK: This patient has a significant risk of stroke!")
        st.info("🚑 Recommendation: Immediate medical evaluation advised.")
    else:
        st.success("✅ Low to moderate risk of stroke based on current data.")

    # Show input summary
    st.subheader("Patient Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    # Optional: Show feature importance or risk factors
    st.subheader("Key Risk Contributors")
    st.write("Age, glucose level, hypertension, and heart disease are usually the strongest predictors.")

else:
    st.info("👈 Please fill in the patient details in the sidebar and click 'Predict Stroke Risk'")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • Model: Random Forest (Trained on Healthcare Stroke Dataset)")