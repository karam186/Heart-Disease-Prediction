import streamlit as st
import pandas as pd
import joblib

st.title("🫀 Heart Disease Risk Predictor")

# Debug message
st.write("App Started Successfully")

# Load trained model safely
try:
    model = joblib.load("src/heart_model.pkl")
    st.write("Model Loaded Successfully")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

st.write("Enter patient details below:")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
restecg = st.selectbox("Resting ECG (0–2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 10.0)
slope = st.selectbox("Slope (0–2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0,1,2,3,4])
thal = st.selectbox("Thal (1–3)", [1,2,3])

if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]], columns=[
        'age','sex','cp','trestbps','chol','fbs',
        'restecg','thalach','exang','oldpeak',
        'slope','ca','thal'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {probability:.2f})")
