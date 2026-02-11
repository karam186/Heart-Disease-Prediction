import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    text-align: center;
    color: #FF4B4B;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #1E222A;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("<h1>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("### AI-powered health risk assessment using Logistic Regression")
st.markdown("---")

# ---------------- LOAD MODEL ---------------- #
try:
    model = joblib.load("src/heart_model.pkl")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# ---------------- INPUT SECTION ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])

with col2:
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [1, 2, 3])

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("")

# Convert categorical inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Predict Risk"):

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

    st.markdown("---")
    st.subheader("🩺 Prediction Result")

    if prediction == 1:
         st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.metric("Risk Probability", f"{probability:.2%}")

st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression • Deployed on Streamlit Cloud")