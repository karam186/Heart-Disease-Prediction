🫀 Heart Disease Prediction Using Machine Learning
📌 Project Overview
This project builds a machine learning model to predict the presence of heart disease using clinical patient data.

The goal is to assist in early risk identification using statistical and machine learning techniques.

📊 Dataset
Source: UCI Heart Disease Dataset

Total Records: 303

Features: 13 clinical features + 1 target variable

Target:

0 → No heart disease

1 → Heart disease present

No missing values were found in the dataset.

🔍 Exploratory Data Analysis
Key insights:

Chest pain type (cp) shows strong correlation with heart disease.

Maximum heart rate achieved (thalach) is positively correlated.

Exercise induced angina (exang), oldpeak, and number of major vessels (ca) are important predictors.

Cholesterol showed weak statistical correlation in this dataset.

Average patient age: 54.36 years.

🤖 Models Used
Logistic Regression (with feature scaling)

Random Forest Classifier

Logistic Regression performed better on this dataset.

📈 Model Performance
Accuracy: ~XX%

ROC-AUC Score: ~XX

Good recall for detecting disease cases

Logistic Regression generalized better due to small dataset size and near-linear feature relationships.

🏥 Business Impact
This model can help:

Identify high-risk patients early

Support preventive healthcare decisions

Reduce late-stage cardiac complications

🚀 How to Run
Clone repository

Create virtual environment

Install requirements

Run Streamlit app


📂 Project Structure
heart-disease-prediction/
│
├── data/
├── notebooks/
├── src/
│   └── heart_model.pkl
├── app.py
└── README.md