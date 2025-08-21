import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sidebar - info
st.sidebar.title("ℹ️ About this App")
st.sidebar.info(
    "This app predicts whether a patient is diabetic or not "
    "based on health parameters (Glucose, BMI, Age, etc.). "
    "\n\n⚡ Built with Streamlit & Random Forest."
)

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("🩺 Diabetes Prediction System")

# =======================
# 1. Dataset Overview
# =======================
st.subheader("📊 Dataset Overview")
df = pd.read_csv("data/diabetes.csv")
st.write(df.head())
st.info(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

# Heatmap
st.subheader("🔍 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# =======================
# 2. Model Accuracy
# =======================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.sidebar.success(f"📈 Model Accuracy: {acc*100:.2f}%")

st.markdown("---")

# =======================
# 3. Prediction Section
# =======================
st.subheader("🤖 Predict Diabetes")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely Diabetic")
    else:
        st.success("✅ The patient is Not Diabetic")
