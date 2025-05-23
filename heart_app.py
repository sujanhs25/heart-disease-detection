import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Page title
st.title("Heart Disease Prediction with KNN")
st.write("Enter the patient data below to predict heart disease and view model insights.")

# Load model and dataset
model = joblib.load("knn_heart_model.joblib")
df = pd.read_csv("heart.csv")

# Standardize features
X = df.drop("target", axis=1)
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# User inputs
st.header("Patient Data Input")
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

input_data = pd.DataFrame([[
    age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]])

# Make prediction
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.success(f"Heart Disease Detected. (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.success(f"No Heart Disease Detected. (Confidence: {prob[0]*100:.2f}%)")

# Show graphs section
st.header("Model Performance Insights")

# Confusion matrix
st.subheader("Confusion Matrix")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

fig1, ax1 = plt.subplots()
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

# Accuracy vs. k plot
st.subheader("Accuracy vs. k (Number of Neighbors)")
k_values = list(range(1, 21))
accuracies = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_scaled, y, cv=5).mean() for k in k_values]

fig2, ax2 = plt.subplots()
ax2.plot(k_values, accuracies, marker="o")
ax2.set_xlabel("k (Number of Neighbors)")
ax2.set_ylabel("Accuracy")
ax2.set_title("Model Accuracy vs. k")
ax2.grid(True)
st.pyplot(fig2)
