# ⚡ Energy Theft Detection System

This project is a Machine Learning–based system designed to detect **abnormal electricity usage patterns**, which may indicate **energy theft**.  
The model is deployed using **Gradio** so that users can interact with it in real time.

---

## 📌 Problem Statement
Electricity theft leads to significant financial losses and grid instability.  
This project aims to classify energy consumption as **Normal** or **Abnormal** based on user and usage parameters.

---

## 🧠 Machine Learning Model
- Algorithm: Logistic Regression
- Accuracy: **Above 90%**
- Type: Binary Classification (Normal / Abnormal)

---

## 📊 Features Used
- Age
- Energy Consumption
- Location
- Time of Use
- Previous Bills
- Average Temperature
- Payment Method
- Consumption Type

---

## 🔧 Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- Gradio
- Joblib

---

## 🚀 How It Works
1. User enters energy usage details.
2. Data is preprocessed using saved scaler and encoder.
3. Trained ML model predicts:
   - **Normal Usage**
   - **Abnormal Usage (Potential Theft)**
4. Output is displayed with confidence score.

---

## 🖥️ Deployment
The application is deployed on **Hugging Face Spaces** using Gradio for interactive predictions.

---

## 📁 Project Files
- `app.py` – Gradio application
- `model.pkl` – Trained ML model
- `scaler.pkl` – Feature scaler
- `encoder.pkl` – Categorical encoder
- `requirements.txt` – Required dependencies

---

## ✅ Output
- **Normal** – Legitimate energy usage
- **Abnormal** – Possible electricity theft

---

## 📌 Author
Energy Theft Detection Project  
Built for academic and demonstration purposes.
