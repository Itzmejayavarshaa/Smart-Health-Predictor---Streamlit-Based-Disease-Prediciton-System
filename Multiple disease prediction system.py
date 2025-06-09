# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:36:44 2025

@author: jayav
"""

import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Load the trained models
try:
    diabetes_model = joblib.load(
        r"C:\Users\jayav\OneDrive\Desktop\capstone project\saved models\diabetes_prediction_model.pkl")
    heart_model = joblib.load(
        r"C:\Users\jayav\OneDrive\Desktop\capstone project\saved models\heart_model.pkl")
    parkinson_model = joblib.load(
        r"C:\Users\jayav\OneDrive\Desktop\capstone project\saved models\parkinsons_model.pkl")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Smart Health Predictor',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson\'s Prediction'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    name = st.text_input("Enter Your Name")
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0)
    insulin = st.number_input('Insulin Level (µU/mL)', min_value=0)
    bmi = st.number_input('BMI (kg/m²)', min_value=0.0, format="%.2f")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f")
    age = st.number_input('Age (years)', min_value=1, step=1)

    if st.button('Predict Diabetes'):
        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(user_data)

        if prediction[0] == 1:
            st.error(f"{name}, you have a high risk of Diabetes. Consult a doctor.")
        else:
            st.success(f"{name}, you are unlikely to have Diabetes.")

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    name = st.text_input("Enter Your Name")
    age = st.number_input("Age", min_value=1, step=1)
    sex = st.radio('Sex', ['Female', 'Male'])
    sex = 0 if sex == 'Female' else 1

    cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3)
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0)
    chol = st.number_input('Serum Cholesterol (mg/dL)', min_value=0)
    fbs = st.radio('Fasting Blood Sugar > 120 mg/dL', ['False', 'True'])
    fbs = 1 if fbs == 'True' else 0

    restecg = st.number_input('Resting ECG Results (0-2)', min_value=0, max_value=2)
    thalach = st.number_input('Max Heart Rate Achieved (bpm)', min_value=0)
    exang = st.radio('Exercise Induced Angina', ['No', 'Yes'])
    exang = 1 if exang == 'Yes' else 0

    oldpeak = st.number_input('ST Depression Induced', min_value=0.0, format="%.2f")
    slope = st.number_input('Slope of ST Segment (0-2)', min_value=0, max_value=2)
    ca = st.number_input('Number of Major Vessels (0-4)', min_value=0, max_value=4)
    thal = st.number_input('Thalassemia Type (0-3)', min_value=0, max_value=3)

    if st.button('Predict Heart Disease'):
        user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                               thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(user_data)

        if prediction[0] == 1:
            st.error(f"{name}, you have a high risk of Heart Disease. Consult a doctor.")
        else:
            st.success(f"{name}, you are unlikely to have Heart Disease.")

# Parkinson's Disease Prediction Page
elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    name = st.text_input("Enter Your Name")
    age = st.number_input("Age", min_value=1, step=1)

    fo = st.number_input('MDVP:Fo(Hz)', min_value=80.0, max_value=300.0, format="%.3f")
    fhi = st.number_input('MDVP:Fhi(Hz)', min_value=100.0, max_value=600.0, format="%.3f")
    flo = st.number_input('MDVP:Flo(Hz)', min_value=50.0, max_value=300.0, format="%.3f")
    jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=0.1, format="%.6f")
    jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=0.1, format="%.6f")
    jitter_ddp = st.number_input('Jitter:DDP', min_value=0.0, max_value=0.1, format="%.6f")
    shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=0.3, format="%.6f")
    shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=3.0, format="%.6f")
    rap = st.number_input('MDVP:RAP', min_value=0.0, max_value=0.1, format="%.6f")
    ppq = st.number_input('MDVP:PPQ', min_value=0.0, max_value=0.1, format="%.6f")
    apq3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=0.2, format="%.6f")
    apq5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=0.2, format="%.6f")
    apq = st.number_input('MDVP:APQ', min_value=0.0, max_value=0.3, format="%.6f")
    dda = st.number_input('Shimmer:DDA', min_value=0.0, max_value=0.3, format="%.6f")
    nhr = st.number_input('NHR', min_value=0.0, max_value=0.5, format="%.6f")
    hnr = st.number_input('HNR', min_value=0.0, max_value=40.0, format="%.3f")
    rpde = st.number_input('RPDE', min_value=0.0, max_value=1.0, format="%.6f")
    dfa = st.number_input('DFA', min_value=0.0, max_value=1.0, format="%.6f")
    spread1 = st.number_input('Spread1', min_value=-10.0, max_value=5.0, format="%.6f")
    spread2 = st.number_input('Spread2', min_value=-5.0, max_value=2.0, format="%.6f")
    d2 = st.number_input('D2', min_value=0.0, max_value=5.0, format="%.6f")
    ppe = st.number_input('PPE', min_value=0.0, max_value=1.0, format="%.6f")

    if st.button("Predict Parkinson's"):
        user_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, jitter_ddp,
                               shimmer, shimmer_db, rap, ppq, apq3, apq5, apq, dda,
                               nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinson_model.predict(user_data)

        if prediction[0] == 1:
            st.error(f"{name}, you have a high risk of Parkinson's Disease. Consult a doctor.")
        else:
            st.success(f"{name}, you are unlikely to have Parkinson's Disease.")
