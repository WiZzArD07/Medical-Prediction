import streamlit as st
import pandas as pd
import joblib

model = joblib.load("neural_network_model.pkl")



st.set_page_config(page_title="AI Healthcare Severity Predictor")
st.title("AI-Driven Healthcare Severity Predictor")

st.markdown("Enter Patient Vital Signs")

blood_pressure = st.number_input("Blood Pressure", min_value=50.0, max_value=250.0, step=1.0)
heart_rate = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, step=1.0)
oxygen_level = st.number_input("Oxygen Level (SpO2)", min_value=50.0, max_value=100.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)


if st.button("Predict Severity"):
    input_df = pd.DataFrame([[blood_pressure, heart_rate, oxygen_level, temperature]],
                            columns=["BloodPressure", "HeartRate", "Oxygen", "Temperature"])
    
    prediction = model.predict(input_df)[0]
  
    
    st.success(f"🩺 Predicted Severity: **{prediction}**")
