from fastapi import FastAPI, Request 
from fastapi.templating import Jinja2Templates    # Way to render HTML templates using Jinja2
from pydantic import BaseModel   # Input schema. Data validation, parsing, and serialization.
import streamlit as st
import numpy as np
import uvicorn
import pickle
import pandas as pd

# Load saved model, scaler, and encoder
model = pickle.load(open("best_model.pkl", 'rb'))

# Streamlit UI
st.title("🏥Medical Insurance Cost Predictor")
st.write("Predict your estimated medical insurance charges based on personal and lifestyle factors.")

# User Inputs
age = st.number_input("Enter Age", min_value=0, max_value=100, step=1)
sex = st.selectbox("Select Gender", ["male", "female"])
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Do you smoke?", ["yes", "no"])
region = st.selectbox("Select Region", ["northeast", "northwest", "southeast", "southwest"])


# Prediction
if st.button("Predict Insurance Cost"):
    # Step 1: Create input DataFrame with same column names used during training
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Step 2: Encode categorical columns 
    input_df['sex'] = input_df['sex'].map({'male':1, 'female':0})
    input_df['smoker'] = input_df['smoker'].map({"yes":1,'no':0})
    input_df['region'] = input_df['region'].map({'southeast':0,"southwest":1,'northwest':2,"northeast":3})


    # Step 3: Scale the entire dataset
    num_cols = ['age','sex','bmi','children','smoker','region']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Step 4: Predict using the trained model
    prediction = model.predict(input_df)

    # Step 5: Display result
    st.success(f"💰 Estimated Annual Insurance Charge: **${prediction[0]:,.2f}**")