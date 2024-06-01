import streamlit as st
import requests
import pandas as pd
import numpy as np
import json

st.title("Time Series Prediction App")

uploaded_file = st.file_uploader("Upload your time series data", type=["csv"])
model_name = st.selectbox("Select Model", ["SimpleNN", "SimpleLSTM", "LinRegNN"])

st.header("Enter Time Series Manually")
manual_input = st.text_input("Enter time series data (comma-separated)")


if uploaded_file is not None:
    data = np.loadtxt(uploaded_file)
    data_list = [float(x) for x in data]
    data_list = data_list[-10:]
    st.write("Data Preview", data_list)
    st.write("Data lenght", len(data_list))


    if st.button("Predict"):
        payload = {
            "data": data_list,
            "model_name": model_name
        }
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write("Prediction", prediction)
        else:
            st.write("Error", response.json()["error"])

elif manual_input:
    data_list = [float(x.strip()) for x in manual_input.split(",")]
    st.write("Entered Time Series Data", data_list)
    st.write("Data lenght", len(data_list))


    if st.button("Predict"):
        payload = {
            "data": data_list,
            "model_name": model_name
        }
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write("Prediction", prediction)
        else:
            st.write("Error", response.json()["error"])
