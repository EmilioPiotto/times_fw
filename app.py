import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

# Disable the warning about pyplot global use
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Time Series Prediction App")

uploaded_file = st.file_uploader("Upload your time series data", type=["csv"])
model_name = st.selectbox("Select Model", ["SimpleNN", "SimpleLSTM", "LinRegNN"])

st.header("Enter Time Series Manually")
manual_input = st.text_input("Enter time series data (comma-separated)")

# Customizing matplotlib style to match Streamlit's theme
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [10, 5]
background_color = st.get_option("theme.backgroundColor")
default_background_color = '#262730'  # Default dark background color
plt.rcParams['axes.facecolor'] = background_color if background_color else default_background_color
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'

if uploaded_file is not None:
    data = np.loadtxt(uploaded_file)
    data_list = [float(x) for x in data]
    data_list = data_list[-10:]
    st.write("Data Preview", data_list)
    st.write("Data length", len(data_list))

    if st.button("Predict"):
        payload = {
            "data": data_list,
            "model_name": model_name
        }
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write("Prediction", prediction)

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(range(len(data_list)), data_list, label='Original Time Series')
            ax.plot(range(len(data_list), len(data_list) + 1), prediction, 'ro', label='Predicted Value')
            ax.plot([len(data_list) - 1, len(data_list)], [data_list[-1], prediction], 'r--', label='Prediction Line')
            ax.set_xlabel('Time Steps', color='white')  # Setting axis labels color to white
            ax.set_ylabel('Value', color='white')      # Setting axis labels color to white
            ax.set_title('Time Series Prediction', color='white')  # Setting title color to white
            ax.legend()
            st.pyplot(fig)

        else:
            st.write("Error", response.json()["error"])

elif manual_input:
    data_list = [float(x.strip()) for x in manual_input.split(",")]
    st.write("Entered Time Series Data", data_list)
    st.write("Data length", len(data_list))

    if st.button("Predict"):
        payload = {
            "data": data_list,
            "model_name": model_name
        }
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write("Prediction", prediction)

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(range(len(data_list)), data_list, label='Original Time Series')
            ax.plot(range(len(data_list), len(data_list) + 1), prediction, 'ro', label='Predicted Value')
            ax.plot([len(data_list) - 1, len(data_list)], [data_list[-1], prediction], 'r--', label='Prediction Line')
            ax.set_xlabel('Time Steps', color='white')  # Setting axis labels color to white
            ax.set_ylabel('Value', color='white')      # Setting axis labels color to white
            ax.set_title('Time Series Prediction', color='white')  # Setting title color to white
            ax.legend()
            st.pyplot(fig)

        else:
            st.write("Error", response.json()["error"])
