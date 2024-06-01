# Time Series Prediction App

## Overview
This project provides a web application for time series prediction using various machine learning models. The frontend is built with Streamlit, and the backend is built with FastAPI.

## Project Structure
- `app.py`: Streamlit frontend.
- `backend/`: FastAPI backend and models.
- `backend/models/`: Pre-trained model weights.
- `data/`: Sample data files.
- `tests/`: Unit tests.

## Clone and run from IDE
source .venv/Scripts/activate
1. uvicorn backend.main:app --reload
2. streamlit run app.py


## Setup and Installation

### Backend
1. Navigate to the `backend` directory:
    ```sh
    cd backend
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```

### Frontend
1. Install Streamlit and other dependencies:
    ```sh
    pip install streamlit requests pandas
    ```

2. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Usage
1. Upload your time series data via the Streamlit interface.
2. Select a model to use for prediction.
3. View the predictions displayed on the Streamlit app.

## Testing
To run the tests, navigate to the project root and use `pytest`:
```sh
pytest tests/

