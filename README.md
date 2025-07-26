# Melbourne Housing Price Predictor

**Author: Aravindan Natarajan**

**Attribution: CC-BY-4.0 International**

This repository contains a Streamlit web application for predicting housing prices in selected suburbs of Melbourne (Brighton, Northcote, and Richmond). The application utilizes a tuned LightGBM regression model trained on scraped real estate data.

## Project Overview

The goal of this project is to develop and deploy a regression model to predict housing prices. The process involved:

1.  **Data Acquisition:** Scraping real estate data from realestate.com.au for the top three chosen suburbs.
2.  **Data Preprocessing & Feature Engineering:** Cleaning the raw data, handling missing values, and creating new features such as distances to key locations (CBD, hospitals, schools) and temporal features.
3.  **Model Development & Evaluation:** Building and evaluating multiple regression models using cross-validation.
4.  **Model Tuning & Selection:** Hyperparameter tuning the best-performing model (LightGBM Regressor).
5.  **Model Deployment:** Saving the trained model pipeline and developing a simple web application for interactive price prediction.

This Streamlit app is the final deployment stage, allowing users to input property details and get an estimated sale price.

## Features

*   User-friendly interface to input property characteristics.
*   Predicts housing prices using a pre-trained LightGBM model.
*   Performs necessary feature engineering on user input in the backend.
*   Includes instructions and sample data in a sidebar.

## Setup and Running the App

To set up and run the Streamlit application locally, you will need to install the necessary dependencies and run the Python script.

### Prerequisites

*   Python 3.7+

### Dependencies

The required Python libraries are listed in the `requirements.txt` file. These typically include:

*   `streamlit`
*   `scikit-learn`
*   `pandas`
*   `numpy`
*   `scipy`
*   `lightgbm`
*   `joblib`

You will also need the saved model file (`tuned_lightgbm_pipeline.pkl`) in the correct paths relative to your Streamlit application script.

### Running the Application

Once the prerequisites and dependencies are met, you can run the application by executing the Streamlit script from your terminal.

## Usage

1.  Open the Streamlit application in your web browser (it will typically open automatically after running the script). [Public Link](https://melbourne-real-estate-price-predictor.streamlit.app/)
2.  Use the input fields on the main page to enter the details of the property you want to get a prediction for.
3.  Refer to the sidebar for instructions on how to use the app and view sample data and predictions.
4.  Click the "Predict Price" button to see the estimated housing price.

## Data Source

The housing data used for training the model was scraped from realestate.com.au for the suburbs of Brighton, Northcote, and Richmond. The places of interest data was obtained from the City of Melbourne dataset.
