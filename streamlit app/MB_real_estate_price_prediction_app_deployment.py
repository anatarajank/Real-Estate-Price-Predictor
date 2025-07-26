import streamlit as st
import joblib # Import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree # Import BallTree for distance calculations
import datetime
import os # Import datetime for date handling

# Define the path to the saved model file (the one saved with joblib)
# Make sure to update this path to point to your joblib-saved .pkl file
# For demonstration in Colab, let's assume it's in the current directory or a specified path
# In a real Streamlit app, this path needs to be correct relative to where the script is run
script_dir = os.path.dirname(__file__)
model_filename = 'deployment_26072025.pkl'
model_path = os.path.join(script_dir, model_filename)

# --- Data/Coordinates needed for backend calculations ---
# Approximate coordinates for the Melbourne CBD (e.g., near Flinders Street Station)
CBD_LATITUDE = -37.8136
CBD_LONGITUDE = 144.9631

# Coordinates for Public Hospitals (extracted from the city_of_melbourne_landmarks_and_interest.csv)
# In a real app, you'd load this from a file or hardcode if small
# Based on previous notebook output (cell 6ea1fed8) for 'Public Hospital'
HOSPITAL_COORDINATES = np.deg2rad(np.array([
    [-37.8099387, 144.9623773], # The Royal Melbourne Hospital
    [-37.8147062, 144.9668536], # St Vincent's Hospital Melbourne
    [-37.8122569, 144.9717253], # Peter MacCallum Cancer Centre
    [-37.8112436, 144.9705748], # The Royal Victorian Eye and Ear Hospital
    [-37.8010845, 144.9647862], # Royal Children's Hospital
    [-37.8001166, 144.9654514], # The Royal Women's Hospital
    [-37.8053234, 144.9700038]  # Melbourne Private Hospital (although technically private, included in previous filtering)
]))

# Coordinates for Schools (extracted from the city_of_melbourne_landmarks_and_interest.csv)
# Based on previous notebook output (cell 2abb915d) for school_themes
SCHOOL_COORDINATES = np.deg2rad(np.array([
    [-37.8020953835112, 144.969406272035], # Carlton Gardens Primary School
    [-37.8342561018747, 144.976285045233], # Melbourne Grammar School
    [-37.7986737155551, 144.951065772876], # North Melbourne Primary School
    [-37.7959079757171, 144.970147317976], # Carlton Primary School
    [-37.8241137662047, 144.96933253996],  # University of Melbourne (VCA and Music)
    [-37.8030607125776, 144.966052836345], # University High School
    [-37.8210942696578, 144.962234721269], # RMIT University
    [-37.8102973141126, 144.963079797054], # University of Melbourne
    [-37.8105784600321, 144.953980764953], # Haileybury College - City Campus
    [-37.8173175278318, 144.966881882211], # Victorian College of the Arts Secondary School
    [-37.7931422209529, 144.96058052793],  # University of Melbourne (Burnley Campus)
    [-37.8085383502744, 144.965251880698], # Melbourne Central Catholic College (Parade College)
    [-37.8252018211194, 144.967704607226]  # Victorian College of the Arts
]))

# Create BallTree for hospitals and schools for efficient distance calculation
tree_hospitals = BallTree(HOSPITAL_COORDINATES, metric='haversine')
tree_schools = BallTree(SCHOOL_COORDINATES, metric='haversine')


# Load the trained model pipeline using joblib
try:
    # In Colab, the file is saved to a specific path. Adjust for deployment.
    # model_path = '/content/drive/MyDrive/MDS Deakin/Units/SIG720 - Machine Learning/Tasks/Task 5D/tuned_lightgbm_pipeline.pkl'
    pipeline = joblib.load(model_path)
    st.success("Model pipeline loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please check the path.")
    pipeline = None # Set pipeline to None if loading fails
except Exception as e:
    st.error(f"An error occurred during model loading: {e}")
    # Provide specific error message for scikit-learn/joblib version mismatch if possible
    # This error message is specific to scikit-learn version issues during loading
    if "No such file or directory" not in str(e) and "sklearn" in str(e):
         st.error("This error might indicate a mismatch between the scikit-learn version used to save the model and the version used to load it. Ensure your local scikit-learn version matches the one used when the model was saved.")
    pipeline = None # Set pipeline to None if loading fails


# Define the expected order of columns for the model input
# This order must match the training data's column order (excluding the target)
# This list includes BOTH user inputs AND derived features in the order the pipeline expects
original_cols_order = ['property_type', 'suburb', 'bathrooms', 'bedrooms', 'parking',
                       'num_garage', 'num_schools_5km', 'distance_to_cbd_km',
                       'distance_to_nearest_hospital_km', 'sale_year', 'sale_quarter',
                       'bathroom_bedroom_ratio', 'bedroom_bathroom_interaction']

# --- Sidebar Content ---
st.sidebar.markdown("## Instructions")
st.sidebar.markdown("""
Welcome to the Melbourne Housing Price Predictor!

Follow these steps to get a price prediction for a property in Brighton, Northcote, or Richmond:

1.  **Enter Property Details:** Use the input fields on the main page to provide information about the property, including:
    *   Suburb
    *   Property Type
    *   Number of Bedrooms
    *   Number of Bathrooms
    *   Latitude and Longitude (approximate location)
    *   Date Sold (select a date)
    *   Number of Parking Spaces
    *   Number of Garages
2.  **Click 'Predict Price':** Once you have entered all the details, click the "Predict Price" button.
3.  **View Prediction:** The predicted sale price for the property will be displayed below the button.

You can use the sample data provided below to test the predictor.
""")

st.sidebar.markdown("## Sample Data and Predictions")
st.sidebar.markdown("""
Here are a few sample properties from the test set and the prices predicted by the model:

| Instance ID | Date Sold  | Latitude   | Longitude  | Actual Price | Predicted Price | 
|-------------|------------|------------|------------|--------------|-----------------|
| 1281        | 22-11-2023 | -37.818948 | 145.013148 | \$670,000.00 | \$690,299.34    |
| 1171        | 28-01-2024 | -37.768565 | 144.999392 | \$565,000.00 | \$611,969.14    |
| 1650        | 05-05-2023 | -37.815269 | 145.008276 | \$690,000.00 | \$594,286.60    |

*Note: The model's prediction may differ from the actual price.*
""")

# --- Main App Content ---
# Center the title using markdown and HTML
st.markdown("<h1 style='text-align: center;'>Melbourne Housing Price Prediction</h1>", unsafe_allow_html=True)

# Add the image below the header and introductory text
image_url = "https://upload.wikimedia.org/wikipedia/commons/7/74/Melbourne_skyline_sor.jpg"
st.image(image_url, caption='Melbourne Skyline, Melbpal, CC BY-SA 4.0, via Wikimedia Commons', use_container_width=True)

st.markdown("<h3 style='text-align: center;'>Enter the property features to get a price prediction!</h3>", unsafe_allow_html=True)

# Create input widgets for the simplified user inputs
st.subheader("Property Details")

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    user_suburb = st.radio("Select Suburb:", ['Richmond', 'Northcote', 'Brighton'], horizontal=True, key="radio_suburb")
    user_property_type = st.radio("Select Property Type:", ['apartment', 'house', 'townhouse', 'unit', 'unitblock', 'villa'], horizontal=True, key="radio_property_type")
    user_bedrooms = st.slider("Number of Bedrooms:", min_value=1, max_value=26, value=2, step=1, key="slider_bedrooms")
    user_bathrooms = st.slider("Number of Bathrooms:", min_value=1, max_value=14, value=1, step=1, key="slider_bathrooms")


with col2:
    user_latitude = st.number_input("Latitude:", value=-37.81, format="%.6f", key="number_latitude")
    user_longitude = st.number_input("Longitude:", value=144.99, format="%.6f", key="number_longitude")
    user_date_sold = st.date_input("Date Sold:", datetime.date(2024, 1, 1), key="date_sold") # Default to a recent date
    user_parking = st.slider("Number of Parking Spaces:", min_value=1, max_value=14, value=1, step=1, key="slider_parking")
    # Changed num_garage to slider input based on min/max from dataset
    user_num_garage = st.slider("Number of Garages:", min_value=0, max_value=12, value=0, step=1, key="slider_num_garage")


# Create a button to trigger prediction
if st.button("Predict Price"):
    if pipeline:
        try:
            # 1. Collect user inputs
            user_input_data = {
                'property_type': user_property_type,
                'suburb': user_suburb,
                'bathrooms': float(user_bathrooms), # Ensure float type as in original data
                'bedrooms': float(user_bedrooms),   # Ensure float type
                'parking': float(user_parking),     # Ensure float type
                'num_garage': int(user_num_garage), # Ensure int type
                'latitude': float(user_latitude),   # Keep for calculations
                'longitude': float(user_longitude), # Keep for calculations
                'date_sold': user_date_sold         # Keep for calculations
            }

            # 2. Calculate derived features
            # Convert user lat/lon to radians for distance calculations
            user_coords_radians = np.deg2rad(np.array([[user_input_data['latitude'], user_input_data['longitude']]]))

            # Calculate distance to CBD
            cbd_coords_radians = np.deg2rad(np.array([[CBD_LATITUDE, CBD_LONGITUDE]]))
            distance_to_cbd = BallTree(cbd_coords_radians, metric='haversine').query(user_coords_radians, k=1)[0][0][0] * 6371 # Distance in km

            # Calculate distance to nearest hospital
            distance_to_nearest_hospital = tree_hospitals.query(user_coords_radians, k=1)[0][0][0] * 6371 # Distance in km

            # Calculate number of schools within 5km radius
            radius_km = 5
            radius_radians = radius_km / 6371 # 5km radius in radians
            num_schools_5km = len(tree_schools.query_radius(user_coords_radians, r=radius_radians)[0])

            # Extract year and quarter from date_sold
            sale_year = user_input_data['date_sold'].year
            sale_quarter = (user_input_data['date_sold'].month - 1) // 3 + 1 # Calculate quarter

            # Calculate bathroom/bedroom ratio and interaction
            bathrooms_val = user_input_data['bathrooms']
            bedrooms_val = user_input_data['bedrooms']
            # Handle division by zero for bathroom_bedroom_ratio if bedrooms is 0
            bathroom_bedroom_ratio = bathrooms_val / bedrooms_val if bedrooms_val != 0 else 0.0
            bedroom_bathroom_interaction = bedrooms_val * bathrooms_val

            # 3. Assemble all features into a DataFrame matching the pipeline's expected input order
            # Create a dictionary with ALL features (user inputs + derived)
            all_features_dict = {
                'property_type': user_input_data['property_type'],
                'suburb': user_input_data['suburb'],
                'bathrooms': user_input_data['bathrooms'],
                'bedrooms': user_input_data['bedrooms'],
                'parking': user_input_data['parking'],
                'num_garage': user_input_data['num_garage'],
                'num_schools_5km': num_schools_5km,
                'distance_to_cbd_km': distance_to_cbd,
                'distance_to_nearest_hospital_km': distance_to_nearest_hospital,
                'sale_year': sale_year,
                'sale_quarter': sale_quarter,
                'bathroom_bedroom_ratio': bathroom_bedroom_ratio,
                'bedroom_bathroom_interaction': bedroom_bathroom_interaction,
                # Latitude and longitude are not needed by the final pipeline, only for calculations
                # date_sold is not needed by the final pipeline
            }

            # Create DataFrame and reindex to match the exact original column order
            input_df = pd.DataFrame([all_features_dict])[original_cols_order]

            # Ensure correct dtypes for categorical features - crucial for the preprocessor
            for col in ['property_type', 'suburb']: # List your categorical columns
                input_df[col] = input_df[col].astype('category')


            # Make prediction using the loaded pipeline
            predicted_price = pipeline.predict(input_df)[0]

            st.subheader("Predicted Housing Price:")
            # Display the predicted price in bold green text
            st.markdown(f"**<span style='color:green;'>${predicted_price:,.2f}</span>**", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e) # Display the full traceback for debugging
    else:
        st.warning("Model pipeline not loaded. Cannot make prediction.")