import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing  # NEW Import!
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# --- 1. Data Loading and Model Training Function ---

@st.cache_resource
def train_california_model():
    """Loads California data, trains Linear Regression model, and saves components."""
    try:
        # Load the California Housing dataset
        housing = fetch_california_housing(as_frame=True)
        X = housing.data
        y = housing.target  # Median House Value (in hundreds of thousands of dollars)

    except Exception as e:
        st.error(f"Error loading California housing data: {e}")
        return None, None

    # Split the Data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save Model and Feature List
    joblib.dump(model, 'cali_lr_model.pkl')
    joblib.dump(X.columns.tolist(), 'cali_features.pkl')

    return model, X.columns.tolist()


# Load trained components
model, feature_list = train_california_model()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 2: California Housing Price Predictor")

st.title("🏠 Day 2: California Housing Price Predictor")
st.subheader("Linear Regression Model Deployment")
st.markdown("We're using the California Housing dataset to predict median house values.")

if model is not None:
    # --- 3. User Input Controls (Sidebar/Sliders) ---
    st.sidebar.header("Input District Parameters")

    # Define the inputs based on California Housing features

    # 1. Median Income (MedInc) - Most impactful feature
    med_inc = st.sidebar.slider('Median Income ($10k)', 0.5, 15.0, 3.5,
                                help="Median income of the district, in tens of thousands of dollars.")

    # 2. House Age (HouseAge)
    house_age = st.sidebar.slider('Median House Age', 1.0, 52.0, 30.0, help="Median age of the houses in the district.")

    # 3. Average Rooms (AveRooms)
    ave_rooms = st.sidebar.slider('Average Number of Rooms', 1.0, 15.0, 5.0,
                                  help="Average number of rooms per household.")

    # 4. Population (Population)
    population = st.sidebar.slider('District Population', 10.0, 35000.0, 1500.0)

    # 5. Longitude (Longitude) - Important for location
    longitude = st.sidebar.slider('Longitude', -125.0, -114.0, -120.0, help="A measure of how far west a district is.")

    # 6. Latitude (Latitude) - Important for location
    latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 35.0, help="A measure of how far north a district is.")

    # --- 4. Prepare Input for Prediction ---

    # The California dataset has 8 features. We need to pass all of them.
    # We set non-slider features to common or median values.

    input_dict = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': 1.10,  # Average number of bedrooms per household (Using dataset average)
        'Population': population,
        'AveOccup': 3.00,  # Average number of household members (Using dataset average)
        'Latitude': latitude,
        'Longitude': longitude
    }

    # Convert dictionary to DataFrame in the correct order
    input_df = pd.DataFrame([input_dict], columns=feature_list)

    # --- 5. Prediction Logic ---

    # Make prediction (Note: result is in hundreds of thousands of dollars)
    predicted_price_raw = model.predict(input_df)[0]

    # Convert prediction to actual USD value for better display
    predicted_price_usd = predicted_price_raw * 100000

    # --- 6. Output Display ---
    st.write("### 💰 Predicted Median House Value:")

    # Display the prediction, formatted as currency
    st.success(f"Estimated Price: **${predicted_price_usd:,.2f}**")

    st.markdown("---")
    st.write("### 🔍 Model Insights")
    st.info("Algorithm used: Linear Regression. The model uses 8 different features to make its prediction.")

