import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np


# --- 1. Data Loading and Model Training Function ---

@st.cache_resource
def train_fraud_detector(csv_path='creditcard_sample.csv'):
    """Loads data, scales it, trains the Isolation Forest model."""
    try:
        # NOTE: Using a highly simplified sample file for quick demonstration
        # Real fraud datasets are large and complex (V1-V28 features).
        # If your file is different, adjust the feature list!
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: '{csv_path}' not found. Please use a sample Credit Card Fraud dataset.")
        st.stop()  # Stop execution if data is missing

    # Assuming standardized features (V1-V4) and 'Amount' for a sample model
    # The Isolation Forest is UNSUPERVISED, so we drop the 'Class' (Target) column for training
    features = ['V1', 'V2', 'V3', 'Amount']

    # Use only a small subset of the data for fast training (e.g., first 10,000 rows)
    data = df[features].head(10000).copy()

    # 1. Standard Scaling (Important for Distance/Magnitude based models)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 2. Apply Isolation Forest (Unsupervised)
    # contamination: This parameter estimates the proportion of outliers in the data.
    # We set it to a very low value (e.g., 0.01%) reflecting expected fraud rate.

    # Train the model
    # NOTE: The contamination value should be based on real-world data;
    # we use 0.001 to ensure a few outliers are detected in the sample.
    model = IsolationForest(contamination=0.001, random_state=42)
    model.fit(data_scaled)

    # Make predictions on the training data to calculate anomaly count (-1=Anomaly, 1=Normal)
    predictions = model.predict(data_scaled)
    anomaly_count = np.sum(predictions == -1)

    # Save components
    joblib.dump(model, 'isolation_forest_model.pkl')

    return model, anomaly_count, scaler, features


# Load trained components
model, anomaly_count, scaler, feature_names = train_fraud_detector()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 8: Fraud Detector")

st.title("🚨 Day 8: Credit Card Fraud Detector")
st.subheader("Unsupervised Anomaly Detection (Isolation Forest)")
st.markdown("This model flags transactions that deviate significantly from normal customer behavior.")

if model is not None:

    st.sidebar.header("Test a Transaction")

    # --- 3. User Input Controls ---

    # Sliders for a simplified set of features
    v1_input = st.sidebar.slider('Feature V1 (Time-Normalized)', -5.0, 5.0, 0.0)
    v2_input = st.sidebar.slider('Feature V2 (Time-Normalized)', -5.0, 5.0, 0.0)
    v3_input = st.sidebar.slider('Feature V3 (Time-Normalized)', -5.0, 5.0, 0.0)
    amount_input = st.sidebar.slider('Transaction Amount ($)', 0.0, 2000.0, 50.0)

    st.sidebar.markdown("---")

    # --- 4. Prediction Logic ---

    # Input must be scaled before prediction
    input_raw = np.array([[v1_input, v2_input, v3_input, amount_input]])
    input_scaled = scaler.transform(input_raw)

    # Predict (-1: Anomaly, 1: Normal)
    prediction = model.predict(input_scaled)[0]

    # Get the anomaly score (lower score = higher probability of fraud)
    score = model.decision_function(input_scaled)[0]

    # --- 5. Output Display ---

    st.write("### 🔍 Model Results")
    st.metric(
        label="Total Anomalies Flagged in Training Set (0.001% contamination)",
        value=f"{anomaly_count}"
    )

    st.markdown("---")

    st.write("### 🚨 Transaction Status:")

    if prediction == -1:
        st.error(f"**ANOMALY DETECTED!** 🚫 (Score: {score:.4f})")
        st.warning("This transaction has highly unusual characteristics and should be flagged for review.")
    else:
        st.success(f"**NORMAL TRANSACTION** ✅ (Score: {score:.4f})")
        st.info("The transaction is consistent with historical patterns.")

    st.markdown("---")
    st.write("### 💡 Insight")
    st.caption(
        "Isolation Forest detects anomalies by calculating the path length required to isolate a point in a random tree. Shorter paths indicate a higher likelihood of being an outlier/fraud.")

