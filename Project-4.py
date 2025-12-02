import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


# --- 1. Data Loading and Model Training Function ---

@st.cache_resource
def train_churn_model(csv_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """Loads, preprocesses data, trains SVC model, and returns the trained pipeline."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: '{csv_path}' not found. Please download the dataset.")
        return None, None

    # Data Cleaning and Preprocessing

    # 1. Convert 'Total Charges' to numeric, forcing errors to NaN
    # Note: 'TotalCharges' is NOT used as a slider input, but is critical for training
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 2. Fill missing TotalCharges with the mean/median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    # 3. Handle 'No service' values in feature columns
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df[col] = df[col].replace('No internet service', 'No')

    # 4. Map target variable 'Churn' to 0 and 1
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 5. Select features and drop 'customerID'
    X = df.drop(columns=['Churn', 'customerID'])
    y = df['Churn']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    # Apply One-Hot Encoding to categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Get the final feature list after encoding
    feature_list = X_encoded.columns.tolist()

    # Split data (used here just to fit the scaler and model)
    X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Create a pipeline: Scale the data first, then apply SVC
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # SVC needs probability=True set for predict_proba
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Save components
    joblib.dump(pipeline, 'churn_svc_pipeline.pkl')
    joblib.dump(feature_list, 'churn_features.pkl')

    return pipeline, feature_list


# Load trained components
pipeline, feature_list = train_churn_model()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 4: Customer Churn Predictor")

st.title("👥 Day 4: Customer Churn Predictor")
st.subheader("Support Vector Classifier (SVC) with Preprocessing")
st.markdown("Predict the likelihood of a customer churning based on their service profile.")

if pipeline is not None:

    st.sidebar.header("Customer Profile")

    # --- 3. User Input Controls (Key Features) ---

    # 1. Numerical: Tenure (How long they've been a customer)
    tenure_input = st.sidebar.slider('Tenure (Months)', 1, 72, 24,
                                     help="Number of months the customer has stayed with the company.")

    # 2. Numerical: Monthly Charges
    monthly_charges_input = st.sidebar.slider('Monthly Charges ($)', 18.0, 118.0, 70.0)

    # 3. Categorical: Contract Type (Crucial predictor)
    contract_input = st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))

    # 4. Categorical: Internet Service
    internet_service_input = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))

    # 5. Categorical: Tech Support
    tech_support_input = st.sidebar.selectbox('Has Tech Support', ('Yes', 'No'))

    # 6. Placeholder for TotalCharges (Using median value for simplicity, as it's not a slider)
    # The training data already imputes this, but we need a value for the input
    total_charges_input = 2283.3  # A reasonable median/mean from the dataset

    # --- 4. Prepare Input for Prediction (FIXED LOGIC) ---

    # Create a base vector of zeros matching the model's expected feature set
    # This prevents NaNs by ensuring all one-hot encoded features not set are 0.
    input_df = pd.DataFrame(0, index=[0], columns=feature_list)

    # Map numerical inputs directly
    input_df['tenure'] = tenure_input
    input_df['MonthlyCharges'] = monthly_charges_input
    input_df['TotalCharges'] = total_charges_input  # Include the placeholder value

    # Map categorical inputs to the one-hot encoded columns (setting the value to 1)

    # Contract Type (Base is 'Month-to-month')
    if contract_input == 'One year':
        input_df['Contract_One year'] = 1
    elif contract_input == 'Two year':
        input_df['Contract_Two year'] = 1

    # Internet Service (Base is 'DSL')
    if internet_service_input == 'Fiber optic':
        input_df['InternetService_Fiber optic'] = 1
    elif internet_service_input == 'No':
        input_df['InternetService_No'] = 1

    # Tech Support (Base is 'No')
    if tech_support_input == 'Yes':
        input_df['TechSupport_Yes'] = 1

    # Include all other features from the dataset that are NOT sliders, using a default of 0
    # For example, Gender, Partner, Dependents, and other services (which are all encoded as 0 for the 'No'/'Female'/'No Phone Service' baseline)

    # The pipeline expects a NumPy array, so extract the values
    final_input = input_df[feature_list].values.reshape(1, -1)

    # --- 5. Prediction Logic ---

    # Predict the probability of Churn (1) using the pipeline
    churn_proba = pipeline.predict_proba(final_input)[0][1]

    # --- 6. Output Display ---

    st.write("### 📈 Churn Prediction:")

    if churn_proba >= 0.5:
        st.error(f"**HIGH CHURN RISK** ({churn_proba * 100:.2f}% Probability of Churn)")
    else:
        st.success(f"**LOW CHURN RISK** ({churn_proba * 100:.2f}% Probability of Churn)")

    st.markdown("---")
    st.write("### 🔍 Model Insights")
    st.info(
        "Algorithm: Support Vector Classifier (SVC). The model was trained using **Standard Scaling** and **One-Hot Encoding** to prepare the data.")