import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


# --- 1. Data Loading and Model Training Function ---

@st.cache_resource
def train_adaboost_model(csv_path='diabetes.csv'):
    """Loads data, scales it, trains the AdaBoost pipeline."""
    try:
        # Assuming the CSV has features and a final 'Outcome' column (1=Diabetic, 0=Non-diabetic)
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: '{csv_path}' not found. Please download the Pima Indians Diabetes dataset.")
        st.stop()

    # Standard Pima dataset columns
    try:
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
    except KeyError:
        st.error("Error: Target column 'Outcome' not found. Please check your CSV.")
        st.stop()

    # Data Cleaning: Replace 0s (missing values) in key columns with the mean (simple imputation)
    cols_to_impute = ['Glucose', 'BloodPressure', 'BMI']
    for col in cols_to_impute:
        df[col] = df[col].replace(0, df[col].mean())

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    feature_list = X.columns.tolist()

    # Create the AdaBoost pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # Use a shallow Decision Tree (max_depth=1) as the weak base estimator
        ('ada', AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),  # Base estimator is a Decision Stump
            n_estimators=100,  # Number of weak learners
            random_state=42
        ))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Save components
    joblib.dump(pipeline, 'adaboost_diabetes_pipeline.pkl')

    return pipeline, feature_list


# Load trained components
pipeline, feature_list = train_adaboost_model()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 9: Diabetic Risk Predictor")

st.title("🩺 Day 9: Diabetic Risk Predictor")
st.subheader("AdaBoost Ensemble Classifier")
st.markdown("Predict the likelihood of a patient developing diabetes based on key health indicators.")

if pipeline is not None:

    st.sidebar.header("Patient Health Indicators")

    # --- 3. User Input Controls ---

    # 1. Glucose
    glucose = st.sidebar.slider('Glucose Concentration (mg/dL)', 0, 200, 120)

    # 2. BMI
    bmi = st.sidebar.slider('BMI', 10.0, 60.0, 32.0)

    # 3. Blood Pressure
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 40, 120, 72)

    # 4. Age
    age = st.sidebar.slider('Age', 21, 81, 35)

    # --- 4. Prepare Input for Prediction ---

    # Create an input DataFrame matching the feature_list (CRITICAL for scaling/prediction)
    input_df = pd.DataFrame(np.zeros(len(feature_list)).reshape(1, -1), columns=feature_list)

    # NOTE: You must include all features in the original dataset for accurate scaling,
    # even if you don't use sliders for them. We will use median values for simplicity
    # for features not controlled by sliders. (e.g., Pregnancies, Insulin, etc.)

    # Map user inputs
    input_df['Glucose'] = glucose
    input_df['BMI'] = bmi
    input_df['BloodPressure'] = bp
    input_df['Age'] = age

    # Use mean/median placeholders for features without sliders (you'd need to calculate these)
    input_df['Pregnancies'] = 3
    input_df['SkinThickness'] = 29
    input_df['Insulin'] = 155
    input_df['DiabetesPedigreeFunction'] = 0.5

    # --- 5. Prediction Logic ---

    # Predict the probability of Outcome=1 (Diabetic)
    risk_proba = pipeline.predict_proba(input_df)[0][1]
    prediction = pipeline.predict(input_df)[0]

    # --- 6. Output Display ---

    st.write("### 📈 Diabetes Risk Prediction:")

    if prediction == 1:
        st.error(f"**HIGH RISK: DIABETIC** 🔴 (Probability: {risk_proba * 100:.2f}%)")
    else:
        st.success(f"**LOW RISK: NON-DIABETIC** 🟢 (Probability: {(1 - risk_proba) * 100:.2f}%)")

    st.markdown("---")
    st.write("### 🔍 Model Insights")
    st.info("Algorithm: AdaBoost (Adaptive Boosting). The model focuses sequentially on the 'hardest' cases.")

