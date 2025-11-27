import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib  # Used to save and load the trained model


# --- 1. Model Training Function ---
# We train the model once and save it to avoid retraining every time the app runs
@st.cache_resource  # Caches the function's output to speed up the app
def train_and_save_model():
    """Loads Iris data, trains the KNN model, and saves it."""
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data (optional for this simple app, but good practice)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Save the model
    joblib.dump(knn, 'knn_iris_model.pkl')
    return iris.target_names


# --- Load Model and Target Names ---
target_names = train_and_save_model()
# Load the model back from the saved file
knn_model = joblib.load('knn_iris_model.pkl')

# --- 2. Streamlit Application UI ---
st.set_page_config(page_title="Day 1: Iris Classifier")

st.title("🌸 Day 1: Iris Flower Species Predictor")
st.markdown("---")
st.subheader("K-Nearest Neighbors (KNN) Model")

st.markdown(
    """
    **Instructions:** Adjust the sliders below to input the measurements 
    of a new Iris flower. The model will predict its species in real-time.
    """
)

# --- 3. User Input Sliders (Sidebar) ---
st.sidebar.header("Input Flower Measurements (cm)")


# Define the acceptable range for each feature based on the Iris dataset
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

    data = {
        'Sepal_Length': sepal_length,
        'Sepal_Width': sepal_width,
        'Petal_Length': petal_length,
        'Petal_Width': petal_width
    }

    # Convert dictionary to NumPy array for prediction
    features = np.array(list(data.values())).reshape(1, -1)
    return features


# Get the user input
input_data = user_input_features()

st.sidebar.markdown("---")
st.sidebar.info("Model used: KNN (K=5)")

# --- 4. Prediction Logic ---

# Make prediction
prediction_index = knn_model.predict(input_data)[0]
predicted_species = target_names[prediction_index].title()

# --- 5. Output Display ---
st.write("### 📏 Current Input Features:")
# Display the input data in a nice format
input_df = np.round(input_data, 2)
st.table(input_df)

st.write("### 🔮 Predicted Iris Species:")

# Display the prediction with styling
if prediction_index == 0:
    st.success(f"The predicted species is: **{predicted_species}** (Setosa)")
elif prediction_index == 1:
    st.warning(f"The predicted species is: **{predicted_species}** (Versicolor)")
else:
    st.error(f"The predicted species is: **{predicted_species}** (Virginica)")

st.balloons()