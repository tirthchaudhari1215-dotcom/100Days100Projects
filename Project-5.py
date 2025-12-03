import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Best for text classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


# --- 1. Data Loading and Model Training Function ---

@st.cache_resource
def train_spam_model(csv_path='spam.csv'):
    """Loads data, preprocesses, trains the Multinomial Naive Bayes Pipeline."""
    try:
        # Load the dataset (it often has odd encoding/columns)
        df = pd.read_csv(csv_path, encoding='latin-1')
    except FileNotFoundError:
        st.error(f"Error: '{csv_path}' not found. Please download the dataset.")
        return None, None

    # Clean and simplify columns: The first two columns are needed
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']

    # Map target variable 'label' to binary (0=ham, 1=spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['message']
    y = df['label']

    # Split data (used for training the pipeline)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline: Vectorize text first, then apply Naive Bayes
    pipeline = Pipeline([
        # Step 1: Convert text messages into numerical features (Bag-of-Words)
        ('vectorizer', CountVectorizer(stop_words='english', lowercase=True)),
        # Step 2: Apply the classification algorithm
        ('nb', MultinomialNB())
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Save components
    joblib.dump(pipeline, 'spam_nb_pipeline.pkl')

    # Optional: Get a quick accuracy check
    # y_pred = pipeline.predict(X_train)
    # accuracy = accuracy_score(y_train, y_pred)

    return pipeline


# Load trained components
pipeline = train_spam_model()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 5: Spam SMS Classifier")

st.title("📧 Day 5: Spam SMS Classifier")
st.subheader("Multinomial Naive Bayes with Text Vectorization")
st.markdown("Enter a short message to check if the model classifies it as Spam or Ham (legitimate).")

if pipeline is not None:

    st.sidebar.header("Test the Classifier")

    # --- 3. User Input ---
    user_input = st.text_area(
        "Enter a message here:",
        "Congratulations! You won a FREE $1,000 gift card! Text BACK to claim."
    )

    # --- 4. Prediction Logic ---
    if st.button('Classify Message'):
        # The pipeline handles both vectorization and prediction in one go
        prediction = pipeline.predict([user_input])[0]

        # Predict probability for confidence
        prediction_proba = pipeline.predict_proba([user_input])[0]

        # --- 5. Output Display ---
        st.write("### 📈 Classification Result:")

        if prediction == 1:
            st.error(f"**SPAM!** 🚨 (Confidence: {prediction_proba[1] * 100:.2f}%)")
        else:
            st.success(f"**HAM (Legitimate)** ✅ (Confidence: {prediction_proba[0] * 100:.2f}%)")

        st.markdown("---")
        st.write("### 🔍 Model Insights")
        st.info("Algorithm: Multinomial Naive Bayes. The text was processed using **CountVectorizer**.")

