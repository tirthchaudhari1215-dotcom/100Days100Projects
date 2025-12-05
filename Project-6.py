import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb  # NEW LIBRARY IMPORT!
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

# --- Constants for IMDB data ---
MAX_WORDS = 10000  # Only consider the top 10,000 most frequent words
INDEX_FROM = 3  # Index shift needed for IMDB data


# --- 1. Data Loading and Model Training Function ---

@st.cache_resource
def train_xgb_model():
    """Loads IMDB data (no CSV needed!), preprocesses, trains XGBoost Pipeline."""

    # Load the data: x_train and x_test are lists of word indices (sequences)
    (x_train, y_train), (_, _) = imdb.load_data(num_words=MAX_WORDS, index_from=INDEX_FROM)

    # 1. Get the word index mapping
    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_to_word = {v: k for k, v in word_to_id.items()}

    # 2. Convert numerical sequences back to text strings (CRITICAL STEP for Vectorizer)
    def sequence_to_text(sequence):
        # Convert the list of indices back to a single string sentence
        return ' '.join(word_to_to_word.get(i, '<UNK>') for i in sequence)

    # Convert training data to text
    X_text = pd.Series([sequence_to_text(s) for s in x_train])
    y = y_train

    # Split data (used for training the pipeline)
    X_train_text, _, y_train, _ = train_test_split(X_text, y, test_size=0.2, random_state=42)

    # 3. Create a pipeline: TF-IDF Vectorizer first, then XGBoost Classifier
    pipeline = Pipeline([
        # Step 1: Convert text reviews into numerical features (TF-IDF)
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, max_features=5000)),
        # Step 2: Apply the classification algorithm
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=100))
    ])

    # Train the pipeline
    pipeline.fit(X_train_text, y_train)

    # Save components
    joblib.dump(pipeline, 'sentiment_imdb_xgb_pipeline.pkl')

    return pipeline


# Load trained components
pipeline = train_xgb_model()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 7: IMDB Sentiment Analyzer (XGBoost)")

st.title("🎬 Day 7: IMDB Review Sentiment Analyzer")
st.subheader("XGBoost with TF-IDF Vectorization (No CSV File Needed!)")
st.markdown("Enter a movie review to see if the model predicts positive or negative sentiment.")

if pipeline is not None:

    st.sidebar.header("Test the Review")

    # --- 3. User Input ---
    user_input = st.text_area(
        "Enter a movie review here:",
        "This movie was an absolute disaster. The acting was terrible and the plot made no sense."
    )

    # --- 4. Prediction Logic ---
    if st.button('Analyze Sentiment'):
        # The pipeline handles both vectorization and prediction
        prediction = pipeline.predict([user_input])[0]

        # Predict probability for confidence
        prediction_proba = pipeline.predict_proba([user_input])[0]

        # --- 5. Output Display ---
        st.write("### 📈 Sentiment Analysis Result:")

        if prediction == 1:
            st.success(f"**POSITIVE (1)** 👍 (Confidence: {prediction_proba[1] * 100:.2f}%)")
        else:
            st.error(f"**NEGATIVE (0)** 👎 (Confidence: {prediction_proba[0] * 100:.2f}%)")

        st.markdown("---")
        st.write("### 🔍 Model Insights")
        st.info("Algorithm: XGBoost. Text features were extracted using **TF-IDF Vectorizer**.")

