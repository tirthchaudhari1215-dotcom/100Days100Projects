import streamlit as st
from transformers import pipeline


# --- 1. Load Tools (Pre-trained Models) ---

@st.cache_resource
def load_models():
    # Model 1: DistilBERT (Fast and efficient)
    model_fast = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Model 2: RoBERTa (Optimized and robust)
    model_robust = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    return model_fast, model_robust


st.set_page_config(page_title="Day 11: Hugging Face Hub")

st.title("🤗 Day 11: Hugging Face Model Hub")
st.subheader("Multi-Model Sentiment Comparison")
st.markdown("Leveraging pre-trained LLMs directly from the Hugging Face ecosystem.")

# Load models
with st.spinner("Downloading models from Hugging Face... (This may take a minute)"):
    fast_pipe, robust_pipe = load_models()

# --- 2. User Interface ---

user_input = st.text_area("Enter text to analyze:",
                          "The performance of this new AI tool is absolutely mind-blowing, though the documentation is a bit sparse.")

if st.button("Analyze Sentiment"):
    if user_input:
        col1, col2 = st.columns(2)

        # --- 3. Inference with Tool 1 ---
        with col1:
            st.write("### ⚡ DistilBERT (Fast)")
            result = fast_pipe(user_input)[0]
            label = result['label']
            score = result['score']

            if label == 'POSITIVE':
                st.success(f"{label} ({score:.2%})")
            else:
                st.error(f"{label} ({score:.2%})")

        # --- 4. Inference with Tool 2 ---
        with col2:
            st.write("### 💪 RoBERTa (Robust)")
            # RoBERTa output labels: Label_0 (Neg), Label_1 (Neu), Label_2 (Pos)
            result = robust_pipe(user_input)[0]
            mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
            label = mapping.get(result['label'], result['label'])
            score = result['score']

            if label == 'POSITIVE':
                st.success(f"{label} ({score:.2%})")
            elif label == 'NEUTRAL':
                st.warning(f"{label} ({score:.2%})")
            else:
                st.error(f"{label} ({score:.2%})")

    else:
        st.warning("Please enter some text first!")

st.markdown("---")
st.write("### 💡 Why this tool matters?")
st.info(
    "Hugging Face allows you to pull 'State-of-the-Art' (SOTA) models into your app with zero training. You just switched between two distinct architectures (BERT vs RoBERTa) using the exact same code structure.")