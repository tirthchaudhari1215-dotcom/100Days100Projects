import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --- 1. LangChain + Groq Setup ---

def analyze_code(code_snippet, groq_api_key):
    # Initialize Groq LLM (High-speed Llama 3)
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"  # One of the best open-source models
    )

    # Define the Logic (Prompt Template)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert Senior Software Engineer. Review the following code for bugs, efficiency, and security."),
        ("human", "{code}")
    ])

    # Create the Modern Chain (LCEL)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"code": code_snippet})


# --- 2. Streamlit UI ---

st.set_page_config(page_title="Day 12: Groq + LangChain", page_icon="⚡")

st.title("⚡ Day 12: Groq Cloud")
st.subheader("The World's Fastest AI Inference Engine")

with st.sidebar:
    api_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("[Get a free Groq Key here](https://console.groq.com/keys)")

code_input = st.text_area("Paste your code here (Python, JS, etc.):", height=200)

if st.button("Review Code"):
    if not api_key:
        st.warning("Please enter your Groq API Key!")
    elif code_input:
        with st.spinner("Groq is analyzing at light speed..."):
            review = analyze_code(code_input, api_key)
            st.markdown("### 🔍 Code Review Results")
            st.markdown(review)
    else:
        st.info("Please paste some code to review.")