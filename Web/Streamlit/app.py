import streamlit as st
import joblib
import os

# Set page config for a beautiful layout
st.set_page_config(
    page_title="Hateful Speech Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for elegance
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        background-color: #4f8cff;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        border-radius: 8px;
        padding: 0.5em;
    }
    .result-box {
        background-color: #e0f7fa;
        border-radius: 10px;
        padding: 1.5em;
        margin-top: 1em;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model

pipeline = joblib.load('././models/hate_speech_pipeline.joblib')

# App title and description
st.title("🛡️ Hateful Speech Detection")
st.write(
    "Enter a sentence below to check if it contains hate speech. "
    "This tool uses a machine learning model to classify your input."
)

# Input box
user_text = st.text_area(
    "Type your sentence here:",
    height=100,
    max_chars=500,
    placeholder="Enter text to analyze..."
)

# Predict button
if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        prediction = pipeline.predict([user_text])[0]
        if prediction == 0:
            result = "✅ Not Hate Speech"
            color = "#43a047"
        else:
            result = "🚫 Hate Speech"
            color = "#e53935"
        st.markdown(
            f'<div class="result-box" style="background-color:{color}20;">'
            f'<span style="color:{color};">{result}</span>'
            '</div>',
            unsafe_allow_html=True
        )
