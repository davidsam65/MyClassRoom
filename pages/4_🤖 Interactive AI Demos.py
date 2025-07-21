import streamlit as st
from transformers import pipeline

st.title("🤖 Try an AI Model Live")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

user_input = st.text_input("Enter a sentence:")
if user_input:
    result = model(user_input)
    st.success(f"Prediction: {result[0]['label']} (Confidence: {result[0]['score']:.2f})")
