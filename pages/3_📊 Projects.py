import streamlit as st
import pandas as pd

st.title("📊 Student Projects")

projects = pd.DataFrame([
    {"Name": "Chatbot for Support", "Tech": "LangChain + RAG", "Student": "Alice"},
    {"Name": "Stock Predictor", "Tech": "LSTM", "Student": "Bob"},
    {"Name": "Thai VoiceBot", "Tech": "Rasa + Google TTS", "Student": "Charlie"},
])

st.dataframe(projects)
