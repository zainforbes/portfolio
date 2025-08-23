import streamlit as st
from src.utils.gemini_client import GeminiClient

st.title("🤖 Personal AI Assistant")

gemini = GeminiClient()
user_input = st.text_input("Ask me anything:")

if user_input:
    st.write("Assistant:", gemini.chat(user_input))
