# Main App

import streamlit as st
from src.core.llm_client import GeminiClient

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖")

st.title("🤖 Personal AI Assistant")

gemini = GeminiClient()

user_input = st.text_input("Ask me something:")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        response = gemini.chat(user_input)
    st.write(response)
