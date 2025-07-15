import streamlit as st
from chatbot import CVChatbot
from utils import extract_pdf_text

# Configure the page
st.set_page_config(page_title="CV Interview Chatbot", layout="wide")

# Callback to process user input immediately
def handle_user_input():
    query = st.session_state.user_query.strip()
    if not query:
        return
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    # Generate AI response
    with st.spinner("AI is typing..."):
        response = st.session_state.bot.generate_response(query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Clear the input
    st.session_state.user_query = ""

# Initialize session state
if "bot" not in st.session_state:
    st.session_state.bot = CVChatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# Sidebar: setup & sample questions
with st.sidebar:
    st.header("⚙️ Setup & Sample Questions")

    # Test API connection
    if st.button("Test API Connection"):
        if st.session_state.bot.test_connection():
            st.success("Connected to Groq API!")
        else:
            st.error("Failed to connect. Please check your API key.")

    # CV upload
    uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")
    if uploaded_file:
        cv_text = extract_pdf_text(uploaded_file)
        st.session_state.bot.set_cv(cv_text)
        st.success(f"CV loaded ({len(cv_text)} chars)")

    st.markdown("---")
    st.subheader("💡 Quick Prompts")
    sample_questions = [
        "Tell me about yourself",
        "What are your key strengths?",
        "Describe your work experience",
        "What skills do I have?",
        "Why are you interested in this role?"
    ]
    for idx, prompt in enumerate(sample_questions):
        if st.button(prompt, key=f"sample_{idx}"):
            # Directly handle sample question
            st.session_state.user_query = prompt
            handle_user_input()

# Main chat interface
st.title("🤖 Chat with Your CV")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input area (at bottom) using text_input with on_change
if st.session_state.bot.cv_text:
    st.text_input(
        "Ask a question about your CV...",
        key="user_query",
        placeholder="Type your question and press Enter",
        on_change=handle_user_input
    )
else:
    st.info("📄 Please upload your CV in the sidebar to start chatting.")