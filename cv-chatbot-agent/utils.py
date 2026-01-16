import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env in local dev
load_dotenv()

# Load .env from the same directory as this file (utils.py) or your project root
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def get_api_key() -> str:
    """
    Retrieve GROQ API key from Streamlit secrets (when deployed) or environment (.env/local).
    Raises an error if not found.
    """
    # First try Streamlit secrets (for Cloud deployments)
    key = None
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # Fallback to environment variable
    if not key:
        key = os.getenv("GROQ_API_KEY")

    if not key:
        raise ValueError(
            "Hello! I couldn’t find your GROQ_API_KEY. "
            "Please add it to a local .env or to your Streamlit Cloud secrets."
        )
    return key


def extract_pdf_text(file) -> str:
    """
    Read uploaded PDF file-like object and return its full text.
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required to extract PDF text. Please install it.")

    reader = PyPDF2.PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)
