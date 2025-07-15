import os
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

def get_api_key() -> str:
    """
    Reads GROQ_API_KEY from .env and errors if missing.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Please set GROQ_API_KEY in your .env")
    return key

def extract_pdf_text(file) -> str:
    """
    Take an uploaded PDF file-like and return its full text.
    """
    reader = PyPDF2.PdfReader(file)
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
    return "\n".join(text_pages)
