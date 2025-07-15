import os
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

def get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Please set GROQ_API_KEY in your .env")
    return key

def extract_pdf_text(file) -> str:
    """
    Read a PDF file-like and return the concatenated text.
    """
    reader = PyPDF2.PdfReader(file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)
