# CV-Chatbot (Mistral-7B-Instruct)

A Retrieval‑Augmented Generation chatbot around my CV, powered by Mistral-7B-Instruct and LangChain agent orchestration.

## Features
- **RAG retrieval** via FAISS + sentence-transformers
- **Agent orchestration** (zero-shot React) with LangChain
- **Mistral-7B-Instruct** inference using `mistral-inference`
- **Gradio** chat UI, local or hosted on Hugging Face Spaces

## Requirements
- Python 3.8+
- Git

## Installation & Local Run
```bash
git clone https://github.com/your-org/cv-chatbot.git
cd cv-chatbot

# 1. Create env
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Download model
python model_download.py

# 4. Index your CV
#    ensure data/my_cv.pdf is present
python index_cv.py

# 5. Launch the chatbot
python app.py
