# 🤖 CV Interview Chatbot

A Streamlit app that lets you engage in a “mock interview” with your own CV. Powered by Groq’s `llama3-8b-8192` API, it combines PDF‐text extraction, prompt engineering, and real-time chat for fast, professional Q&A based on your résumé content.

---

## 🔍 Features

- **PDF CV ingestion**: Drag & drop your résumé (PDF) and extract its full text.  
- **Quick-prompt buttons**: One-click interview questions, e.g. “Tell me about yourself.”  
- **Free-form chat**: Type any question (“What skills do I have?”), press Enter, get an instant answer.  
- **Streamlit chat UI**: Modern message bubbles (`st.chat_message`) and sidebar input—no scrolling fatigue.  
- **No model training**: All LLM work happens via the Groq API—zero infra to manage.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- A free Groq API key (sign up at https://console.groq.com — 14 400 free calls/day)  
- Git & GitHub account (for optional deployment)

### Installation

```bash
git clone https://github.com/your-username/portfolio.git
cd portfolio/cv-chatbot
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

