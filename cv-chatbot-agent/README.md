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
```

Activate the virtual environment and install dependencies:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Configuration

Create a file named `.env` in `cv-chatbot/` with:

```ini
GROQ_API_KEY=sk-<your-groq-key-here>
```

> **Note**: `.env` is listed in `.gitignore`, so it won’t be committed.

### Run Locally

```bash
streamlit run main.py
```

Open your browser to [http://localhost:8501](http://localhost:8501).

---

## 📦 Deployment

Use **Streamlit Community Cloud** to host for free:

1. Push the `cv-chatbot/` folder to GitHub.  
2. Visit https://share.streamlit.io → **New app** → connect your repo.  
3. Set **Repository** to `your-username/portfolio`, **Branch** to `main`, **File path** to `cv-chatbot/main.py`.  
4. In **Advanced settings**, add an environment variable:  
   - **Key:** `GROQ_API_KEY`  
   - **Value:** your Groq API key  
5. Click **Deploy**.

You’ll get a live URL (e.g. `https://your-username-portfolio-cv-chatbot.streamlit.app`) to share—no local startup needed.

---

## 🎯 Usage

- **Test API**: Click “Test API Connection” in the sidebar to verify your key.  
- **Upload CV**: Drag your résumé PDF into the sidebar.  
- **Ask Questions**:  
  - **Quick-prompt**: click any sample question.  
  - **Free-form**: type in the “Ask a question…” box and press Enter.  
- **View responses** in the main chat window, which scrolls independently of the input.

---

## 🤝 Contributing & Support

This repo is intended as a demo scaffold—you’re welcome to:

- Swap in other LLMs or RAG backends  
- Extend with agent/tool-calling logic  
- Polish UI styling or deploy via other platforms (Heroku, Vercel)

If you run into issues or have feedback, open an issue or PR on GitHub.

---

## 📄 License

MIT License © 2025 Your Name
