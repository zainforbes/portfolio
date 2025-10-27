# 🧠 Wikipedia Research Agent (Gemini + Agno)

An interactive command-line research assistant powered by **Google Gemini** and **Agno**.  
It searches Wikipedia for any topic you type and summarizes the main insights clearly and concisely.

---

## 🚀 Features

- 🔍 Search *any* topic on Wikipedia using Gemini 2.0 Flash  
- 🧩 Clean Markdown summaries of key points  
- 💬 Interactive loop — ask as many questions as you like  
- 🔒 Secure API key handling via `.env`  
- ⚙️ Compatible across Agno versions

---

## 🧩 Requirements

- Python 3.9
- A valid **Google Gemini API key**
- Internet connection (Wikipedia queries are live)

---

## ⚙️ Setup

### 1️⃣ Clone or copy this project

```bash
git clone https://github.com/yourusername/wiki_agent.git
cd wiki_agent
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate         # Windows PowerShell
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add your Gemini API key

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Run the Agent

```bash
python wiki_agent.py
```

Example session:

```
🔍 Wikipedia Research Agent
Type 'exit' to quit.

Enter a topic to search: Machine learning

📚 Searching Wikipedia for 'Machine learning'...

1. **Definition:** Machine learning is a subset of AI focused on pattern recognition and prediction.  
2. **Applications:** Used in recommendation systems, speech recognition, and computer vision.  
3. **Approaches:** Includes supervised, unsupervised, and reinforcement learning.

================================================================================
Enter a topic to search: exit
👋 Goodbye!
```

---

## 🧪 Run Unit Tests

```bash
pytest -v
```

Tests include:
- Environment variable loading check  
- Agent initialization verification  

---

## 📁 Project Structure

```
wiki_agent/
├── wiki_agent.py
├── tests/
        test_wikipedia_agent.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Future Enhancements
 
- Add voice or chat UI using Gradio  
- Log query history for later review  

---

© 2025 – Built by Zain Forbes
