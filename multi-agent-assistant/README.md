# Multi-Agent Personal AI Assistant

A sophisticated personal AI assistant built with **LangChain** that integrates **email, calendar, and web search** through Google APIs and Brave Search. The system uses LangChain's tool-calling capabilities to handle complex tasks with intelligent reasoning.

---

## ✨ Features

- **Gmail Integration** – Read, list, and send emails directly from the chat
- **Calendar Management** – View, create, update, and delete Google Calendar events
- **Web Search** – Brave Search integration for real-time information gathering
- **Intelligent Reasoning** – Powered by Google Gemini models for task planning and tool selection
- **Interactive UI** – Streamlit-based chat interface with real-time feedback
- **Secure Authentication** – OAuth2 flow for Google services and environment-based API keys

---

## 🧠 System Architecture

### Agent Logic

The system uses a unified **LangChain Agent** that leverages the `gemini-1.5-flash` model for reasoning and execution.

#### 1. **Core Agent (`LangChainAgent`)**
- **Orchestration**: Built using LangChain's `create_tool_calling_agent`.
- **Reasoning**: Analyzes user input, maintains conversation history, and decides which tools to call.
- **Tools**: Accesses specialized modules for Gmail, Google Calendar, and Brave Search.
- **Callbacks**: Uses custom Streamlit callbacks to provide real-time updates of tool execution in the UI.

#### 2. **Tool Ecosystem**
- **Email Tool**: Interfaces with Gmail API for listing, reading, and sending messages.
- **Calendar Tool**: Manages Google Calendar events with natural language support for dates and times.
- **Search Tool**: Utilizes Brave Search API for up-to-date web information.

---

### Agent Responsibility Matrix

| Agent Component | Primary Responsibility                       | Tools Used                                            | Confirmation |
|-----------------|-----------------------------------------------|-------------------------------------------------------|--------------|
| **Gemini LLM**  | Reasoning, tool selection, response synthesis | `google-generativeai`                                 | No           |
| **Email**       | Gmail operations (list, read, send)          | `gmail_list`, `gmail_read`, `gmail_send`              | Yes (Manual) |
| **Calendar**    | Scheduling, queries, event management        | `gcal_list`, `gcal_create`, `gcal_update`, `gcal_delete` | Yes (Manual) |
| **Search**      | Web research and information gathering       | `web_search` (Brave API)                              | No           |

---

## 🔌 Communication & Workflows

### State Management
- **AssistantState**: A centralized schema for maintaining conversation history, agent messages, and session memory.
- **Real-time Feedback**: Tool execution steps are pushed to the UI as they happen, giving visibility into the assistant's "thinking" process.

### Confirmation Workflow
1. Certain actions (like sending an email or creating a calendar event) can be configured to require explicit user confirmation.
2. The agent prepares the action and waits for a "send" or "confirm" command from the user.
3. Once confirmed, the action is executed via the corresponding tool.

---

## 🗂️ Tool Integration

The system integrates with external services using specialized server implementations:

- **Search**: `search_server.py` handles Brave Search API calls with TTL caching.
- **Gmail**: `gmail_server.py` manages Google OAuth2 and Gmail API interactions.
- **Calendar**: `calendar_server.py` handles Google Calendar with robust time normalization for natural language queries.

---

## 🔧 Prerequisites

- Python 3.8+
- Google Cloud project with Gmail and Calendar APIs enabled
- Brave Search API key
- Google Gemini API key (from AI Studio)

---

## 🚀 Setup Instructions

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd multi-agent-assistant
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**  
   Create `.env` in the root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   BRAVE_API_KEY=your_brave_api_key
   ```

5. **Google OAuth Setup**
   - Create project in [Google Cloud Console](https://console.cloud.google.com/)  
   - Enable Gmail + Calendar APIs  
   - Create OAuth credentials (Desktop App)  
   - Save as `config/credentials.json`  
   - Run bootstrap to generate `token.json`:
     ```bash
     python scripts/google_oauth_bootstrap.py
     ```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🧪 Testing

The project includes a comprehensive test suite using `pytest`.

```bash
pytest tests/
```

Individual smoke scripts are also available in `scripts/`.

---

## 🏗️ Technical Stack

- **Framework**: LangChain
- **LLM**: Google Gemini 1.5 Flash
- **UI**: Streamlit
- **APIs**: Google Gmail, Google Calendar, Brave Search
- **Asynchronous**: Built with `asyncio` and `httpx` for high performance
