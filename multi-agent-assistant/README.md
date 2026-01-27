# Multi-Agent Personal AI Assistant

A sophisticated personal AI assistant built with **LangChain** that integrates **email, calendar, and web search** through Google APIs and MCP (Model Context Protocol) servers. The system uses a tool-calling agent to handle complex tasks through intelligent planning and execution.

---

## ✨ Features

- **Gmail Integration** – Read, compose, and send emails with intelligent drafting  
- **Calendar Management** – View, create, and manage Google Calendar events  
- **Web Search** – Brave Search integration for real-time information  
- **LangChain-based Agent** – Unified agent logic with robust tool calling
- **Smart Planning** – LLM-driven task decomposition and execution  
- **Interactive UI** – Streamlit-based chat interface with confirmation workflows  
- **Secure Authentication** – OAuth2 flow for Google services  

---

## 🧠 System Architecture

### Agent Interactions & Decision-Making

The system employs a LangChain `ToolCallingAgent` that orchestrates various capabilities:

#### 1. **LangChain Agent (Central Orchestrator)**
- **Role**: Primary decision maker and task executor
- **Intelligence**: Powered by Google Gemini LLM
- **Process**:
  1. Analyze user input and conversation history  
  2. Plan necessary steps and select appropriate tools
  3. Execute tools sequentially, maintaining context between steps
  4. Handle confirmation-gated actions for security

#### 2. **Specialized Tools**
- **Email Tools** – Gmail operations (list, read, send)
- **Calendar Tools** – Google Calendar (list, create, update, delete)
- **Search Tool** – Web search via Brave API

---

### Tool Responsibility Matrix

| Tool Category | Responsibility                           | Underlying Implementation                              | Confirmation |
|---------------|------------------------------------------|--------------------------------------------------------|--------------|
| **Email**     | Gmail ops, drafting, sending             | `gmail_list_recent`, `gmail_read`, `gmail_send`        | Mutating ops |
| **Calendar**  | Scheduling, queries, conflict detection  | `gcal_list_events`, `gcal_create`, `gcal_update`, ...  | Mutating ops |
| **Search**    | Web research, synthesis, info gathering  | `web_search` (Brave API)                               | No           |

---

## 🔌 Communication & Workflows

### State Management
- Centralized **AssistantState** schema with memory/history  
- Shared context handled by LangChain's internal scratchpad
- Structured `agent_messages` for real-time UI updates via `StreamlitCallbackHandler`

### Confirmation Workflow
1. Agent identifies a mutating action (e.g., sending an email)
2. System pauses and requests user confirmation in the UI
3. User approves (e.g., “send”) or provides edits
4. Action is executed only after explicit confirmation

### Error Handling
- Graceful recovery from tool errors
- Clear error messages displayed in the UI
- State preservation for seamless continuation

---

## 🗂️ Tool Integration

The agent uses a suite of tools that connect to external services via the Model Context Protocol (MCP) server patterns.

### Data Flow
- User Input → LangChain Agent → Tool Selection → Tool Execution → UI Update → Final Response
- Secure OAuth2 token management for Google services

---

## ⚡ Performance

- Fully **async/await** operations  
- Respects API quotas (Google + Brave)  
- Intelligent caching with `cache_manager.py`  
- Rolling conversation history management

---

## 🔧 Prerequisites

- Python 3.8+  
- Google Cloud account with Gmail + Calendar APIs enabled  
- Brave Search API key  
- Google Gemini API key  

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
   - Run bootstrap:  
     ```bash
     python scripts/google_oauth_bootstrap.py
     ```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open in browser: [http://localhost:8501](http://localhost:8501)

---

## 🔑 API Keys Setup

- **Gemini API Key**: Get from [Google AI Studio](https://aistudio.google.com/)  
- **Brave Search Key**: Get from [Brave Search API](https://brave.com/search/api/)  

Add both to `.env`.  

---

## 💡 Usage Examples

- **Email**: “Check my recent emails” / “Send an email to john@example.com”  
- **Calendar**: “What’s on my calendar today?” / “Schedule a meeting tomorrow at 3pm”  
- **Search**: “Search the latest AI news” / “Find info about Python asyncio”  
- **Multi-step**: “Search for X, summarize, and email Y”  

---

## 🧪 Testing

```bash
python -m pytest tests/test_langchain_agent.py
```

---

## 🏗️ Architecture

- **Core**: LangChain agent orchestration
- **Tools**: Specialized wrappers for email, calendar, and search
- **UI**: Streamlit-based chat with custom callback handlers for real-time trace

---

## 📦 Dependencies

- `streamlit` – Web UI  
- `langchain` – Agent framework and tool orchestration
- `langchain-google-genai` – Google Gemini integration for LangChain
- `google-api-python-client` – Google APIs  
- `mcp` – Model Context Protocol concepts
- `aiohttp` – Async HTTP client  

---

## 🛡️ Security Notes

- Do **not** commit `credentials.json` or `token.json`  
- Store API keys in environment variables  
- Regularly review OAuth scopes and permissions  
