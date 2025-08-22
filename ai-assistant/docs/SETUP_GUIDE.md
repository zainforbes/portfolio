# AI Assistant Setup Guide

This guide will help you set up the AI Assistant with all required dependencies and configurations.

## 🔧 Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required: Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Brave Search API Key (for web search functionality)
BRAVE_API_KEY=your_brave_search_api_key_here

# Optional: Google API Credentials (for Gmail and Calendar)
# Place credentials.json in the config/ folder
```

### 3. Google API Setup (Optional)

For Gmail and Calendar functionality:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API and Calendar API
4. Create credentials (OAuth 2.0 Client ID)
5. Download the credentials file as `config/credentials.json`

### 4. Brave Search API Setup (Optional)

For web search functionality:

1. Sign up at [Brave Search API](https://brave.com/search/api/)
2. Get your API key
3. Add it to your `.env` file as `BRAVE_API_KEY`

## 🧪 Testing the Setup

Run the test script to verify everything works:

```bash
python test_fixes.py
```

This will test all agents and show you what's working and what needs attention.

## 🚀 Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

## 🔍 Troubleshooting

### Search Agent Issues
- **Problem**: Web search returns fallback responses
- **Solution**: Check that `BRAVE_API_KEY` is set and `httpx` is installed
- **Fallback**: Even without API key, the agent will provide helpful responses using Gemini

### Email Agent Issues
- **Problem**: "Gmail service not initialized" errors
- **Solution**: Ensure `credentials.json` is in the `config/` folder and run the app to complete OAuth flow

### Calendar Agent Issues
- **Problem**: Calendar events not loading
- **Solution**: Same as email - ensure Google API credentials are set up correctly

### General Issues
- **Problem**: Agents returning error responses
- **Solution**: Check that `GEMINI_API_KEY` is valid and the model `gemini-2.5-flash-lite` is accessible

## 📋 Features by Agent

### 🔍 Search Agent
- Web search with Brave API
- Fallback responses using Gemini knowledge
- Query optimization and result analysis

### 📧 Email Agent  
- List and classify emails
- Email summarization
- Priority detection
- Inbox management

### 📅 Calendar Agent
- View upcoming events
- Create new events
- Schedule analysis
- Conflict detection
- Availability checking

### 🤖 Orchestrator
- Intelligent request routing
- Multi-agent coordination
- Enhanced fallback responses
- Error recovery

## 💡 Usage Examples

Try these commands once the app is running:

```
# Search functionality
"what is the capital of prague"
"search for latest AI developments"

# Email functionality  
"show my unread emails"
"classify my emails by priority"

# Calendar functionality
"show my upcoming events"
"when am I free tomorrow?"
"create a meeting for next week"
```

## 🐛 Known Limitations

1. **Google API Authentication**: Requires manual OAuth flow on first run
2. **Search API**: Rate limits apply to Brave Search API
3. **Model Availability**: Relies on Gemini 2.5 Flash Lite availability

## 📞 Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Run `python test_fixes.py` to identify specific problems
3. Verify all environment variables are set correctly
4. Check API key validity and quotas