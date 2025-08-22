# Main App

import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the AI Assistant workflow
from src.core.langgraph_workflow import AIAssistantWorkflow

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖")

st.title("🤖 Personal AI Assistant")
st.caption("✨ Powered by Email & Calendar Agents with LangGraph")

# Initialize session state
if "workflow" not in st.session_state:
    st.session_state.workflow = None
    st.session_state.messages = []

# Initialize workflow (cached)
@st.cache_resource
def get_workflow():
    try:
        workflow = AIAssistantWorkflow()
        return workflow
    except Exception as e:
        st.error(f"Error initializing AI Assistant: {str(e)}")
        return None

# Get workflow instance
workflow = get_workflow()

if workflow is None:
    st.error("⚠️ Unable to initialize AI Assistant. Please check your credentials and MCP servers.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show routing info for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("🔍 Routing Details"):
                metadata = message["metadata"]
                st.write(f"**Route:** {metadata.get('route', 'unknown')}")
                st.write(f"**Agent:** {metadata.get('current_agent', 'unknown')}")
                st.write(f"**Confidence:** {metadata.get('route_confidence', 0):.2f}")

# Chat input
if prompt := st.chat_input("Ask me about your calendar or emails..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with AI Assistant
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            try:
                # Run the workflow
                result_state = asyncio.run(workflow.process_request(prompt))
                
                # Extract response
                response = result_state.get('final_response', 'No response generated')
                
                # Display response
                st.markdown(response)
                
                # Add to chat history with metadata
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "metadata": {
                        "route": result_state.get('route', 'unknown'),
                        "current_agent": result_state.get('current_agent', 'unknown'),
                        "route_confidence": result_state.get('route_confidence', 0),
                        "route_reason": result_state.get('route_reason', 'unknown'),
                        "verification_scores": result_state.get('verification_scores', {})
                    }
                })
                
                # Show debug info
                with st.expander("🔧 Debug Info"):
                    st.json({
                        "route": result_state.get('route', 'unknown'),
                        "route_confidence": result_state.get('route_confidence', 0),
                        "route_reason": result_state.get('route_reason', 'unknown'),
                        "current_agent": result_state.get('current_agent', 'unknown'),
                        "fallback_triggered": result_state.get('fallback_triggered', False),
                        "task_type": result_state.get('task_type', 'unknown')
                    })
                
                # Show any errors
                if result_state.get('error_log'):
                    with st.expander("⚠️ Errors Encountered"):
                        for error in result_state['error_log']:
                            st.error(f"**{error.get('agent', 'System')}:** {error.get('error', 'Unknown error')}")
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with system info and examples
with st.sidebar:
    st.header("🛠️ System Status")
    
    if workflow:
        with st.spinner("Checking system health..."):
            try:
                health = asyncio.run(workflow.health_check())
                
                st.subheader("Components")
                for component, status in health.items():
                    if isinstance(status, dict):
                        st.write(f"**{component.title()}:**")
                        for sub_component, sub_status in status.items():
                            icon = "✅" if sub_status == "healthy" else "❌"
                            st.write(f"  {icon} {sub_component}")
                    else:
                        icon = "✅" if status == "healthy" else "❌"
                        st.write(f"{icon} **{component.title()}:** {status}")
                        
            except Exception as e:
                st.error(f"Health check failed: {str(e)}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
