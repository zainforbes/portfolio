# Enhanced Personal AI Assistant App
import streamlit as st
import asyncio
import os
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Import the Enhanced AI Assistant workflow
from src.core.enhanced_langgraph_workflow import EnhancedAIAssistantWorkflow
from src.core.enhanced_state_schema import safe_get_resource_metric

# Page configuration
st.set_page_config(
    page_title="Enhanced AI Assistant", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .agent-card {
        background: linear-gradient(145deg, #f0f4f8 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-healthy { background-color: #48bb78; }
    .status-warning { background-color: #ed8936; }
    .status-error { background-color: #f56565; }
    
    .conversation-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "enhanced_workflow" not in st.session_state:
    st.session_state.enhanced_workflow = None
    st.session_state.messages = []
    st.session_state.conversation_id = f"conv_{int(time.time())}"
    st.session_state.session_stats = {
        'messages_sent': 0,
        'agents_used': set(),
        'total_confidence': 0.0,
        'session_start': datetime.now(),
        'performance_history': []
    }

# Initialize Enhanced workflow (cached)
@st.cache_resource
def get_enhanced_workflow():
    try:
        workflow = EnhancedAIAssistantWorkflow()
        return workflow
    except Exception as e:
        st.error(f"Error initializing Enhanced AI Assistant: {str(e)}")
        return None

# Header
st.markdown('<h1 class="main-header">🚀 Enhanced AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #718096; margin-bottom: 2rem;">
    ✨ <strong>Next-Generation AI with Advanced Multi-Agent Orchestration</strong> ✨<br>
    🧠 Intelligent Routing • 🤝 Dynamic Collaboration • 🎯 Autonomous Decisions • 🔄 Error Recovery
</div>
""", unsafe_allow_html=True)

# Get workflow instance
workflow = get_enhanced_workflow()

if workflow is None:
    st.error("⚠️ Unable to initialize Enhanced AI Assistant. Please check your credentials and MCP servers.")
    st.stop()

# Sidebar with enhanced system monitoring
with st.sidebar:
    st.header("🏗️ System Architecture")
    
    # Agent Status Cards
    st.subheader("🤖 Enhanced Agents")
    
    agents_info = {
        "📧 Enhanced Email Agent": {
            "capabilities": ["Smart Composition", "Context Management", "Send Confirmation"],
            "status": "healthy",
            "description": "Advanced email handling with conversation state management"
        },
        "📅 Enhanced Calendar Agent": {
            "capabilities": ["Intelligent Scheduling", "Conflict Resolution", "Optimization"],
            "status": "healthy", 
            "description": "AI-powered calendar management with smart scheduling"
        },
        "🔍 Enhanced Search Agent": {
            "capabilities": ["Query Optimization", "Source Validation", "Fact Verification"],
            "status": "healthy",
            "description": "Intelligent web search with comprehensive analysis"
        }
    }
    
    for agent_name, info in agents_info.items():
        with st.container():
            st.markdown(f"""
            <div class="agent-card">
                <h4>{agent_name}</h4>
                <p style="font-size: 0.9em; color: #4a5568;">{info['description']}</p>
                <div style="margin-top: 10px;">
                    <span class="status-indicator status-{info['status']}"></span>
                    <strong>Status:</strong> {info['status'].title()}
                </div>
                <div style="margin-top: 8px;">
                    <strong>Capabilities:</strong><br>
                    {'<br>'.join([f"• {cap}" for cap in info['capabilities']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # System Health Monitor
    st.subheader("📊 System Health")
    
    if workflow:
        with st.spinner("Checking system health..."):
            try:
                health = asyncio.run(workflow.health_check())
                
                health_metrics = []
                for component, status in health.items():
                    if isinstance(status, bool):
                        health_metrics.append({
                            'Component': component.replace('_', ' ').title(),
                            'Status': '✅ Healthy' if status else '❌ Error',
                            'Value': 100 if status else 0
                        })
                
                if health_metrics:
                    df_health = pd.DataFrame(health_metrics)
                    
                    # Create health chart
                    fig = go.Figure(data=go.Bar(
                        x=df_health['Component'],
                        y=df_health['Value'],
                        marker_color=['#48bb78' if v == 100 else '#f56565' for v in df_health['Value']],
                        text=df_health['Status'],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title="System Component Health",
                        xaxis_title="Components",
                        yaxis_title="Health %",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                if hasattr(workflow, 'get_performance_metrics'):
                    metrics = workflow.get_performance_metrics()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Total Requests", 
                            metrics.get('total_requests', 0),
                            delta=f"{metrics.get('successful_completions', 0)} successful"
                        )
                    with col2:
                        avg_time = metrics.get('avg_response_time', 0)
                        st.metric(
                            "Avg Response", 
                            f"{avg_time:.2f}s",
                            delta=f"{metrics.get('cache_hit_rate', 0):.0%} cache hit"
                        )
                        
            except Exception as e:
                st.error(f"Health check failed: {str(e)}")
    
    st.divider()
    
    # Session Statistics
    st.subheader("📈 Session Stats")
    
    stats = st.session_state.session_stats
    session_duration = datetime.now() - stats['session_start']
    
    st.markdown(f"""
    <div class="conversation-stats">
        <h4>💬 Conversation Analytics</h4>
        <p><strong>Duration:</strong> {str(session_duration).split('.')[0]}</p>
        <p><strong>Messages:</strong> {stats['messages_sent']}</p>
        <p><strong>Agents Used:</strong> {len(stats['agents_used'])}</p>
        <p><strong>Avg Confidence:</strong> {(stats['total_confidence'] / max(stats['messages_sent'], 1)):.0%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance History Chart
    if stats['performance_history']:
        df_perf = pd.DataFrame(stats['performance_history'])
        fig = px.line(df_perf, x='timestamp', y='confidence', 
                     title="Confidence Over Time",
                     labels={'confidence': 'Confidence %', 'timestamp': 'Time'})
        fig.update_traces(line_color='#667eea')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Control Panel
    st.subheader("🎛️ Control Panel")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.session_stats = {
                'messages_sent': 0,
                'agents_used': set(),
                'total_confidence': 0.0,
                'session_start': datetime.now(),
                'performance_history': []
            }
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", type="secondary"):
            st.rerun()

# Main chat interface
st.subheader("💬 Intelligent Conversation")

# Example prompts
with st.expander("💡 Try These Examples", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    example_prompts = [
        "Schedule a team meeting for tomorrow at 2 PM",
        "Send an email to john@example.com about the project update",
        "Search for the latest AI trends in 2025",
        "Analyze my calendar for scheduling conflicts",
        "Help me optimize my daily schedule",
        "Find information about quantum computing advances"
    ]
    
    for i, prompt in enumerate(example_prompts):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(f"💡 {prompt[:30]}...", key=f"example_{i}"):
                st.session_state.example_prompt = prompt

# Display chat messages with enhanced formatting
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Enhanced metadata display for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>🎯 Route:</strong> {metadata.get('route', 'unknown')}<br>
                    <strong>🤖 Agent:</strong> {metadata.get('current_agent', 'unknown')}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                confidence = metadata.get('route_confidence', 0)
                confidence_color = "#48bb78" if confidence > 0.8 else "#ed8936" if confidence > 0.6 else "#f56565"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>📊 Confidence:</strong><br>
                    <span style="color: {confidence_color}; font-weight: bold;">{confidence:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                processing_time = metadata.get('processing_time', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <strong>⏱️ Processing:</strong><br>
                    {processing_time:.2f}s
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed routing information
            with st.expander("🔍 Detailed Analysis"):
                st.json({
                    "routing_decision": metadata.get('route_reason', 'unknown'),
                    "agent_capabilities": metadata.get('agent_capabilities', []),
                    "verification_scores": metadata.get('verification_scores', {}),
                    "resource_usage": metadata.get('resource_usage', {}),
                    "collaboration_requests": metadata.get('collaboration_requests', []),
                    "escalation_triggered": metadata.get('escalation_triggered', False)
                })

# Handle example prompt selection
if 'example_prompt' in st.session_state:
    st.session_state.messages.append({"role": "user", "content": st.session_state.example_prompt})
    with st.chat_message("user"):
        st.markdown(st.session_state.example_prompt)
    del st.session_state.example_prompt
    st.rerun()

# Chat input with enhanced processing
if prompt := st.chat_input("Ask me anything - I can handle emails, calendar, search, and more..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with Enhanced AI Assistant
    with st.chat_message("assistant"):
        processing_placeholder = st.empty()
        processing_placeholder.markdown("🚀 **Initializing Enhanced AI Pipeline...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            start_time = time.time()
            
            # Update progress
            progress_bar.progress(20)
            status_text.text("🧠 Analyzing request and routing to optimal agent...")
            
            # Extract conversation history from session state
            conversation_history = []
            for msg in st.session_state.messages[:-1]:  # Exclude current message
                if msg["role"] == "user":
                    conversation_history.append({"user": "me", "text": msg["content"]})
                elif msg["role"] == "assistant":
                    conversation_history.append({"assistant": "ai", "text": msg["content"]})
            
            progress_bar.progress(40)
            status_text.text("⚡ Executing with full AI capabilities...")
            
            # Run the enhanced workflow
            result_state = asyncio.run(workflow.process_request(prompt, user="demo_user"))
            
            progress_bar.progress(80)
            status_text.text("🔍 Performing quality verification and optimization...")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.text("✅ Complete! Response generated.")
            
            # Clear progress indicators
            time.sleep(0.5)
            processing_placeholder.empty()
            progress_bar.empty()
            status_text.empty()
            
            # Extract response
            response = result_state.get('final_response', 'No response generated')
            
            # Display response
            st.markdown(response)
            
            # Update session statistics
            stats = st.session_state.session_stats
            stats['messages_sent'] += 1
            
            current_agent = result_state.get('current_agent', 'unknown')
            if current_agent != 'unknown':
                stats['agents_used'].add(current_agent)
            
            confidence = result_state.get('route_confidence', 0.0)
            stats['total_confidence'] += confidence
            
            # Add to performance history
            stats['performance_history'].append({
                'timestamp': datetime.now(),
                'confidence': confidence * 100,
                'processing_time': processing_time,
                'agent': current_agent
            })
            
            # Keep only last 20 entries
            if len(stats['performance_history']) > 20:
                stats['performance_history'] = stats['performance_history'][-20:]
            
            # Enhanced metadata collection
            metadata = {
                "route": result_state.get('route', 'unknown'),
                "current_agent": current_agent,
                "route_confidence": confidence,
                "route_reason": result_state.get('route_reason', 'unknown'),
                "processing_time": processing_time,
                "verification_scores": result_state.get('verification_scores', {}),
                "agent_capabilities": result_state.get('agent_capabilities', []),
                "resource_usage": {
                    "tokens_used": safe_get_resource_metric(result_state, 'token_count', 0),
                    "api_calls": safe_get_resource_metric(result_state, 'api_calls', 0),
                    "cache_hits": safe_get_resource_metric(result_state, 'cache_hits', 0)
                },
                "collaboration_requests": result_state.get('collaboration_requests', []),
                "escalation_triggered": result_state.get('escalation_requests', []) != [],
                "optimization_flags": result_state.get('optimization_flags', [])
            }
            
            # Add to chat history with enhanced metadata
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "metadata": metadata
            })
            
            # Show enhanced debug information
            with st.expander("🔧 Enhanced Debug Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Routing Information")
                    st.json({
                        "route": result_state.get('route', 'unknown'),
                        "route_confidence": confidence,
                        "route_reason": result_state.get('route_reason', 'unknown'),
                        "agent_selection": result_state.get('agent_selection_rationale', 'unknown')
                    })
                    
                    st.subheader("⚡ Performance Metrics")
                    st.json({
                        "processing_time": f"{processing_time:.3f}s",
                        "tokens_used": safe_get_resource_metric(result_state, 'token_count', 0),
                        "api_calls": safe_get_resource_metric(result_state, 'api_calls', 0),
                        "cache_efficiency": f"{safe_get_resource_metric(result_state, 'cache_hits', 0)} hits"
                    })
                
                with col2:
                    st.subheader("🤝 Collaboration Status")
                    st.json({
                        "collaboration_requests": len(result_state.get('collaboration_requests', [])),
                        "escalation_requests": len(result_state.get('escalation_requests', [])),
                        "agent_coordination": result_state.get('agent_coordination_status', 'none')
                    })
                    
                    st.subheader("🛡️ Quality Assurance")
                    st.json({
                        "verification_passed": result_state.get('verification_passed', False),
                        "verification_score": result_state.get('verification_score', 0.0),
                        "hallucination_check": result_state.get('hallucination_check_passed', True),
                        "confidence_threshold_met": confidence > 0.7
                    })
            
            # Show errors if any
            if result_state.get('error_log'):
                with st.expander("⚠️ Issues and Recoveries"):
                    for error in result_state['error_log']:
                        error_level = error.get('level', 'error')
                        icon = "🚨" if error_level == 'error' else "⚠️"
                        st.markdown(f"{icon} **{error.get('agent', 'System')}:** {error.get('error', 'Unknown error')}")
                        
                        if 'recovery_action' in error:
                            st.markdown(f"   🔄 **Recovery:** {error['recovery_action']}")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            processing_placeholder.empty()
            
            error_msg = f"🚨 **System Error:** {str(e)}"
            st.error(error_msg)
            
            # Add error to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"I apologize, but I encountered a technical issue: {str(e)}\n\nPlease try rephrasing your request or contact support if the issue persists.",
                "metadata": {"error": True, "error_type": type(e).__name__}
            })

# Footer with system information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 20px;">
    <h4>🚀 Enhanced AI Assistant v2.0</h4>
    <p>
        Powered by <strong>Enhanced LangGraph Workflow</strong> • 
        <strong>Multi-Agent Orchestration</strong> • 
        <strong>Advanced Error Recovery</strong><br>
        🧠 Intelligent • 🤝 Collaborative • 🎯 Autonomous • 🔄 Resilient
    </p>
    <p><em>Built with ❤️ by Zain using cutting-edge AI architecture</em></p>
</div>
""", unsafe_allow_html=True)