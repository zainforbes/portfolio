from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio

from src.core.state_schema import AssistantState
from src.core.message_types import AgentMessage

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in the system.
    Provides common functionality and enforces consistent interface.
    """
    
    def __init__(self, 
                 gemini_mcp_client,
                 agent_name: str,
                 capabilities: List[str] = None):
        """
        Initialize the base agent with shared components.
        
        Args:
            gemini_mcp_client: Gemini MCP client for all AI operations and tool access
            agent_name: Unique name for this agent
            capabilities: List of capabilities this agent provides
        """
        self.gemini_mcp_client = gemini_mcp_client
        self.agent_name = agent_name
        self.capabilities = capabilities or []
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.logger.setLevel(logging.INFO)
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'average_response_time': 0.0,
            'last_execution': None
        }
        
        # Prompt templates - can be overridden by specific agents
        self.system_prompts = {
            'base': f"""You are {agent_name}, a specialized AI agent.
Your capabilities include: {', '.join(self.capabilities)}
Always provide clear, actionable responses.
Focus on your specific domain of expertise.""",
            
            'error_handling': """When encountering errors:
1. Log the issue clearly
2. Attempt reasonable recovery if possible
3. Provide helpful error messages to the user
4. Escalate to coordinator if needed""",
            
            'collaboration': """When working with other agents:
1. Communicate clearly and concisely
2. Share relevant context
3. Respect other agents' expertise
4. Coordinate to avoid conflicts"""
        }
    
    @abstractmethod
    async def execute(self, state: AssistantState) -> AssistantState:
        """
        Execute the agent's primary function.
        
        Args:
            state: Current assistant state
            
        Returns:
            Updated assistant state
        """
        pass
    
    async def can_handle(self, request: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if this agent can handle the given request.
        
        Args:
            request: User request or task description
            context: Additional context information
            
        Returns:
            True if agent can handle the request
        """
        # Default implementation - override in specific agents
        return False
    
    async def get_tools(self) -> List[str]:
        """Get list of MCP tools available to this agent."""
        try:
            tools = await self.mcp_client.list_tools()
            # Filter tools relevant to this agent's capabilities
            relevant_tools = [
                tool for tool in tools 
                if any(cap.lower() in tool.lower() for cap in self.capabilities)
            ]
            return relevant_tools
        except Exception as e:
            self.logger.error(f"Error getting tools: {e}")
            return []
    
    async def use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Use an MCP tool via Gemini MCP client with error handling and logging.
        
        Args:
            tool_name: Name of the tool to use
            parameters: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        try:
            self.logger.info(f"Using tool: {tool_name} with params: {parameters}")
            
            # Route different tools through the appropriate Gemini MCP client methods
            if tool_name.startswith('gmail_') or tool_name in ['list_emails', 'read_email', 'send_email']:
                # Use Gmail tools through Gemini MCP client
                if tool_name == 'list_emails' or tool_name == 'gmail_list_messages':
                    result = await self.gemini_mcp_client._list_emails(
                        parameters.get('query', ''),
                        parameters.get('max_results', 10)
                    )
                elif tool_name == 'read_email' or tool_name == 'gmail_get_message':
                    result = await self.gemini_mcp_client._read_email(
                        parameters.get('message_id', '')
                    )
                elif tool_name == 'send_email' or tool_name == 'gmail_send_message':
                    result = await self.gemini_mcp_client._send_email(
                        parameters.get('to', ''),
                        parameters.get('subject', ''),
                        parameters.get('body', '')
                    )
                else:
                    result = {"error": f"Unknown Gmail tool: {tool_name}"}
                    
            elif tool_name.startswith('calendar_') or tool_name in ['list_calendar_events', 'create_calendar_event']:
                # Use Calendar tools through Gemini MCP client
                if tool_name == 'list_calendar_events' or tool_name == 'calendar_list_events':
                    result = await self.gemini_mcp_client._list_calendar_events(
                        parameters.get('days_ahead', 7)
                    )
                elif tool_name == 'create_calendar_event' or tool_name == 'calendar_create_event':
                    result = await self.gemini_mcp_client._create_calendar_event(
                        parameters.get('title', ''),
                        parameters.get('start_time', ''),
                        parameters.get('end_time', ''),
                        parameters.get('description', '')
                    )
                else:
                    result = {"error": f"Unknown Calendar tool: {tool_name}"}
                    
            elif tool_name.startswith('search_') or tool_name in ['search_web', 'web_search', 'brave_search']:
                # Use search tools through Gemini MCP client
                if tool_name == 'search_web' or tool_name == 'web_search':
                    result = await self.gemini_mcp_client._search_web(
                        parameters.get('query', ''),
                        parameters.get('count', 10)
                    )
                elif tool_name == 'brave_search':
                    result = await self.gemini_mcp_client._brave_search(
                        parameters.get('query', ''),
                        parameters.get('count', 10),
                        **{k: v for k, v in parameters.items() if k not in ['query', 'count']}
                    )
                elif tool_name == 'search_summarize':
                    result = await self.gemini_mcp_client._search_summarize(
                        parameters.get('query', ''),
                        parameters.get('max_results', 5)
                    )
                elif tool_name == 'search_analyze':
                    result = await self.gemini_mcp_client._search_analyze(
                        parameters.get('query', ''),
                        parameters.get('include_domains', True),
                        parameters.get('include_keywords', True)
                    )
                else:
                    result = {"error": f"Unknown search tool: {tool_name}"}
                
            elif tool_name == 'gemini_generate':
                # Use Gemini directly through MCP client
                result = await self.gemini_mcp_client.chat(parameters.get('prompt', ''))
                
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            self.logger.info(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Tool {tool_name} failed: {e}")
            raise
    
    async def generate_response(self, 
                              prompt: str, 
                              context: str = None,
                              temperature: float = 0.7) -> str:
        """
        Generate response using Gemini via MCP tools.
        
        Args:
            prompt: Main prompt for generation
            context: Additional context
            temperature: Generation temperature
            
        Returns:
            Generated response
        """
        try:
            # Combine system prompt with specific prompt
            full_prompt = f"{self.system_prompts['base']}\n\n"
            if context:
                full_prompt += f"Context: {context}\n\n"
            full_prompt += f"Request: {prompt}"
            
            # Use Gemini MCP client for all AI generation
            response = await self.gemini_mcp_client.chat(full_prompt)
            return response if response else 'No response generated'
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def send_message(self, 
                          recipient: str, 
                          message_type: str, 
                          payload: Dict[str, Any]) -> AgentMessage:
        """
        Send message to another agent.
        
        Args:
            recipient: Target agent name
            message_type: Type of message
            payload: Message content
            
        Returns:
            Formatted agent message
        """
        message = AgentMessage(
            sender=self.agent_name,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.utcnow()
        )
        
        self.logger.info(f"Sending {message_type} message to {recipient}")
        return message
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Handle incoming message from another agent.
        
        Args:
            message: Incoming agent message
            
        Returns:
            Optional response message
        """
        self.logger.info(f"Received {message.message_type} from {message.sender}")
        
        # Default message handling - override in specific agents
        if message.message_type == "ping":
            return await self.send_message(
                message.sender,
                "pong",
                {"status": "active", "capabilities": self.capabilities}
            )
        
        return None
    
    def update_stats(self, execution_time: float, success: bool):
        """Update agent performance statistics."""
        self.execution_stats['total_executions'] += 1
        if success:
            self.execution_stats['successful_executions'] += 1
        
        # Update average response time
        total = self.execution_stats['total_executions']
        current_avg = self.execution_stats['average_response_time']
        self.execution_stats['average_response_time'] = (
            (current_avg * (total - 1) + execution_time) / total
        )
        
        self.execution_stats['last_execution'] = datetime.utcnow()
    
    async def execute_with_tracking(self, state: AssistantState) -> AssistantState:
        """
        Execute the agent with performance tracking and error handling.
        
        Args:
            state: Current assistant state (TypedDict)
            
        Returns:
            Updated assistant state
        """
        start_time = datetime.utcnow()
        success = False
        
        try:
            # Execute the agent's main functionality
            updated_state = await self.execute(state)
            success = True
            
            # Update execution history in state using your schema
            completed_tasks = updated_state.get('completed_tasks', [])
            completed_tasks.append({
                'agent': self.agent_name,
                'timestamp': start_time.isoformat(),
                'success': success,
                'execution_time': (datetime.utcnow() - start_time).total_seconds(),
                'task_type': updated_state.get('task_type', 'unknown')
            })
            updated_state['completed_tasks'] = completed_tasks
            
            # Update current agent
            updated_state['current_agent'] = self.agent_name
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            
            # Add error to state using your schema
            error_log = state.get('error_log', [])
            error_log.append({
                'agent': self.agent_name,
                'error': str(e),
                'timestamp': start_time.isoformat(),
                'retry_count': state.get('retry_count', 0)
            })
            state['error_log'] = error_log
            
            return state
        
        finally:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.update_stats(execution_time, success)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and performance metrics."""
        return {
            'name': self.agent_name,
            'capabilities': self.capabilities,
            'status': 'active',
            'stats': self.execution_stats.copy()
        }