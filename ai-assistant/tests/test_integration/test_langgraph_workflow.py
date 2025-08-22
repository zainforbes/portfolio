# tests/test_workflow/test_langgraph_workflow.py
import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.core.langgraph_workflow import AIAssistantWorkflow, process_user_request
from src.core.state_schema import make_initial_state, AssistantState

class TestAIAssistantWorkflow(unittest.TestCase):
    """Test suite for AIAssistantWorkflow functionality."""
    
    def setUp(self):
        """Set up test fixtures with extensive mocking."""
        # Create workflow with mocked dependencies
        with patch('src.mcp_integration.gemini_mcp_client.GeminiMCPClient') as mock_gemini, \
             patch('src.mcp_integration.mcp_client.MCPClient') as mock_mcp, \
             patch('src.core.orchestrator.CoreOrchestrator') as mock_orchestrator, \
             patch('src.agents.email_agent.EmailAgent') as mock_email_agent, \
             patch('src.agents.calendar_agent.CalendarAgent') as mock_calendar_agent, \
             patch('src.agents.task_agent.TaskAgent') as mock_task_agent, \
             patch('src.agents.coordinator_agent.CoordinatorAgent') as mock_coordinator_agent:
            
            # Setup mock instances
            self.mock_gemini_client = mock_gemini.return_value
            self.mock_mcp_client = mock_mcp.return_value
            self.mock_orchestrator = mock_orchestrator.return_value
            
            # Setup mock agents
            self.mock_email_agent = mock_email_agent.return_value
            self.mock_calendar_agent = mock_calendar_agent.return_value
            self.mock_task_agent = mock_task_agent.return_value
            self.mock_coordinator_agent = mock_coordinator_agent.return_value
            
            # Setup async methods
            self.mock_orchestrator.route_request = AsyncMock()
            self.mock_orchestrator.verify_response = AsyncMock()
            self.mock_orchestrator.handle_fallback = AsyncMock()
            self.mock_orchestrator.health_check = AsyncMock()
            
            self.mock_email_agent.execute_with_tracking = AsyncMock()
            self.mock_calendar_agent.execute_with_tracking = AsyncMock()
            self.mock_task_agent.execute_with_tracking = AsyncMock()
            self.mock_coordinator_agent.execute_with_tracking = AsyncMock()
            
            # Create the workflow
            self.workflow = AIAssistantWorkflow()
    
    def test_workflow_initialization(self):
        """Test workflow initialization and component setup."""
        # Check that all components are initialized
        self.assertIsNotNone(self.workflow.gemini_client)
        self.assertIsNotNone(self.workflow.mcp_client)
        self.assertIsNotNone(self.workflow.orchestrator)
        
        # Check agents are set up
        agents = self.workflow.agents
        self.assertIn('email', agents)
        self.assertIn('calendar', agents)
        self.assertIn('task', agents)
        self.assertIn('coordinator', agents)
        
        # Check workflow graph is compiled
        self.assertIsNotNone(self.workflow.workflow)
    
    async def test_route_request_node(self):
        """Test the route_request node functionality."""
        # Setup mock routing response
        self.mock_orchestrator.route_request.return_value = {
            'route': 'email',
            'confidence': 0.85,
            'reason': 'Request involves email management',
            'task_type': 'email_organization'
        }
        
        # Create test state
        state = make_initial_state("Help me organize my emails")
        
        # Test routing
        result_state = await self.workflow._route_request(state)
        
        # Verify routing was called correctly
        self.mock_orchestrator.route_request.assert_called_once_with("Help me organize my emails")
        
        # Verify state was updated
        self.assertEqual(result_state['route'], 'email')
        self.assertEqual(result_state['route_confidence'], 0.85)
        self.assertEqual(result_state['task_type'], 'email_organization')
    
    async def test_route_request_with_error(self):
        """Test route_request node with orchestrator error."""
        # Setup mock routing failure
        self.mock_orchestrator.route_request.side_effect = Exception("Routing failed")
        
        # Create test state
        state = make_initial_state("Test request")
        
        # Test routing with error
        result_state = await self.workflow._route_request(state)
        
        # Verify fallback route was set
        self.assertEqual(result_state['route'], 'fallback')
        self.assertIn('error_log', result_state)
        self.assertEqual(len(result_state['error_log']), 1)
        self.assertIn('Routing failed', result_state['error_log'][0]['error'])
    
    async def test_email_agent_execution(self):
        """Test email agent execution node."""
        # Setup mock agent response
        expected_state = make_initial_state("Test email request")
        expected_state['final_response'] = "Email task completed successfully"
        expected_state['current_agent'] = 'email'
        
        self.mock_email_agent.execute_with_tracking.return_value = expected_state
        
        # Create test state
        state = make_initial_state("Check my emails")
        
        # Test email agent execution
        result_state = await self.workflow._execute_email_agent(state)
        
        # Verify agent was called
        self.mock_email_agent.execute_with_tracking.assert_called_once()
        
        # Verify state was updated
        self.assertEqual(result_state['current_agent'], 'email')
        self.assertEqual(result_state['final_response'], "Email task completed successfully")
    
    async def test_verification_node(self):
        """Test response verification node."""
        # Setup mock verification response
        self.mock_orchestrator.verify_response.return_value = {
            'quality': 0.8,
            'completeness': 0.9,
            'accuracy': 0.85,
            'helpfulness': 0.9
        }
        
        # Create test state with response
        state = make_initial_state("Test request")
        state['final_response'] = "Test response"
        state['current_agent'] = 'email'
        
        # Test verification
        result_state = await self.workflow._verify_response(state)
        
        # Verify verification was called correctly
        self.mock_orchestrator.verify_response.assert_called_once_with(
            "Test request", "Test response", "email"
        )
        
        # Verify verification scores were stored
        scores = result_state['verification_scores']
        self.assertEqual(scores['quality'], 0.8)
        self.assertEqual(scores['completeness'], 0.9)
    
    async def test_fallback_handling(self):
        """Test fallback handling node."""
        # Setup mock fallback response
        self.mock_orchestrator.handle_fallback.return_value = "I apologize, but I couldn't process your request. I can help with email, calendar, and task management."
        
        # Create test state with errors
        state = make_initial_state("Unclear request")
        state['error_log'] = [
            {'error': 'Routing failed', 'timestamp': datetime.now().isoformat()}
        ]
        
        # Test fallback handling
        result_state = await self.workflow._handle_fallback(state)
        
        # Verify fallback was called correctly
        self.mock_orchestrator.handle_fallback.assert_called_once_with(
            "Unclear request", state['error_log']
        )
        
        # Verify fallback response was set
        self.assertIn("I apologize", result_state['final_response'])
        self.assertTrue(result_state['fallback_triggered'])
        self.assertEqual(result_state['current_agent'], 'fallback')
    
    def test_route_decision_logic(self):
        """Test routing decision logic."""
        test_cases = [
            # (route, confidence, expected_decision)
            ('email', 0.8, 'email'),
            ('calendar', 0.7, 'calendar'),
            ('task', 0.9, 'task'),
            ('multi_agent', 0.75, 'coordinator'),
            ('complex', 0.8, 'coordinator'),
            ('email', 0.5, 'fallback'),  # Low confidence
            ('unknown_route', 0.9, 'fallback')  # Unknown route
        ]
        
        for route, confidence, expected in test_cases:
            with self.subTest(route=route, confidence=confidence):
                state = make_initial_state("Test")
                state['route'] = route
                state['route_confidence'] = confidence
                
                decision = self.workflow._route_decision(state)
                self.assertEqual(decision, expected)
    
    def test_verification_decision_logic(self):
        """Test verification decision logic."""
        test_cases = [
            # (quality, completeness, retry_count, expected)
            (0.8, 0.8, 0, 'success'),  # Good quality
            (0.5, 0.6, 0, 'retry'),    # Medium quality, can retry
            (0.5, 0.6, 2, 'fallback'), # Medium quality, max retries
            (0.3, 0.3, 0, 'fallback'), # Poor quality
            (0.8, 0.6, 1, 'success'),  # Good enough overall
        ]
        
        for quality, completeness, retry_count, expected in test_cases:
            with self.subTest(quality=quality, completeness=completeness):
                state = make_initial_state("Test")
                state['verification_scores'] = {
                    'quality': quality,
                    'completeness': completeness
                }
                state['retry_count'] = retry_count
                
                decision = self.workflow._verification_decision(state)
                self.assertEqual(decision, expected)
    
    async def test_health_check(self):
        """Test workflow health check functionality."""
        # Setup mock health check responses
        self.mock_orchestrator.health_check.return_value = True
        
        # Mock agent status
        for agent in self.workflow.agents.values():
            agent.get_status = Mock(return_value={'status': 'active'})
        
        # Run health check
        health_status = await self.workflow.health_check()
        
        # Verify health check structure
        self.assertIn('workflow', health_status)
        self.assertIn('orchestrator', health_status)
        self.assertIn('agents', health_status)
        self.assertIn('clients', health_status)
        
        # Verify orchestrator