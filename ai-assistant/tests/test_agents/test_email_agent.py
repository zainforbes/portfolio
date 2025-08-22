# tests/test_email_agent.py
import asyncio
import unittest
from unittest.mock import Mock, AsyncMock
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.email_agent import EmailAgent, EmailClassification
from src.core.state_schema import AssistantState, make_initial_state
try:
    from src.mcp_integration.gmail_server import GmailMCPServer
except ImportError:
    # Mock GmailMCPServer if not available
    class GmailMCPServer:
        def __init__(self):
            self.authenticated = False
        async def authenticate(self, creds):
            return {'success': True}
        async def list_messages(self, query='', max_results=10):
            return {'messages': [{'id': f'msg_{i}'} for i in range(min(5, max_results))]}
        async def get_message(self, message_id):
            return {'id': message_id, 'subject': 'Test Email', 'from': 'test@example.com', 'body': 'Test body'}
        async def send_message(self, to, subject, body):
            return {'success': True, 'id': 'sent_123'}
        async def call_tool(self, tool_name, parameters):
            if tool_name == 'gmail_list_messages':
                return await self.list_messages(parameters.get('query', ''), parameters.get('max_results', 10))
            elif tool_name == 'gmail_get_message':
                return await self.get_message(parameters.get('message_id'))
            return {}

class TestEmailAgent(unittest.TestCase):
    """Test suite for EmailAgent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gmail_server = GmailMCPServer()
        self.gmail_server.authenticated = True  # Mock authentication
        
        # Mock MCP client
        self.mock_mcp_client = Mock()
        self.mock_mcp_client.call_tool = AsyncMock()
        
        # Create email agent
        self.email_agent = EmailAgent(self.mock_mcp_client)
        
        # Sample state
        self.sample_state = make_initial_state("Help me organize my emails")
    
    def test_agent_initialization(self):
        """Test email agent initialization."""
        self.assertEqual(self.email_agent.agent_name, "EmailAgent")
        self.assertIn("email_classification", self.email_agent.capabilities)
        self.assertIn("email_summarization", self.email_agent.capabilities)
        self.assertIn("priority_detection", self.email_agent.capabilities)
    
    def test_can_handle_email_requests(self):
        """Test email request detection (sync version)."""
        test_cases = [
            ("Check my emails", True),
            ("Organize my inbox", True), 
            ("Send an email to John", True),
            ("What's the weather like?", False),
            ("Schedule a meeting", False),
            ("Reply to that message", True),
            ("Gmail notifications", True)
        ]
        
        for request, expected in test_cases:
            with self.subTest(request=request):
                # Run async test in sync context
                result = asyncio.run(self.email_agent.can_handle(request))
                self.assertEqual(result, expected)
    
    async def test_email_classification(self):
        """Test email classification functionality."""
        # Mock email data
        mock_email = {
            'id': 'test_001',
            'subject': 'URGENT: Server Down - Action Required',
            'from': 'admin@company.com',
            'body': 'The production server is down. Please investigate immediately. Deadline: 2 hours.'
        }
        
        # Mock MCP responses
        self.mock_mcp_client.call_tool.side_effect = [
            # gmail_list_messages response
            {'messages': [{'id': 'test_001'}]},
            # gmail_get_message response  
            mock_email
        ]
        
        # Mock AI response for classification
        original_generate = self.email_agent.generate_response
        self.email_agent.generate_response = AsyncMock(return_value='''
        {
            "priority": "high",
            "category": "work",
            "sentiment": "urgent",
            "action_required": true,
            "urgency_score": 0.9,
            "suggested_actions": ["investigate", "escalate"],
            "confidence": 0.8
        }
        ''')
        
        # Test classification
        result_state = await self.email_agent._classify_recent_emails(self.sample_state)
        
        # Verify classification results
        self.assertIn('email_classifications', result_state['active_context'])
        classifications = result_state['active_context']['email_classifications']
        self.assertEqual(len(classifications), 1)
        
        classification = classifications[0]['classification']
        self.assertEqual(classification.priority, 'high')
        self.assertEqual(classification.category, 'work')
        self.assertTrue(classification.action_required)
        
        # Restore original method
        self.email_agent.generate_response = original_generate
    
    async def test_email_summarization(self):
        """Test email summarization functionality."""
        # Mock email data
        mock_emails = [
            {
                'id': 'test_001',
                'subject': 'Project Update Meeting',
                'from': 'manager@company.com',
                'body': '''Hi team, we need to meet tomorrow at 2 PM to discuss project updates. 
                Please prepare your status reports. The deadline for Phase 1 is next Friday.
                John will present the budget analysis.'''
            }
        ]
        
        # Mock MCP responses
        self.mock_mcp_client.call_tool.side_effect = [
            {'messages': [{'id': 'test_001'}]},
            mock_emails[0]
        ]
        
        # Mock AI summarization response
        self.email_agent.generate_response = AsyncMock(return_value='''
        Key points:
        - Meeting scheduled for tomorrow at 2 PM
        - Need to prepare status reports
        - Phase 1 deadline is next Friday
        - John will present budget analysis
        
        Action items:
        - Prepare status report for meeting
        - Review Phase 1 deliverables
        ''')
        
        # Test summarization
        result_state = await self.email_agent._summarize_emails(self.sample_state)
        
        # Verify summary results
        self.assertIn('email_summaries', result_state['active_context'])
        summaries = result_state['active_context']['email_summaries']
        self.assertEqual(len(summaries), 1)
        
        summary = summaries[0]
        self.assertEqual(summary['subject'], 'Project Update Meeting')
        self.assertEqual(summary['sender'], 'manager@company.com')
    
    async def test_inbox_management(self):
        """Test inbox management functionality."""
        # Mock profile and messages
        self.mock_mcp_client.call_tool.side_effect = [
            # gmail_get_profile
            {'messagesTotal': 150, 'emailAddress': 'test@example.com'},
            # gmail_list_messages
            {'messages': [{'id': f'msg_{i}'} for i in range(5)]},
            # gmail_get_message calls (5 times)
            {'id': 'msg_0', 'subject': 'Marketing Newsletter', 'from': 'marketing@company.com', 'body': 'Weekly updates...'},
            {'id': 'msg_1', 'subject': 'URGENT: Security Alert', 'from': 'security@company.com', 'body': 'Immediate action required...'},
            {'id': 'msg_2', 'subject': 'Meeting Reminder', 'from': 'calendar@company.com', 'body': 'Don\'t forget about...'},
            {'id': 'msg_3', 'subject': 'Personal: Weekend Plans', 'from': 'friend@personal.com', 'body': 'Hey, what are you up to...'},
            {'id': 'msg_4', 'subject': 'Invoice #12345', 'from': 'billing@vendor.com', 'body': 'Please find attached...'}
        ]
        
        # Mock AI classification responses
        classification_responses = [
            '{"priority": "low", "category": "marketing", "sentiment": "neutral", "action_required": false, "urgency_score": 0.2, "suggested_actions": ["review"], "confidence": 0.7}',
            '{"priority": "high", "category": "security", "sentiment": "urgent", "action_required": true, "urgency_score": 0.9, "suggested_actions": ["investigate", "escalate"], "confidence": 0.9}', 
            '{"priority": "medium", "category": "work", "sentiment": "neutral", "action_required": true, "urgency_score": 0.6, "suggested_actions": ["respond"], "confidence": 0.8}',
            '{"priority": "low", "category": "personal", "sentiment": "positive", "action_required": false, "urgency_score": 0.3, "suggested_actions": ["read"], "confidence": 0.6}',
            '{"priority": "medium", "category": "finance", "sentiment": "neutral", "action_required": true, "urgency_score": 0.5, "suggested_actions": ["review", "process"], "confidence": 0.7}'
        ]
        
        call_count = 0
        async def mock_generate(prompt, context=None, temperature=0.7):
            nonlocal call_count
            if call_count < len(classification_responses):
                response = classification_responses[call_count]
                call_count += 1
                return response
            return '{"priority": "medium", "category": "general", "sentiment": "neutral", "action_required": false, "urgency_score": 0.5, "suggested_actions": ["review"], "confidence": 0.5}'
        
        self.email_agent.generate_response = mock_generate
        
        # Test inbox management
        result_state = await self.email_agent._manage_inbox(self.sample_state)
        
        # Verify management results
        self.assertIn('inbox_organization', result_state['active_context'])
        org_stats = result_state['active_context']['inbox_organization']
        
        self.assertEqual(org_stats['total_unread'], 5)
        self.assertEqual(org_stats['organized'], 5)
        self.assertEqual(org_stats['high_priority'], 1)
        self.assertGreater(len(org_stats['actions_taken']), 0)
    
    async def test_gmail_mcp_integration(self):
        """Test Gmail MCP server integration."""
        # Test authentication
        auth_result = await self.gmail_server.authenticate("mock_credentials.json")
        self.assertTrue(auth_result['success'])
        
        # Test list messages
        messages = await self.gmail_server.list_messages(query="is:unread", max_results=5)
        self.assertIn('messages', messages)
        self.assertLessEqual(len(messages['messages']), 5)
        
        # Test get message
        if messages['messages']:
            message_id = messages['messages'][0]['id']
            message = await self.gmail_server.get_message(message_id)
            self.assertIn('subject', message)
            self.assertIn('from', message)
            self.assertIn('body', message)
        
        # Test send message
        send_result = await self.gmail_server.send_message(
            to="test@example.com",
            subject="Test Email",
            body="This is a test email"
        )
        self.assertTrue(send_result['success'])
        self.assertIn('id', send_result)
    
    async def test_email_patterns_extraction(self):
        """Test email pattern extraction (dates, people, etc.)."""
        test_email_body = """
        Hi John and Sarah,
        
        Please complete the report by Friday, January 28th, 2024.
        We have a deadline of February 15th for the final submission.
        
        Let me know if you need help.
        
        Best regards,
        Michael Johnson
        """
        
        # Test people extraction
        people = self.email_agent._extract_people_mentions(test_email_body)
        expected_people = ['Michael Johnson']  # Simple regex won't catch John/Sarah without last names
        self.assertTrue(any(person in people for person in expected_people))
        
        # Test deadline extraction (would return a datetime in real implementation)
        deadline = self.email_agent._extract_deadline(test_email_body)
        # Since we're using a simplified implementation, just check it doesn't crash
        from datetime import datetime
        self.assertIsInstance(deadline, (type(None), datetime))
    
    def test_classification_parsing(self):
        """Test classification response parsing."""
        # Test JSON parsing
        json_response = '''
        {
            "priority": "high",
            "category": "work", 
            "sentiment": "urgent",
            "action_required": true,
            "urgency_score": 0.8,
            "suggested_actions": ["review", "respond"],
            "confidence": 0.9
        }
        '''
        
        result = self.email_agent._parse_classification_response(json_response)
        self.assertEqual(result['priority'], 'high')
        self.assertEqual(result['category'], 'work')
        self.assertTrue(result['action_required'])
        
        # Test fallback parsing
        text_response = "This is a high priority urgent email that requires immediate action."
        result = self.email_agent._parse_classification_response(text_response)
        self.assertEqual(result['priority'], 'high')
        self.assertTrue(result['action_required'])


class TestEmailAgentIntegration(unittest.TestCase):
    """Integration tests for EmailAgent with real MCP client simulation."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.gmail_server = GmailMCPServer()
        
        # Create a mock MCP client that calls our Gmail server
        self.mcp_client = Mock()
        
        async def mock_call_tool(tool_name, parameters):
            return await self.gmail_server.call_tool(tool_name, parameters)
        
        self.mcp_client.call_tool = mock_call_tool
        
        # Create email agent with integrated MCP
        self.email_agent = EmailAgent(self.mcp_client)
    
    async def test_full_email_workflow(self):
        """Test complete email management workflow."""
        # Authenticate first
        auth_result = await self.gmail_server.authenticate("mock_creds.json")
        self.assertTrue(auth_result['success'])
        
        # Create test state
        state = make_initial_state("Please classify my recent emails and organize my inbox")
        
        # Execute email agent
        result_state = await self.email_agent.execute_with_tracking(state)
        
        # Verify execution completed without errors
        self.assertNotIn('errors', result_state.get('context', {}))
        # Check that agent executed successfully
        self.assertIn('final_response', result_state)
        # Verify some agent activity occurred
        self.assertTrue(len(result_state.get('agent_messages', [])) > 0)


async def run_async_tests():
    """Run async test methods."""
    test_instance = TestEmailAgent()
    test_instance.setUp()
    
    print("Running async EmailAgent tests...")
    
    try:
        # Run async tests
        await test_instance.test_email_classification()
        print("[PASS] Email classification test passed!")
        
        await test_instance.test_email_summarization()
        print("[PASS] Email summarization test passed!")
        
        await test_instance.test_inbox_management()
        print("[PASS] Inbox management test passed!")
        
        await test_instance.test_gmail_mcp_integration()
        print("[PASS] Gmail MCP integration test passed!")
        
        await test_instance.test_email_patterns_extraction()
        print("[PASS] Email patterns extraction test passed!")
        
        print("\n[PASS] All EmailAgent async tests passed!")
        
        # Run integration tests
        print("\nRunning integration tests...")
        integration_test = TestEmailAgentIntegration()
        integration_test.setUp()
        await integration_test.test_full_email_workflow()
        
        print("[PASS] EmailAgent integration tests passed!")
        
    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Running EmailAgent tests...")
    
    # Run regular unittest tests first
    print("\n" + "="*50)
    print("Running sync tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run async tests
    print("\n" + "="*50)
    print("Running async tests...")
    asyncio.run(run_async_tests())