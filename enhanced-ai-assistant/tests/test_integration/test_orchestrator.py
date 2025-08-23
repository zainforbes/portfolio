# tests/test_core/test_orchestrator.py
import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.core.orchestrator import CoreOrchestrator
from src.core.state_schema import make_initial_state

class TestCoreOrchestrator(unittest.TestCase):
    """Test suite for CoreOrchestrator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock clients
        self.mock_gemini_client = Mock()
        self.mock_mcp_client = Mock()
        self.mock_mcp_client.call_tool = AsyncMock()
        
        # Create orchestrator
        self.orchestrator = CoreOrchestrator(
            self.mock_gemini_client, 
            self.mock_mcp_client
        )
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator.gemini_client)
        self.assertIsNotNone(self.orchestrator.mcp_client)
        self.assertIn('email', self.orchestrator.route_patterns)
        self.assertIn('calendar', self.orchestrator.route_patterns)
        self.assertIn('task', self.orchestrator.route_patterns)
        
        # Check initial stats
        stats = self.orchestrator.routing_stats
        self.assertEqual(stats['total_requests'], 0)
        self.assertEqual(stats['successful_routes'], 0)
    
    async def test_keyword_analysis(self):
        """Test keyword-based routing analysis."""
        test_cases = [
            ("Check my emails", "email"),
            ("Schedule a meeting tomorrow", "calendar"),
            ("What are my priority tasks?", "task"),
            ("Random unrelated query", "fallback")
        ]
        
        for user_input, expected_route in test_cases:
            with self.subTest(user_input=user_input):
                analysis = await self.orchestrator._analyze_keywords(user_input)
                
                if expected_route == "fallback":
                    self.assertEqual(analysis['confidence'], 0.0)
                else:
                    self.assertEqual(analysis['best_route'], expected_route)
                    self.assertGreater(analysis['confidence'], 0.0)
    
    async def test_ai_analysis_success(self):
        """Test AI-based routing analysis with successful response."""
        # Mock successful AI response
        self.mock_mcp_client.call_tool.return_value = {
            'text': '''
            {
                "category": "email",
                "confidence": 0.8,
                "reasoning": "User wants to manage emails",
                "task_type": "email_organization"
            }
            '''
        }
        
        analysis = await self.orchestrator._analyze_with_ai("Help me organize my inbox")
        
        self.assertEqual(analysis['best_route'], 'email')
        self.assertEqual(analysis['confidence'], 0.8)
        self.assertIn('reasoning', analysis)
        self.assertEqual(analysis['method'], 'ai_analysis')
    
    async def test_ai_analysis_fallback(self):
        """Test AI analysis with MCP failure fallback."""
        # Mock MCP failure, but successful direct Gemini call
        self.mock_mcp_client.call_tool.side_effect = Exception("MCP failed")
        
        # Mock direct Gemini client
        mock_response = Mock()
        mock_response.text = "This request is about email management with high confidence."
        self.mock_gemini_client.generate_content_async = AsyncMock(return_value=mock_response)
        
        analysis = await self.orchestrator._analyze_with_ai("Check my emails")
        
        # Should fallback to keyword-based analysis within the method
        self.assertIn('best_route', analysis)
        self.assertEqual(analysis['method'], 'ai_analysis')
    
    async def test_complex_request_detection(self):
        """Test detection of complex requests requiring multiple agents."""
        complex_requests = [
            "Schedule a meeting and send an email about it",
            "Organize my emails and update my task priorities",
            "Check my calendar, prioritize tasks, and send status email"
        ]
        
        simple_requests = [
            "What's on my calendar today?",
            "Send an email to John",
            "What are my tasks?"
        ]
        
        for request in complex_requests:
            with self.subTest(request=request):
                is_complex = self.orchestrator._is_complex_request(request)
                self.assertTrue(is_complex, f"Should detect complexity in: {request}")
        
        for request in simple_requests:
            with self.subTest(request=request):
                is_complex = self.orchestrator._is_complex_request(request)
                self.assertFalse(is_complex, f"Should not detect complexity in: {request}")
    
    async def test_full_routing_workflow(self):
        """Test complete routing workflow with mocked AI responses."""
        # Mock AI response
        self.mock_mcp_client.call_tool.return_value = {
            'text': '''
            {
                "category": "calendar",
                "confidence": 0.9,
                "reasoning": "User wants to schedule a meeting",
                "task_type": "scheduling"
            }
            '''
        }
        
        result = await self.orchestrator.route_request("Schedule a team meeting for next Tuesday")
        
        # Verify routing result
        self.assertEqual(result['route'], 'calendar')
        self.assertGreater(result['confidence'], 0.0)
        self.assertIn('reason', result)
        self.assertIn('task_type', result)
        
        # Check stats were updated
        stats = self.orchestrator.routing_stats
        self.assertEqual(stats['total_requests'], 1)
        self.assertEqual(stats['successful_routes'], 1)
    
    async def test_response_verification(self):
        """Test response quality verification."""
        # Mock AI verification response
        self.mock_mcp_client.call_tool.return_value = {
            'text': '''
            {
                "quality": 0.8,
                "completeness": 0.9,
                "accuracy": 0.85,
                "helpfulness": 0.9,
                "explanation": "Good response that addresses the user's needs"
            }
            '''
        }
        
        user_input = "Help me organize my emails"
        agent_response = "I've organized your emails by priority. High priority emails are now at the top."
        
        verification = await self.orchestrator.verify_response(
            user_input, agent_response, "EmailAgent"
        )
        
        # Check verification scores
        self.assertEqual(verification['quality'], 0.8)
        self.assertEqual(verification['completeness'], 0.9)
        self.assertEqual(verification['accuracy'], 0.85)
        self.assertEqual(verification['helpfulness'], 0.9)
        self.assertEqual(verification['agent'], 'EmailAgent')
        self.assertIn('verification_timestamp', verification)
    
    async def test_fallback_handling(self):
        """Test fallback response generation."""
        # Mock AI fallback response
        self.mock_mcp_client.call_tool.return_value = {
            'text': "I understand you need help, but I'm having trouble with your specific request. I can help with email, calendar, and task management. Could you be more specific?"
        }
        
        error_log = [
            {'error': 'Routing failed', 'timestamp': datetime.now().isoformat()},
            {'error': 'Agent execution failed', 'timestamp': datetime.now().isoformat()}
        ]
        
        fallback_response = await self.orchestrator.handle_fallback(
            "Some unclear request", error_log
        )
        
        self.assertIsInstance(fallback_response, str)
        self.assertGreater(len(fallback_response), 0)
        self.assertIn("help", fallback_response.lower())
    
    async def test_fallback_with_ai_failure(self):
        """Test fallback when AI response generation also fails."""
        # Mock AI failure
        self.mock_mcp_client.call_tool.side_effect = Exception("AI failed")
        self.mock_gemini_client.generate_content_async = AsyncMock(side_effect=Exception("Gemini failed"))
        
        fallback_response = await self.orchestrator.handle_fallback("Help me")
        
        # Should get default fallback response
        self.assertIsInstance(fallback_response, str)
        self.assertIn("assistance", fallback_response.lower())
        self.assertIn("email", fallback_response.lower())
        self.assertIn("calendar", fallback_response.lower())
        self.assertIn("task", fallback_response.lower())
    
    async def test_health_check(self):
        """Test orchestrator health check."""
        # Mock successful routing for health check
        self.mock_mcp_client.call_tool.return_value = {
            'text': '{"category": "fallback", "confidence": 0.5, "reasoning": "health check"}'
        }
        
        is_healthy = await self.orchestrator.health_check()
        self.assertTrue(is_healthy)
    
    async def test_health_check_failure(self):
        """Test health check with failure."""
        # Mock routing failure
        self.mock_mcp_client.call_tool.side_effect = Exception("Health check failed")
        self.mock_gemini_client.generate_content_async = AsyncMock(side_effect=Exception("Gemini down"))
        
        is_healthy = await self.orchestrator.health_check()
        self.assertFalse(is_healthy)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.orchestrator.get_stats()
        
        self.assertIn('routing_stats', stats)
        self.assertIn('route_patterns', stats)
        self.assertIn('status', stats)
        self.assertEqual(stats['status'], 'active')
        
        # Check route pattern counts
        route_patterns = stats['route_patterns']
        self.assertGreater(route_patterns['email'], 0)
        self.assertGreater(route_patterns['calendar'], 0)
        self.assertGreater(route_patterns['task'], 0)


class TestOrchestratorIntegration(unittest.TestCase):
    """Integration tests for orchestrator with mock components."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Create more realistic mocks
        self.mock_gemini_client = Mock()
        self.mock_mcp_client = Mock()
        
        # Setup realistic async responses
        async def mock_call_tool(tool_name, parameters):
            if tool_name == "gemini_generate":
                prompt = parameters.get("prompt", "")
                
                # Simulate realistic AI responses based on prompt content
                if "email" in prompt.lower():
                    return {
                        'text': '''
                        {
                            "category": "email",
                            "confidence": 0.85,
                            "reasoning": "Request involves email management",
                            "task_type": "email_management"
                        }
                        '''
                    }
                elif "calendar" in prompt.lower() or "schedule" in prompt.lower():
                    return {
                        'text': '''
                        {
                            "category": "calendar", 
                            "confidence": 0.9,
                            "reasoning": "Request involves scheduling",
                            "task_type": "scheduling"
                        }
                        '''
                    }
                elif "task" in prompt.lower():
                    return {
                        'text': '''
                        {
                            "category": "task",
                            "confidence": 0.8,
                            "reasoning": "Request involves task management", 
                            "task_type": "task_management"
                        }
                        '''
                    }
                elif "quality" in prompt.lower():  # Verification prompt
                    return {
                        'text': '''
                        {
                            "quality": 0.8,
                            "completeness": 0.85,
                            "accuracy": 0.9,
                            "helpfulness": 0.85,
                            "explanation": "Response adequately addresses the request"
                        }
                        '''
                    }
                else:
                    return {
                        'text': "I can help with email, calendar, and task management. Please be more specific."
                    }
            
            return {"error": "Unknown tool"}
        
        self.mock_mcp_client.call_tool = mock_call_tool
        
        # Create orchestrator
        self.orchestrator = CoreOrchestrator(
            self.mock_gemini_client,
            self.mock_mcp_client
        )
    
    async def test_realistic_email_routing(self):
        """Test realistic email routing scenario."""
        requests = [
            "Check my unread emails and prioritize them",
            "Compose an email to the team about today's meeting",
            "Organize my inbox and delete spam messages"
        ]
        
        for request in requests:
            with self.subTest(request=request):
                result = await self.orchestrator.route_request(request)
                
                self.assertEqual(result['route'], 'email')
                self.assertGreater(result['confidence'], 0.7)
                self.assertIn('email', result['reason'].lower())
    
    async def test_realistic_calendar_routing(self):
        """Test realistic calendar routing scenario."""
        requests = [
            "Schedule a meeting with Sarah for next Wednesday",
            "What's on my calendar tomorrow?",
            "Reschedule the 3 PM meeting to 4 PM"
        ]
        
        for request in requests:
            with self.subTest(request=request):
                result = await self.orchestrator.route_request(request)
                
                self.assertEqual(result['route'], 'calendar')
                self.assertGreater(result['confidence'], 0.7)
    
    async def test_realistic_task_routing(self):
        """Test realistic task routing scenario."""
        requests = [
            "What are my highest priority tasks today?",
            "Mark the project report as completed",
            "Add 'Review budget proposal' to my task list"
        ]
        
        for request in requests:
            with self.subTest(request=request):
                result = await self.orchestrator.route_request(request)
                
                self.assertEqual(result['route'], 'task')
                self.assertGreater(result['confidence'], 0.7)
    
    async def test_realistic_complex_routing(self):
        """Test complex requests that should go to coordinator."""
        requests = [
            "Schedule a team meeting and send email invitations to everyone",
            "Check my calendar, prioritize urgent tasks, and send status update email",
            "Organize today's emails and update my task list based on action items"
        ]
        
        for request in requests:
            with self.subTest(request=request):
                result = await self.orchestrator.route_request(request)
                
                self.assertEqual(result['route'], 'coordinator')
                self.assertGreater(result['confidence'], 0.5)
    
    async def test_end_to_end_workflow(self):
        """Test complete orchestrator workflow from routing to verification."""
        user_input = "Help me organize my emails by priority"
        
        # 1. Route the request
        route_result = await self.orchestrator.route_request(user_input)
        self.assertEqual(route_result['route'], 'email')
        
        # 2. Simulate agent response
        agent_response = "I've organized your emails by priority. You have 5 high-priority emails that need immediate attention."
        
        # 3. Verify the response
        verification = await self.orchestrator.verify_response(
            user_input, agent_response, "EmailAgent"
        )
        
        self.assertGreater(verification['quality'], 0.7)
        self.assertGreater(verification['completeness'], 0.7)
        
        # 4. Check orchestrator stats
        stats = self.orchestrator.get_stats()
        self.assertEqual(stats['routing_stats']['total_requests'], 1)
        self.assertEqual(stats['routing_stats']['successful_routes'], 1)


async def run_orchestrator_tests():
    """Run all orchestrator tests."""
    print("🧪 Testing CoreOrchestrator...")
    
    # Run basic tests
    test_instance = TestCoreOrchestrator()
    test_instance.setUp()
    
    try:
        print("  📋 Testing initialization...")
        test_instance.test_orchestrator_initialization()
        print("  ✅ Initialization test passed")
        
        print("  🔍 Testing keyword analysis...")
        await test_instance.test_keyword_analysis()
        print("  ✅ Keyword analysis test passed")
        
        print("  🤖 Testing AI analysis...")
        await test_instance.test_ai_analysis_success()
        print("  ✅ AI analysis test passed")
        
        print("  🔀 Testing complex request detection...")
        await test_instance.test_complex_request_detection()
        print("  ✅ Complex request detection test passed")
        
        print("  🎯 Testing full routing workflow...")
        await test_instance.test_full_routing_workflow()
        print("  ✅ Full routing workflow test passed")
        
        print("  ✅ Testing response verification...")
        await test_instance.test_response_verification()
        print("  ✅ Response verification test passed")
        
        print("  🚨 Testing fallback handling...")
        await test_instance.test_fallback_handling()
        print("  ✅ Fallback handling test passed")
        
        print("  🏥 Testing health check...")
        await test_instance.test_health_check()
        print("  ✅ Health check test passed")
        
        print("  📊 Testing statistics...")
        test_instance.test_get_stats()
        print("  ✅ Statistics test passed")
        
        print("\n🔄 Running integration tests...")
        integration_test = TestOrchestratorIntegration()
        integration_test.setUp()
        
        await integration_test.test_realistic_email_routing()
        print("  ✅ Email routing integration test passed")
        
        await integration_test.test_realistic_calendar_routing()
        print("  ✅ Calendar routing integration test passed")
        
        await integration_test.test_realistic_task_routing()
        print("  ✅ Task routing integration test passed")
        
        await integration_test.test_end_to_end_workflow()
        print("  ✅ End-to-end workflow test passed")
        
        print("\n🎉 All CoreOrchestrator tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting CoreOrchestrator tests...\n")
    
    # Run regular unittest tests
    print("📝 Running unit tests:")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoreOrchestrator)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Run async tests
    print("\n" + "="*60)
    print("⚡ Running async tests...")
    success = asyncio.run(run_orchestrator_tests())
    
    print("\n" + "="*60)
    print("🏁 CoreOrchestrator testing complete!")
    
    if result.wasSuccessful() and success:
        print("🎉 All tests passed successfully!")
    else:
        print(f"❌ Some tests failed")