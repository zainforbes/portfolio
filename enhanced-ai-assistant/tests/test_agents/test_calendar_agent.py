# tests/test_calendar_agent.py
import asyncio
import unittest
from unittest.mock import Mock, AsyncMock
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.calendar_agent import CalendarAgent, EventSummary, CalendarAnalysis
from src.core.state_schema import AssistantState, make_initial_state

try:
    from src.mcp_integration.calendar_server import CalendarServer
except ImportError:
    # Mock CalendarServer if not available
    class CalendarServer:
        def __init__(self):
            self.authenticated = False
        
        async def authenticate(self):
            return {'success': True}
        
        async def list_events(self, days_ahead=7):
            # Mock calendar events
            now = datetime.now()
            events = [
                {
                    'id': f'event_{i}',
                    'title': f'Test Event {i}',
                    'start_time': (now + timedelta(days=i, hours=i+9)).isoformat(),
                    'end_time': (now + timedelta(days=i, hours=i+10)).isoformat(),
                    'description': f'Description for event {i}',
                    'location': f'Room {i}'
                }
                for i in range(min(5, days_ahead))
            ]
            return {'events': events, 'total': len(events)}
        
        async def create_event(self, title, start_time, end_time, description=""):
            return {
                'event_id': 'new_event_123',
                'status': 'created',
                'html_link': 'https://calendar.google.com/event/123'
            }
        
        async def call_tool(self, tool_name, parameters):
            if tool_name == 'list_events':
                return await self.list_events(parameters.get('days_ahead', 7))
            elif tool_name == 'create_event':
                return await self.create_event(
                    parameters.get('title'),
                    parameters.get('start_time'),
                    parameters.get('end_time'),
                    parameters.get('description', '')
                )
            return {}


class TestCalendarAgent(unittest.TestCase):
    """Test suite for CalendarAgent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calendar_server = CalendarServer()
        
        # Mock MCP client
        self.mock_mcp_client = Mock()
        self.mock_mcp_client.call_tool = AsyncMock()
        
        # Create calendar agent
        self.calendar_agent = CalendarAgent(self.mock_mcp_client)
        
        # Sample state
        self.sample_state = make_initial_state("Help me manage my calendar")
    
    def test_agent_initialization(self):
        """Test calendar agent initialization."""
        self.assertEqual(self.calendar_agent.agent_name, "CalendarAgent")
        self.assertIn("event_scheduling", self.calendar_agent.capabilities)
        self.assertIn("availability_checking", self.calendar_agent.capabilities)
        self.assertIn("calendar_analysis", self.calendar_agent.capabilities)
        self.assertIn("meeting_optimization", self.calendar_agent.capabilities)
    
    def test_can_handle_calendar_requests(self):
        """Test calendar request detection (sync version)."""
        test_cases = [
            ("Schedule a meeting for tomorrow", True),
            ("Check my calendar", True),
            ("Book an appointment", True),
            ("What's my availability?", True),
            ("Create an event", True),
            ("What's the weather like?", False),
            ("Send an email", False),
            ("Cancel my 3pm meeting", True),
            ("Reschedule the client call", True)
        ]
        
        for request, expected in test_cases:
            with self.subTest(request=request):
                # Run async test in sync context
                result = asyncio.run(self.calendar_agent.can_handle(request))
                self.assertEqual(result, expected)
    
    async def test_event_creation(self):
        """Test calendar event creation functionality."""
        # Mock event creation input
        test_state = make_initial_state("Schedule a meeting with John tomorrow at 2 PM")
        
        # Mock MCP responses
        self.mock_mcp_client.call_tool.side_effect = [
            # calendar_create_event response
            {
                'event_id': 'new_event_123',
                'status': 'created',
                'html_link': 'https://calendar.google.com/event/123'
            }
        ]
        
        # Mock AI response for event parsing
        original_generate = self.calendar_agent.generate_response
        self.calendar_agent.generate_response = AsyncMock(return_value='''
        {
            "title": "Meeting with John",
            "start_time": "2024-01-15T14:00:00",
            "end_time": "2024-01-15T15:00:00",
            "description": "Meeting scheduled for tomorrow at 2 PM",
            "location": ""
        }
        ''')
        
        # Test event creation
        result_state = await self.calendar_agent._create_calendar_event(test_state)
        
        # Verify event creation results
        self.assertIn('created_event', result_state['active_context'])
        created_event = result_state['active_context']['created_event']
        self.assertEqual(created_event['event_id'], 'new_event_123')
        self.assertEqual(created_event['status'], 'created')
        
        # Verify response message
        self.assertIn('final_response', result_state)
        self.assertIn('Event Created Successfully', result_state['final_response'])
        
        # Restore original method
        self.calendar_agent.generate_response = original_generate
    
    async def test_event_listing(self):
        """Test calendar event listing functionality."""
        # Mock events data
        now = datetime.now()
        mock_events = [
            {
                'id': 'event_1',
                'title': 'Team Standup',
                'start_time': (now + timedelta(hours=1)).isoformat(),
                'end_time': (now + timedelta(hours=1.5)).isoformat(),
                'description': 'Daily team standup meeting',
                'location': 'Conference Room A'
            },
            {
                'id': 'event_2',
                'title': 'Client Presentation',
                'start_time': (now + timedelta(hours=3)).isoformat(),
                'end_time': (now + timedelta(hours=4)).isoformat(),
                'description': 'Quarterly review presentation',
                'location': 'Boardroom'
            }
        ]
        
        # Mock MCP response
        self.mock_mcp_client.call_tool.side_effect = [
            {'events': mock_events, 'total': len(mock_events)}
        ]
        
        # Test event listing
        test_state = make_initial_state("Show me my upcoming events")
        result_state = await self.calendar_agent._list_calendar_events(test_state)
        
        # Verify event listing results
        self.assertIn('calendar_events', result_state['active_context'])
        events = result_state['active_context']['calendar_events']
        self.assertEqual(len(events), 2)
        
        # Check event details
        self.assertEqual(events[0]['title'], 'Team Standup')
        self.assertEqual(events[1]['title'], 'Client Presentation')
        
        # Verify response
        self.assertIn('Upcoming Events', result_state['final_response'])
        self.assertIn('Team Standup', result_state['final_response'])
    
    async def test_schedule_analysis(self):
        """Test schedule analysis functionality."""
        # Mock events for analysis
        now = datetime.now()
        mock_events = [
            {
                'id': 'event_1',
                'title': 'Morning Meeting',
                'start_time': (now + timedelta(hours=1)).isoformat(),
                'end_time': (now + timedelta(hours=2)).isoformat(),
                'description': 'Project planning',
                'location': 'Office'
            },
            {
                'id': 'event_2',
                'title': 'Overlapping Meeting',  # This will create a conflict
                'start_time': (now + timedelta(hours=1.5)).isoformat(),
                'end_time': (now + timedelta(hours=2.5)).isoformat(),
                'description': 'Conflicting meeting',
                'location': 'Online'
            }
        ]
        
        # Mock MCP response
        self.mock_mcp_client.call_tool.side_effect = [
            {'events': mock_events, 'total': len(mock_events)}
        ]
        
        # Test schedule analysis
        test_state = make_initial_state("Analyze my schedule for conflicts")
        result_state = await self.calendar_agent._analyze_schedule(test_state)
        
        # Verify analysis results
        self.assertIn('schedule_analysis', result_state['active_context'])
        analysis = result_state['active_context']['schedule_analysis']
        
        self.assertEqual(analysis['total_events'], 2)
        self.assertGreaterEqual(len(analysis['conflicts']), 1)  # Should detect the overlap
        self.assertIsInstance(analysis['recommendations'], list)
        
        # Verify response
        self.assertIn('Schedule Analysis', result_state['final_response'])
    
    async def test_availability_checking(self):
        """Test availability checking functionality."""
        # Mock current events
        now = datetime.now()
        mock_events = [
            {
                'id': 'busy_event',
                'title': 'Busy Period',
                'start_time': (now + timedelta(hours=2)).isoformat(),
                'end_time': (now + timedelta(hours=3)).isoformat(),
                'description': 'Scheduled meeting',
                'location': ''
            }
        ]
        
        # Mock MCP response
        self.mock_mcp_client.call_tool.side_effect = [
            {'events': mock_events, 'total': len(mock_events)}
        ]
        
        # Test availability check
        test_state = make_initial_state("When am I free tomorrow?")
        result_state = await self.calendar_agent._check_availability(test_state)
        
        # Verify availability results
        self.assertIn('availability_check', result_state['active_context'])
        availability = result_state['active_context']['availability_check']
        
        self.assertIn('available_slots', availability)
        self.assertIn('total_events', availability)
        self.assertEqual(availability['total_events'], 1)
        
        # Should find available slots outside the busy period
        available_slots = availability['available_slots']
        self.assertIsInstance(available_slots, list)
        
        # Verify response
        self.assertIn('Available Time Slots', result_state['final_response'])
    
    async def test_conflict_resolution(self):
        """Test scheduling conflict resolution."""
        # Mock conflicting events
        now = datetime.now()
        conflict_events = [
            {
                'id': 'event_1',
                'title': 'First Meeting',
                'start_time': (now + timedelta(hours=1)).isoformat(),
                'end_time': (now + timedelta(hours=2)).isoformat(),
                'description': 'Important client call',
                'location': ''
            },
            {
                'id': 'event_2', 
                'title': 'Second Meeting',
                'start_time': (now + timedelta(hours=1.5)).isoformat(),  # Overlaps with first
                'end_time': (now + timedelta(hours=2.5)).isoformat(),
                'description': 'Team meeting',
                'location': ''
            }
        ]
        
        # Mock MCP response
        self.mock_mcp_client.call_tool.side_effect = [
            {'events': conflict_events, 'total': len(conflict_events)}
        ]
        
        # Test conflict resolution
        test_state = make_initial_state("Find and resolve my scheduling conflicts")
        result_state = await self.calendar_agent._resolve_scheduling_conflicts(test_state)
        
        # Verify conflict resolution results
        self.assertIn('schedule_conflicts', result_state['active_context'])
        conflicts = result_state['active_context']['schedule_conflicts']
        
        self.assertEqual(conflicts['conflict_count'], 1)
        self.assertGreater(len(conflicts['conflicts']), 0)
        self.assertGreater(len(conflicts['resolutions']), 0)
        
        # Verify response mentions conflicts
        self.assertIn('Scheduling Conflicts Found', result_state['final_response'])
        self.assertIn('First Meeting', result_state['final_response'])
    
    async def test_schedule_optimization(self):
        """Test schedule optimization functionality."""
        # Mock events for optimization
        mock_events = [
            {'id': '1', 'title': 'Morning Meeting', 'start_time': '2024-01-15T09:00:00', 'end_time': '2024-01-15T10:00:00'},
            {'id': '2', 'title': 'Another Meeting', 'start_time': '2024-01-15T11:00:00', 'end_time': '2024-01-15T12:00:00'},
            {'id': '3', 'title': 'Afternoon Call', 'start_time': '2024-01-15T14:00:00', 'end_time': '2024-01-15T15:00:00'}
        ]
        
        # Mock MCP and AI responses
        self.mock_mcp_client.call_tool.side_effect = [
            {'events': mock_events, 'total': len(mock_events)}
        ]
        
        self.calendar_agent.generate_response = AsyncMock(return_value='''
        Based on your schedule analysis:
        
        1. **Batch Similar Meetings**: Group your morning meetings to create larger focus blocks
        2. **Buffer Time**: Add 15-minute buffers between meetings for transitions
        3. **Focus Blocks**: Reserve 2-hour blocks for deep work between 10 AM - 12 PM
        4. **Energy Management**: Schedule creative work during your peak hours
        
        Your current schedule has good spacing but could benefit from clustering meetings.
        ''')
        
        # Test schedule optimization
        test_state = make_initial_state("Optimize my schedule for better productivity")
        result_state = await self.calendar_agent._optimize_schedule(test_state)
        
        # Verify optimization results
        self.assertIn('schedule_optimization', result_state['active_context'])
        optimization = result_state['active_context']['schedule_optimization']
        
        self.assertEqual(len(optimization['current_events']), 3)
        self.assertIn('recommendations', optimization)
        self.assertIn('optimization_date', optimization)
        
        # Verify response contains optimization advice
        self.assertIn('Schedule Optimization Analysis', result_state['final_response'])
        self.assertIn('Batch Similar Meetings', result_state['final_response'])
    
    async def test_calendar_server_integration(self):
        """Test Calendar MCP server integration."""
        # Test server methods directly
        events_result = await self.calendar_server.list_events(days_ahead=3)
        self.assertIn('events', events_result)
        self.assertIn('total', events_result)
        self.assertLessEqual(len(events_result['events']), 3)
        
        # Test event creation
        create_result = await self.calendar_server.create_event(
            title="Test Event",
            start_time="2024-01-15T14:00:00",
            end_time="2024-01-15T15:00:00",
            description="Integration test event"
        )
        
        self.assertIn('event_id', create_result)
        self.assertIn('status', create_result)
        self.assertEqual(create_result['status'], 'created')
    
    async def test_time_parsing_utilities(self):
        """Test time parsing and utility functions."""
        # Test time range parsing
        self.assertEqual(self.calendar_agent._parse_time_range("show me today's events"), 1)
        self.assertEqual(self.calendar_agent._parse_time_range("what's on my calendar this week"), 7)
        self.assertEqual(self.calendar_agent._parse_time_range("next month's schedule"), 30)
        
        # Test duration calculation
        start_time = "2024-01-15T14:00:00"
        end_time = "2024-01-15T15:30:00"
        duration = self.calendar_agent._calculate_duration(start_time, end_time)
        self.assertEqual(duration, 90)  # 1.5 hours = 90 minutes
        
        # Test datetime formatting
        formatted = self.calendar_agent._format_datetime("2024-01-15T14:00:00")
        self.assertIn("Monday", formatted)  # Assuming this date is a Monday
        self.assertIn("2:00 PM", formatted)
    
    async def test_conflict_identification(self):
        """Test conflict identification logic."""
        # Create test events with known conflicts
        now = datetime.now()
        events = [
            {
                'title': 'Meeting A',
                'start_time': (now + timedelta(hours=1)).isoformat(),
                'end_time': (now + timedelta(hours=2)).isoformat()
            },
            {
                'title': 'Meeting B',
                'start_time': (now + timedelta(hours=1.5)).isoformat(),  # 30-minute overlap
                'end_time': (now + timedelta(hours=2.5)).isoformat()
            },
            {
                'title': 'Meeting C',
                'start_time': (now + timedelta(hours=3)).isoformat(),  # No conflict
                'end_time': (now + timedelta(hours=4)).isoformat()
            }
        ]
        
        conflicts = self.calendar_agent._identify_conflicts(events)
        
        # Should identify one conflict between Meeting A and Meeting B
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['event1'], 'Meeting A')
        self.assertEqual(conflicts[0]['event2'], 'Meeting B')
        self.assertEqual(conflicts[0]['overlap_minutes'], 30)
    
    async def test_available_slot_finding(self):
        """Test available slot finding algorithm."""
        # Mock events that block certain times
        now = datetime.now()
        events = [
            {
                'start_time': (now + timedelta(hours=1)).isoformat(),
                'end_time': (now + timedelta(hours=2)).isoformat()
            },
            {
                'start_time': (now + timedelta(hours=4)).isoformat(),
                'end_time': (now + timedelta(hours=5)).isoformat()
            }
        ]
        
        preferences = {'days_ahead': 1, 'duration_minutes': 60}
        available_slots = self.calendar_agent._find_available_slots(events, preferences)
        
        # Should find slots that don't conflict with existing events
        self.assertIsInstance(available_slots, list)
        
        # Each slot should have required fields
        if available_slots:
            slot = available_slots[0]
            self.assertIn('start', slot)
            self.assertIn('end', slot)
            self.assertIn('duration', slot)


class TestCalendarAgentIntegration(unittest.TestCase):
    """Integration tests for CalendarAgent with real MCP client simulation."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.calendar_server = CalendarServer()
        
        # Create a mock MCP client that calls our calendar server
        self.mcp_client = Mock()
        
        async def mock_call_tool(tool_name, parameters):
            # Map tool names to server methods
            if tool_name == "list_events":
                return await self.calendar_server.list_events(parameters.get('days_ahead', 7))
            elif tool_name == "create_event":
                return await self.calendar_server.create_event(
                    parameters.get('title'),
                    parameters.get('start_time'),  
                    parameters.get('end_time'),
                    parameters.get('description', '')
                )
            return {}
        
        self.mcp_client.call_tool = mock_call_tool
        
        # Create calendar agent with integrated MCP
        self.calendar_agent = CalendarAgent(self.mcp_client)
    
    async def test_full_calendar_workflow(self):
        """Test complete calendar management workflow."""
        # Test event creation workflow
        create_state = make_initial_state("Schedule a team meeting for tomorrow at 10 AM")
        
        # Mock the AI parsing response
        self.calendar_agent.generate_response = AsyncMock(return_value='''
        {
            "title": "Team Meeting",
            "start_time": "2024-01-16T10:00:00",
            "end_time": "2024-01-16T11:00:00",
            "description": "Weekly team meeting",
            "location": "Conference Room"
        }
        ''')
        
        result_state = await self.calendar_agent.execute_with_tracking(create_state)
        
        # Verify execution completed without errors
        self.assertNotIn('errors', result_state.get('active_context', {}))
        self.assertIn('final_response', result_state)
        self.assertTrue(len(result_state.get('agent_messages', [])) > 0)
        
        # Test event listing workflow
        list_state = make_initial_state("Show me my upcoming events")
        result_state = await self.calendar_agent.execute_with_tracking(list_state)
        
        # Should complete successfully
        self.assertIn('final_response', result_state)
        self.assertIn('current_agent', result_state)
        self.assertEqual(result_state['current_agent'], 'CalendarAgent')
    
    async def test_calendar_agent_error_handling(self):
        """Test calendar agent error handling."""
        # Create a state that will cause an error (invalid MCP call)
        error_state = make_initial_state("Invalid calendar request")
        
        # Mock MCP client to raise an exception
        async def failing_call_tool(tool_name, parameters):
            raise Exception("MCP connection failed")
        
        self.mcp_client.call_tool = failing_call_tool
        
        # Execute and verify error handling
        result_state = await self.calendar_agent.execute_with_tracking(error_state)
        
        # Should handle errors gracefully
        self.assertIn('error_log', result_state)
        self.assertTrue(len(result_state['error_log']) > 0)
        self.assertIn('CalendarAgent', result_state['error_log'][0]['agent'])


async def run_async_tests():
    """Run async test methods."""
    test_instance = TestCalendarAgent()
    test_instance.setUp()
    
    print("Running async CalendarAgent tests...")
    
    try:
        # Run async tests
        await test_instance.test_event_creation()
        print("[PASS] Calendar event creation test passed!")
        
        await test_instance.test_event_listing()
        print("[PASS] Calendar event listing test passed!")
        
        await test_instance.test_schedule_analysis()
        print("[PASS] Schedule analysis test passed!")
        
        await test_instance.test_availability_checking()
        print("[PASS] Availability checking test passed!")
        
        await test_instance.test_conflict_resolution()
        print("[PASS] Conflict resolution test passed!")
        
        await test_instance.test_schedule_optimization()
        print("[PASS] Schedule optimization test passed!")
        
        await test_instance.test_calendar_server_integration()
        print("[PASS] Calendar server integration test passed!")
        
        await test_instance.test_time_parsing_utilities()
        print("[PASS] Time parsing utilities test passed!")
        
        await test_instance.test_conflict_identification()
        print("[PASS] Conflict identification test passed!")
        
        await test_instance.test_available_slot_finding()
        print("[PASS] Available slot finding test passed!")
        
        print("\n[PASS] All CalendarAgent async tests passed!")
        
        # Run integration tests
        print("\nRunning integration tests...")
        integration_test = TestCalendarAgentIntegration()
        integration_test.setUp()
        
        await integration_test.test_full_calendar_workflow()
        print("[PASS] Full calendar workflow test passed!")
        
        await integration_test.test_calendar_agent_error_handling()
        print("[PASS] Calendar agent error handling test passed!")
        
        print("[PASS] CalendarAgent integration tests passed!")
        
    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Running CalendarAgent tests...")
    
    # Run regular unittest tests first
    print("\n" + "="*50)
    print("Running sync tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run async tests
    print("\n" + "="*50)
    print("Running async tests...")
    asyncio.run(run_async_tests())