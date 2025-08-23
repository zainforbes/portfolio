import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from src.agents.base_agent import BaseAgent
from src.core.state_schema import AssistantState
from src.core.message_types import AgentMessage, MessageTypes

@dataclass
class EventSummary:
    """Calendar event summary for quick overview"""
    id: str
    title: str
    start_time: str
    end_time: str
    description: str
    location: str
    duration_minutes: int
    is_recurring: bool
    attendees: List[str]

@dataclass
class CalendarAnalysis:
    """Calendar analysis result"""
    total_events: int
    busy_periods: List[Dict[str, str]]
    free_periods: List[Dict[str, str]]
    conflicts: List[Dict[str, Any]]
    recommendations: List[str]
    next_available_slot: Optional[Dict[str, str]]

class CalendarAgent(BaseAgent):
    """
    Specialized agent for calendar management and scheduling intelligence.
    Handles event creation, scheduling optimization, and availability management.
    """
    
    def __init__(self, gemini_mcp_client, agent_name: str = "CalendarAgent"):
        capabilities = [
            "event_scheduling",
            "availability_checking", 
            "calendar_analysis",
            "meeting_optimization",
            "time_management",
            "conflict_resolution",
            "event_creation",
            "schedule_planning"
        ]
        
        super().__init__(gemini_mcp_client, agent_name, capabilities)
        
        # Calendar-specific prompt templates
        self.system_prompts.update({
            'scheduling': """You are an expert calendar and scheduling assistant. Help users with:
1. Creating and managing calendar events
2. Finding optimal meeting times
3. Analyzing schedule conflicts
4. Optimizing time blocks for productivity
5. Managing availability and scheduling preferences

Consider factors like:
- Time zones and working hours
- Meeting duration and buffer time
- Travel time between locations
- Priority levels of different activities
- Work-life balance preferences""",

            'analysis': """You are an expert at calendar analysis. Provide insights on:
1. Schedule patterns and optimization opportunities
2. Time allocation across different activities
3. Meeting efficiency and scheduling habits
4. Availability patterns and busy periods
5. Recommendations for better time management

Focus on actionable insights that help improve productivity and work-life balance.""",

            'conflict_resolution': """You are an expert at resolving scheduling conflicts. When conflicts arise:
1. Identify the nature and severity of conflicts
2. Analyze priority levels of conflicting events
3. Suggest optimal solutions and alternatives
4. Consider rescheduling options and impact
5. Provide clear recommendations with reasoning"""
        })
        
        # Calendar patterns and utilities
        self.time_patterns = {
            'time_expressions': [
                r'(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)',
                r'(\d{1,2})\s*(am|pm|AM|PM)',
                r'at\s+(\d{1,2}):?(\d{2})?\s*(am|pm|AM|PM)?'
            ],
            'date_expressions': [
                r'(today|tomorrow|yesterday)',
                r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                r'(\d{1,2})/(\d{1,2})/(\d{2,4})',
                r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})',
                r'next\s+(week|month|monday|tuesday|wednesday|thursday|friday)'
            ],
            'duration_expressions': [
                r'(\d+)\s*hour?s?',
                r'(\d+)\s*minute?s?',
                r'(\d+)\s*min',
                r'(\d+)h\s*(\d+)m?'
            ]
        }

    async def execute(self, state: AssistantState) -> AssistantState:
        """Execute calendar agent operations based on current state and context."""
        try:
            user_request = state.get('user_input', '')
            task_type = state.get('task_type', '')
            
            # Handle follow-up response for calendar action selection
            if task_type == "calendar_action_request":
                return await self._handle_calendar_action_response(state)
            
            # Check if we need to clarify the user's intent first
            active_context = state.get('active_context', {})
            requested_action = active_context.get('requested_calendar_action')
            
            if requested_action is None:
                # Ask user what they want to do with their calendar
                return await self._clarify_calendar_intent(state)
            
            # Determine what calendar operation to perform based on stored action
            operation = requested_action
            
            if operation == "create_event":
                return await self._create_calendar_event(state)
            elif operation == "list_events":
                return await self._list_calendar_events(state)
            elif operation == "analyze_schedule":
                return await self._analyze_schedule(state)
            elif operation == "check_availability":
                return await self._check_availability(state)
            elif operation == "resolve_conflicts":
                return await self._resolve_scheduling_conflicts(state)
            elif operation == "optimize_schedule":
                return await self._optimize_schedule(state)
            else:
                return await self._general_calendar_assistance(state)
                
        except Exception as e:
            self.logger.error(f"Calendar agent execution failed: {e}")
            # Add to error log using state schema
            error_log = state.get('error_log', [])
            error_log.append({
                'agent': self.agent_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            state['error_log'] = error_log
            return state

    async def can_handle(self, request: str, context: Dict[str, Any] = None) -> bool:
        """Determine if this agent can handle the calendar-related request."""
        calendar_keywords = [
            'schedule', 'meeting', 'appointment', 'calendar', 'event',
            'book', 'reschedule', 'cancel', 'availability', 'time',
            'date', 'tomorrow', 'today', 'next week', 'busy', 'free',
            'conflict', 'organize time', 'plan', 'remind'
        ]
        
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in calendar_keywords)

    async def _determine_operation(self, request: str, context: Dict[str, Any]) -> str:
        """Determine what calendar operation to perform based on request."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['create', 'schedule', 'book', 'add event']):
            return "create_event"
        elif any(word in request_lower for word in ['list', 'show', 'upcoming', 'events']):
            return "list_events"
        elif any(word in request_lower for word in ['analyze', 'analysis', 'overview', 'summary']):
            return "analyze_schedule"
        elif any(word in request_lower for word in ['available', 'free', 'busy', 'availability']):
            return "check_availability"
        elif any(word in request_lower for word in ['conflict', 'overlapping', 'double-booked']):
            return "resolve_conflicts"
        elif any(word in request_lower for word in ['optimize', 'reorganize', 'improve']):
            return "optimize_schedule"
        else:
            return "general_assistance"

    def _add_agent_message(self, state: AssistantState, content: str, message_type: str = "info") -> AssistantState:
        """Helper to add agent messages to state using state schema."""
        # Add follow-up prompt for completed tasks (not for clarifications or input requests)
        if message_type in ["events_listed", "event_created", "schedule_analyzed", "availability_checked", "conflicts_analyzed", "schedule_optimized"] and not content.endswith("anything else"):
            follow_up = "\n\n📅 **Is there anything else I can help you with regarding your calendar?**\n• List different events\n• Create new events\n• Analyze your schedule\n• Check availability\n• Resolve conflicts\n• Optimize your schedule"
            content += follow_up
            
        agent_messages = state.get('agent_messages', [])
        agent_messages.append({
            'agent': self.agent_name,
            'content': content,
            'message_type': message_type,
            'timestamp': datetime.now().isoformat()
        })
        state['agent_messages'] = agent_messages
        
        # Also update final_response
        state['final_response'] = content
        
        return state

    async def _create_calendar_event(self, state: AssistantState) -> AssistantState:
        """Create a new calendar event based on user request."""
        try:
            user_input = state.get('user_input', '')
            
            # Parse event details from user input
            event_details = await self._parse_event_details(user_input)
            
            if not event_details.get('title'):
                return self._add_agent_message(
                    state, 
                    "I need more information to create the event. Please provide at least a title for the event.",
                    "info"
                )
            
            # Create the event using MCP calendar tools
            event_result = await self.use_tool(
                "create_calendar_event", 
                {
                    "title": event_details['title'],
                    "start_time": event_details.get('start_time', self._get_default_start_time()),
                    "end_time": event_details.get('end_time', self._get_default_end_time()),
                    "description": event_details.get('description', '')
                }
            )
            
            # Update state with event creation result
            active_context = state.get('active_context', {})
            active_context['created_event'] = {
                'event_id': event_result.get('event_id'),
                'title': event_details['title'],
                'status': event_result.get('status', 'created'),
                'html_link': event_result.get('html_link', '')
            }
            state['active_context'] = active_context
            
            # Generate response
            response = f"""=� **Event Created Successfully!**

**{event_details['title']}**
" Start: {event_details.get('start_time', 'TBD')}
" End: {event_details.get('end_time', 'TBD')}
" Description: {event_details.get('description', 'No description')}

Event ID: {event_result.get('event_id', 'N/A')}
{f"View event: {event_result.get('html_link', '')}" if event_result.get('html_link') else ''}
"""
            
            return self._add_agent_message(state, response, "event_created")
            
        except Exception as e:
            self.logger.error(f"Event creation failed: {e}")
            return self._add_agent_message(
                state, 
                f"I encountered an error creating the event: {str(e)}",
                "error"
            )

    async def _list_calendar_events(self, state: AssistantState) -> AssistantState:
        """List upcoming calendar events."""
        try:
            user_input = state.get('user_input', '')
            
            # Determine time range from user input
            days_ahead = self._parse_time_range(user_input)
            
            # Get events using MCP calendar tools
            events_result = await self.use_tool(
                "list_calendar_events",
                {"days_ahead": days_ahead}
            )
            
            events = events_result.get('events', [])
            
            # Create event summaries
            event_summaries = []
            for event in events:
                summary = EventSummary(
                    id=event.get('id', ''),
                    title=event.get('title', 'Untitled Event'),
                    start_time=event.get('start_time', ''),
                    end_time=event.get('end_time', ''),
                    description=event.get('description', ''),
                    location=event.get('location', ''),
                    duration_minutes=self._calculate_duration(
                        event.get('start_time', ''), 
                        event.get('end_time', '')
                    ),
                    is_recurring=False,  # Would need to parse this from event data
                    attendees=[]  # Would need to parse this from event data
                )
                event_summaries.append(summary)
            
            # Update state with event list
            active_context = state.get('active_context', {})
            active_context['calendar_events'] = [summary.__dict__ for summary in event_summaries]
            active_context['events_count'] = len(event_summaries)
            state['active_context'] = active_context
            
            # Generate response
            response = await self._generate_events_list_response(event_summaries, days_ahead)
            return self._add_agent_message(state, response, "events_listed")
            
        except Exception as e:
            self.logger.error(f"Event listing failed: {e}")
            return self._add_agent_message(
                state,
                f"I encountered an error retrieving calendar events: {str(e)}",
                "error"
            )

    async def _analyze_schedule(self, state: AssistantState) -> AssistantState:
        """Analyze the user's schedule and provide insights."""
        try:
            # Get recent events for analysis
            events_result = await self.use_tool(
                "list_calendar_events",
                {"days_ahead": 7}
            )
            
            events = events_result.get('events', [])
            
            # Perform schedule analysis
            analysis = await self._perform_schedule_analysis(events)
            
            # Update state with analysis
            active_context = state.get('active_context', {})
            active_context['schedule_analysis'] = analysis.__dict__
            state['active_context'] = active_context
            
            # Generate analysis response
            response = await self._generate_analysis_response(analysis)
            return self._add_agent_message(state, response, "schedule_analyzed")
            
        except Exception as e:
            self.logger.error(f"Schedule analysis failed: {e}")
            return self._add_agent_message(
                state,
                f"I encountered an error analyzing your schedule: {str(e)}",
                "error"
            )

    async def _check_availability(self, state: AssistantState) -> AssistantState:
        """Check availability for scheduling new events."""
        try:
            user_input = state.get('user_input', '')
            
            # Parse time preferences from user input
            time_preference = self._parse_availability_request(user_input)
            
            # Get current events for the requested period
            events_result = await self.use_tool(
                "list_calendar_events",
                {"days_ahead": time_preference.get('days_ahead', 7)}
            )
            
            events = events_result.get('events', [])
            
            # Find available time slots
            available_slots = self._find_available_slots(events, time_preference)
            
            # Update state
            active_context = state.get('active_context', {})
            active_context['availability_check'] = {
                'requested_period': time_preference,
                'available_slots': available_slots,
                'total_events': len(events)
            }
            state['active_context'] = active_context
            
            # Generate availability response
            response = self._generate_availability_response(available_slots, time_preference)
            return self._add_agent_message(state, response, "availability_checked")
            
        except Exception as e:
            self.logger.error(f"Availability check failed: {e}")
            return self._add_agent_message(
                state,
                f"I encountered an error checking availability: {str(e)}",
                "error"
            )

    async def _resolve_scheduling_conflicts(self, state: AssistantState) -> AssistantState:
        """Identify and provide solutions for scheduling conflicts."""
        try:
            # Get current events
            events_result = await self.use_tool(
                "list_calendar_events",
                {"days_ahead": 14}
            )
            
            events = events_result.get('events', [])
            
            # Identify conflicts
            conflicts = self._identify_conflicts(events)
            
            # Generate conflict resolution suggestions
            resolutions = await self._generate_conflict_resolutions(conflicts)
            
            # Update state
            active_context = state.get('active_context', {})
            active_context['schedule_conflicts'] = {
                'conflicts': conflicts,
                'resolutions': resolutions,
                'conflict_count': len(conflicts)
            }
            state['active_context'] = active_context
            
            # Generate response
            response = self._generate_conflict_resolution_response(conflicts, resolutions)
            return self._add_agent_message(state, response, "conflicts_analyzed")
            
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return self._add_agent_message(
                state,
                f"I encountered an error analyzing conflicts: {str(e)}",
                "error"
            )

    async def _optimize_schedule(self, state: AssistantState) -> AssistantState:
        """Provide schedule optimization recommendations."""
        try:
            # Get events for optimization analysis
            events_result = await self.use_tool(
                "list_calendar_events",
                {"days_ahead": 7}
            )
            
            events = events_result.get('events', [])
            
            # Generate optimization recommendations using AI
            optimization_prompt = f"""
            Analyze this calendar schedule and provide optimization recommendations:
            
            Events: {json.dumps(events, indent=2)}
            
            Consider:
            1. Meeting clustering and batching
            2. Focus time blocks
            3. Travel time between meetings
            4. Energy management throughout the day
            5. Work-life balance
            
            Provide specific, actionable recommendations.
            """
            
            recommendations = await self.generate_response(
                optimization_prompt,
                context=self.system_prompts['analysis']
            )
            
            # Update state
            active_context = state.get('active_context', {})
            active_context['schedule_optimization'] = {
                'current_events': events,
                'recommendations': recommendations,
                'optimization_date': datetime.now().isoformat()
            }
            state['active_context'] = active_context
            
            response = f"""=� **Schedule Optimization Analysis**

{recommendations}

*Analysis based on your upcoming {len(events)} events*
"""
            
            return self._add_agent_message(state, response, "schedule_optimized")
            
        except Exception as e:
            self.logger.error(f"Schedule optimization failed: {e}")
            return self._add_agent_message(
                state,
                f"I encountered an error optimizing your schedule: {str(e)}",
                "error"
            )

    async def _general_calendar_assistance(self, state: AssistantState) -> AssistantState:
        """Provide general calendar assistance."""
        user_input = state.get('user_input', 'How can I help with your calendar?')
        response = await self.generate_response(
            user_input,
            context="You are a helpful calendar assistant. Provide guidance on calendar management, scheduling, and time organization best practices."
        )
        return self._add_agent_message(state, response, "assistance")

    async def _clarify_calendar_intent(self, state: AssistantState) -> AssistantState:
        """Ask user what they want to do with their calendar."""
        clarification_prompt = (
            "What would you like me to do with your calendar?\n\n"
            "📅 **Available Options:**\n"
            "• 📋 **List Events** - View your upcoming events\n"
            "• ➕ **Create Event** - Schedule a new event\n"
            "• 📊 **Analyze Schedule** - Get insights on your schedule\n"
            "• 🔍 **Check Availability** - Find free time slots\n"
            "• ⚠️ **Resolve Conflicts** - Fix scheduling conflicts\n"
            "• 🎯 **Optimize Schedule** - Get optimization suggestions\n\n"
            "Please tell me which option you'd like, or describe what you need help with:"
        )
        
        state["task_type"] = "calendar_action_request"
        state["pending_requests"] = ["calendar_action"]
        state["current_agent"] = self.agent_name
        return self._add_agent_message(state, clarification_prompt, "clarification")

    async def _handle_calendar_action_response(self, state: AssistantState) -> AssistantState:
        """Handle user's response with their chosen calendar action."""
        user_input = state.get('user_input', '').strip().lower()
        
        # Map user responses to calendar operations
        action_mapping = {
            'list': 'list_events',
            'events': 'list_events', 
            'show': 'list_events',
            'view': 'list_events',
            'upcoming': 'list_events',
            '1': 'list_events',
            
            'create': 'create_event',
            'schedule': 'create_event',
            'add': 'create_event',
            'new': 'create_event',
            '2': 'create_event',
            
            'analyze': 'analyze_schedule',
            'analysis': 'analyze_schedule',
            'insights': 'analyze_schedule',
            'overview': 'analyze_schedule',
            '3': 'analyze_schedule',
            
            'availability': 'check_availability',
            'available': 'check_availability',
            'free': 'check_availability',
            'slots': 'check_availability',
            '4': 'check_availability',
            
            'conflicts': 'resolve_conflicts',
            'conflict': 'resolve_conflicts',
            'resolve': 'resolve_conflicts',
            'overlapping': 'resolve_conflicts',
            '5': 'resolve_conflicts',
            
            'optimize': 'optimize_schedule',
            'optimization': 'optimize_schedule',
            'improve': 'optimize_schedule',
            'suggestions': 'optimize_schedule',
            '6': 'optimize_schedule'
        }
        
        # Find matching action
        chosen_action = None
        for keyword, action in action_mapping.items():
            if keyword in user_input:
                chosen_action = action
                break
        
        if chosen_action:
            # Store the chosen action and proceed
            active_context = state.get('active_context', {})
            active_context['requested_calendar_action'] = chosen_action
            state['active_context'] = active_context
            
            # Clear the task type to proceed with normal execution
            state['task_type'] = 'calendar_execution'
            
            # Execute the chosen action
            if chosen_action == "create_event":
                return await self._create_calendar_event(state)
            elif chosen_action == "list_events":
                return await self._list_calendar_events(state)
            elif chosen_action == "analyze_schedule":
                return await self._analyze_schedule(state)
            elif chosen_action == "check_availability":
                return await self._check_availability(state)
            elif chosen_action == "resolve_conflicts":
                return await self._resolve_scheduling_conflicts(state)
            elif chosen_action == "optimize_schedule":
                return await self._optimize_schedule(state)
        else:
            # No valid action found, ask again
            prompt = (
                "I didn't understand your choice. Please select one of the options:\n\n"
                "1️⃣ List Events\n2️⃣ Create Event\n3️⃣ Analyze Schedule\n"
                "4️⃣ Check Availability\n5️⃣ Resolve Conflicts\n6️⃣ Optimize Schedule\n\n"
                "You can type the number or the option name:"
            )
            return self._add_agent_message(state, prompt, "input_request")

    # Utility Methods
    
    async def _parse_event_details(self, user_input: str) -> Dict[str, Any]:
        """Parse event details from user input using AI."""
        try:
            parsing_prompt = f"""
            Extract event details from this request: "{user_input}"
            
            Return JSON with:
            - title: event title/subject
            - start_time: ISO format datetime (or null)
            - end_time: ISO format datetime (or null)
            - description: any additional details
            - location: meeting location if mentioned
            """
            
            response = await self.generate_response(parsing_prompt)
            
            # Parse the AI response
            event_details = self._parse_ai_json_response(response)
            return event_details
            
        except Exception as e:
            self.logger.error(f"Event parsing failed: {e}")
            # Fallback to basic parsing
            return {
                'title': user_input[:50],  # Use first 50 chars as title
                'description': user_input
            }

    def _parse_ai_json_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response that should contain JSON."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        return {'title': 'New Event', 'description': ''}

    def _parse_time_range(self, user_input: str) -> int:
        """Parse time range from user input."""
        if 'today' in user_input.lower():
            return 1
        elif 'tomorrow' in user_input.lower():
            return 2
        elif 'week' in user_input.lower():
            return 7
        elif 'month' in user_input.lower():
            return 30
        else:
            return 7  # Default to 1 week

    def _calculate_duration(self, start_time: str, end_time: str) -> int:
        """Calculate duration in minutes between two ISO datetime strings."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return int((end - start).total_seconds() / 60)
        except:
            return 0

    def _get_default_start_time(self) -> str:
        """Get default start time (next hour)."""
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_hour.isoformat()

    def _get_default_end_time(self) -> str:
        """Get default end time (1 hour after start)."""
        start_time = self._get_default_start_time()
        start_dt = datetime.fromisoformat(start_time)
        end_dt = start_dt + timedelta(hours=1)
        return end_dt.isoformat()

    async def _perform_schedule_analysis(self, events: List[Dict[str, Any]]) -> CalendarAnalysis:
        """Perform comprehensive schedule analysis."""
        total_events = len(events)
        
        # Basic analysis - in a real implementation, this would be more sophisticated
        busy_periods = []
        free_periods = []
        conflicts = self._identify_conflicts(events)
        
        # Simple recommendations based on event count
        recommendations = []
        if total_events > 10:
            recommendations.append("Consider consolidating meetings to create focus blocks")
        if len(conflicts) > 0:
            recommendations.append("Resolve scheduling conflicts to avoid double-booking")
        
        # Find next available slot
        next_available = self._find_next_available_slot(events)
        
        return CalendarAnalysis(
            total_events=total_events,
            busy_periods=busy_periods,
            free_periods=free_periods,
            conflicts=conflicts,
            recommendations=recommendations,
            next_available_slot=next_available
        )

    def _identify_conflicts(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify scheduling conflicts in events."""
        conflicts = []
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda x: x.get('start_time', ''))
        
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            try:
                current_end = datetime.fromisoformat(current.get('end_time', '').replace('Z', '+00:00'))
                next_start = datetime.fromisoformat(next_event.get('start_time', '').replace('Z', '+00:00'))
                
                if current_end > next_start:
                    conflicts.append({
                        'event1': current.get('title', 'Untitled'),
                        'event2': next_event.get('title', 'Untitled'),
                        'overlap_minutes': int((current_end - next_start).total_seconds() / 60)
                    })
            except:
                continue
        
        return conflicts

    def _find_available_slots(self, events: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, str]]:
        """Find available time slots based on existing events."""
        # Simplified implementation - would be more sophisticated in practice
        available_slots = []
        
        # Example: find 1-hour slots between 9 AM and 5 PM
        for day_offset in range(preferences.get('days_ahead', 7)):
            date = datetime.now() + timedelta(days=day_offset)
            
            # Check each hour from 9 AM to 5 PM
            for hour in range(9, 17):
                slot_start = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(hours=1)
                
                # Check if slot conflicts with existing events
                is_available = True
                for event in events:
                    try:
                        event_start = datetime.fromisoformat(event.get('start_time', '').replace('Z', '+00:00'))
                        event_end = datetime.fromisoformat(event.get('end_time', '').replace('Z', '+00:00'))
                        
                        if (slot_start < event_end and slot_end > event_start):
                            is_available = False
                            break
                    except:
                        continue
                
                if is_available:
                    available_slots.append({
                        'start': slot_start.isoformat(),
                        'end': slot_end.isoformat(),
                        'duration': 60
                    })
        
        return available_slots[:10]  # Return first 10 slots

    def _find_next_available_slot(self, events: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Find the next available 1-hour time slot."""
        available_slots = self._find_available_slots(events, {'days_ahead': 7})
        return available_slots[0] if available_slots else None

    def _parse_availability_request(self, user_input: str) -> Dict[str, Any]:
        """Parse availability request parameters."""
        return {
            'days_ahead': self._parse_time_range(user_input),
            'duration_minutes': 60,  # Default 1 hour
            'working_hours_only': True
        }

    async def _generate_conflict_resolutions(self, conflicts: List[Dict[str, Any]]) -> List[str]:
        """Generate conflict resolution suggestions."""
        resolutions = []
        for conflict in conflicts:
            resolution = f"Reschedule '{conflict['event1']}' or '{conflict['event2']}' - {conflict['overlap_minutes']} minute overlap"
            resolutions.append(resolution)
        return resolutions

    # Response Generators
    
    async def _generate_events_list_response(self, events: List[EventSummary], days_ahead: int) -> str:
        """Generate formatted response for events list."""
        if not events:
            return f"=� No events found for the next {days_ahead} day{'s' if days_ahead != 1 else ''}."
        
        response = f"=� **Upcoming Events ({len(events)} events)**\n\n"
        
        for event in events[:10]:  # Show max 10 events
            start_time = self._format_datetime(event.start_time)
            duration_text = f" ({event.duration_minutes} min)" if event.duration_minutes > 0 else ""
            
            response += f"**{event.title}**\n"
            response += f"• {start_time}{duration_text}\n"
            if event.location:
                response += f"• Location: {event.location}\n"
            if event.description:
                response += f"• {event.description[:100]}{'...' if len(event.description) > 100 else ''}\n"
            response += "\n"
        
        if len(events) > 10:
            response += f"*...and {len(events) - 10} more events*"
        
        return response

    async def _generate_analysis_response(self, analysis: CalendarAnalysis) -> str:
        """Generate formatted schedule analysis response."""
        response = f"""=� **Schedule Analysis**

**Overview:**
" Total Events: {analysis.total_events}
" Scheduling Conflicts: {len(analysis.conflicts)}

**Recommendations:**
"""
        for rec in analysis.recommendations:
            response += f"• {rec}\n"
        
        if analysis.next_available_slot:
            next_time = self._format_datetime(analysis.next_available_slot.get('start', ''))
            response += f"\n**Next Available Slot:** {next_time}"
        
        return response

    def _generate_availability_response(self, available_slots: List[Dict[str, str]], preferences: Dict[str, Any]) -> str:
        """Generate availability check response."""
        if not available_slots:
            return f"� No available slots found for the next {preferences.get('days_ahead', 7)} days."
        
        response = f"� **Available Time Slots** (showing {len(available_slots)} options)\n\n"
        
        for slot in available_slots[:5]:  # Show top 5 slots
            start_time = self._format_datetime(slot['start'])
            response += f"• {start_time} ({slot.get('duration', 60)} minutes)\n"
        
        if len(available_slots) > 5:
            response += f"\n*...and {len(available_slots) - 5} more slots available*"
        
        return response

    def _generate_conflict_resolution_response(self, conflicts: List[Dict[str, Any]], resolutions: List[str]) -> str:
        """Generate conflict resolution response."""
        if not conflicts:
            return " **No Scheduling Conflicts Found**\n\nYour calendar looks good with no overlapping events."
        
        response = f"� **Scheduling Conflicts Found** ({len(conflicts)} conflicts)\n\n"
        
        for i, conflict in enumerate(conflicts):
            response += f"**Conflict {i+1}:**\n"
            response += f"• {conflict['event1']} overlaps with {conflict['event2']}\n"
            response += f"• Overlap: {conflict['overlap_minutes']} minutes\n\n"
        
        response += "**Suggested Resolutions:**\n"
        for resolution in resolutions:
            response += f"• {resolution}\n"
        
        return response

    def _format_datetime(self, dt_string: str) -> str:
        """Format datetime string for display."""
        try:
            dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
            return dt.strftime("%A, %B %d at %I:%M %p")
        except:
            return dt_string