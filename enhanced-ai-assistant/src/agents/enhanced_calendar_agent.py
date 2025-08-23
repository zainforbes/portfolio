import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from src.agents.enhanced_base_agent import EnhancedBaseAgent
from src.core.enhanced_state_schema import (
    EnhancedAssistantState, TaskResult, AgentDecision, TaskType,
    ConfidenceLevel, TaskComplexity,
    update_resource_metrics, record_agent_decision, is_rate_limited,
    increment_rate_limit_counter
)
from src.core.enhanced_message_types import MessageType, MessageFactory
from src.intelligence.decision_logger import get_decision_logger
from src.intelligence.agent_helpers import (
    generate_scheduling_clarification, format_scheduling_options_response,
    parse_availability_request, find_optimal_availability, 
    generate_conflict_resolutions, classify_event_operation,
    handle_event_update, handle_event_deletion, handle_event_listing
)

class EnhancedCalendarAgent(EnhancedBaseAgent):
    """
    Enhanced Calendar Agent with full AI capabilities:
    - Intelligent calendar analysis and optimization
    - Advanced event parsing and scheduling
    - Conflict resolution and availability management
    - Autonomous scheduling decisions
    - Context-aware calendar insights
    - Resource optimization and caching
    - Error recovery and fallback strategies
    """
    
    def __init__(self, mcp_client, agent_name: str = "EnhancedCalendarAgent"):
        capabilities = [
            'calendar_scheduling', 'calendar_search', 'calendar_analysis',
            'availability_checking', 'conflict_resolution', 'schedule_optimization',
            'event_creation', 'time_management', 'meeting_coordination',
            'recurring_event_management'
        ]
        
        super().__init__(mcp_client, agent_name, capabilities)
        
        # Calendar-specific configuration
        self.calendar_cache: Dict[str, Any] = {}
        self.scheduling_patterns = self._initialize_scheduling_patterns()
        self.time_zones = ['UTC', 'EST', 'PST', 'GMT']
        self.working_hours = {'start': 9, 'end': 17}  # 9 AM to 5 PM
        
    def get_task_types(self) -> List[TaskType]:
        """Return calendar task types this agent can handle"""
        return [
            TaskType.CALENDAR_SCHEDULING,
            TaskType.CALENDAR_ANALYSIS,
            TaskType.AVAILABILITY_CHECKING,
            TaskType.CONFLICT_RESOLUTION,
            TaskType.EVENT_MANAGEMENT
        ]
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize calendar-specific prompt templates"""
        return {
            'calendar_analysis': """ADVANCED CALENDAR ANALYSIS

Analyze this calendar data for intelligent scheduling insights:

CALENDAR EVENTS:
{calendar_events}

CONVERSATION CONTEXT:
{conversation_context}

ANALYSIS REQUIREMENTS:
1. Schedule patterns and optimization opportunities
2. Time allocation analysis across activities
3. Availability windows and busy periods
4. Conflict detection and resolution strategies
5. Productivity insights and recommendations
6. Meeting efficiency analysis
7. Work-life balance assessment

RESPOND WITH STRUCTURED JSON:
{{
    "schedule_overview": {{
        "total_events": 0,
        "busy_hours_per_week": 0,
        "free_time_blocks": [],
        "recurring_patterns": []
    }},
    "conflict_analysis": {{
        "conflicts_detected": [],
        "potential_conflicts": [],
        "resolution_suggestions": []
    }},
    "optimization_recommendations": {{
        "meeting_batching": [],
        "focus_time_blocks": [],
        "schedule_adjustments": [],
        "efficiency_improvements": []
    }},
    "availability_insights": {{
        "best_meeting_times": [],
        "protected_focus_blocks": [],
        "flexible_time_slots": []
    }},
    "confidence": 0.0-1.0,
    "reasoning": "detailed analysis explanation"
}}

HALLUCINATION PREVENTION:
- Only analyze provided calendar data
- Mark inferred insights with confidence scores
- Distinguish between factual and suggested improvements""",

            'event_parsing': """INTELLIGENT EVENT PARSING

Extract and structure event information from this request:

USER REQUEST: "{user_input}"
CONVERSATION HISTORY: {conversation_history}

PARSING REQUIREMENTS:
1. Event title and description
2. Date and time information (handle relative dates like "tomorrow", "next week")
3. Duration or end time
4. Location (physical or virtual)
5. Attendees or participants
6. Recurring pattern if mentioned
7. Priority level and importance
8. Preparation or follow-up requirements

RESPOND WITH STRUCTURED JSON:
{{
    "event_details": {{
        "title": "extracted_title",
        "description": "event_description",
        "start_datetime": "ISO_format_or_null",
        "end_datetime": "ISO_format_or_null",
        "duration_minutes": 60,
        "location": "location_or_virtual_link",
        "attendees": ["list", "of", "attendees"],
        "recurring": {{
            "pattern": "daily|weekly|monthly|none",
            "frequency": 1,
            "end_date": "ISO_format_or_null"
        }},
        "priority": "high|medium|low",
        "category": "meeting|task|appointment|personal"
    }},
    "parsing_confidence": {{
        "title": 0.0-1.0,
        "datetime": 0.0-1.0,
        "location": 0.0-1.0,
        "overall": 0.0-1.0
    }},
    "missing_information": ["list", "of", "missing", "fields"],
    "clarification_needed": true/false,
    "suggested_defaults": {{}},
    "reasoning": "parsing explanation"
}}""",

            'scheduling_optimization': """INTELLIGENT SCHEDULING OPTIMIZATION

Optimize scheduling for this request considering constraints:

SCHEDULING REQUEST: {scheduling_request}
CURRENT CALENDAR: {current_events}
CONSTRAINTS: {scheduling_constraints}

OPTIMIZATION CRITERIA:
1. Minimize conflicts and overlaps
2. Respect working hours and preferences
3. Allow buffer time between meetings
4. Consider travel time if applicable
5. Maximize focus time blocks
6. Balance workload across days
7. Consider attendee availability

RESPOND WITH STRUCTURED JSON:
{{
    "optimal_schedule": {{
        "recommended_datetime": "ISO_format",
        "alternative_times": ["list", "of", "ISO_times"],
        "duration_recommendation": 60,
        "location_suggestion": "optimal_location"
    }},
    "scheduling_rationale": {{
        "conflict_avoidance": "explanation",
        "efficiency_factors": "explanation",
        "attendee_considerations": "explanation"
    }},
    "calendar_impact": {{
        "conflicts_resolved": 0,
        "focus_time_preserved": "hours",
        "travel_time_minimized": true/false
    }},
    "confidence": 0.0-1.0,
    "scheduling_score": 0.0-1.0
}}""",

            'conflict_resolution': """ADVANCED CONFLICT RESOLUTION

Resolve scheduling conflicts with intelligent solutions:

CONFLICT DETAILS: {conflict_details}
CALENDAR CONTEXT: {calendar_context}
PRIORITY LEVELS: {event_priorities}

RESOLUTION STRATEGIES:
1. Priority-based rescheduling
2. Duration optimization
3. Location consolidation
4. Meeting merger opportunities
5. Delegation possibilities
6. Alternative time slots

RESPOND WITH STRUCTURED JSON:
{{
    "conflict_analysis": {{
        "severity": "high|medium|low",
        "impact_assessment": "description",
        "affected_attendees": ["list"]
    }},
    "resolution_options": [
        {{
            "strategy": "reschedule|merge|delegate|cancel",
            "affected_events": ["event_ids"],
            "new_scheduling": {{}},
            "pros": ["list", "of", "benefits"],
            "cons": ["list", "of", "drawbacks"],
            "confidence": 0.0-1.0
        }}
    ],
    "recommended_action": "detailed_recommendation",
    "automation_possible": true/false,
    "requires_confirmation": ["list", "of", "confirmations"],
    "confidence": 0.0-1.0
}}"""
        }
    
    def _initialize_scheduling_patterns(self) -> Dict[str, Any]:
        """Initialize scheduling pattern recognition"""
        return {
            'time_expressions': [
                r'(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)',
                r'(\d{1,2}):(\d{2})',
                r'(?:at\s+)?(\d{1,2})\s*(am|pm|AM|PM)',
                r'(?:from\s+)(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)\s*(?:to|until|-)\s*(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)'
            ],
            'date_expressions': [
                r'\b(today|tomorrow|yesterday)\b',
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(next|this)\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:,?\s*(\d{4}))?\b'
            ],
            'duration_expressions': [
                r'(\d+)\s*(?:hour|hr)s?',
                r'(\d+)\s*(?:minute|min)s?',
                r'(\d+)h\s*(?:(\d+)m)?',
                r'(?:for\s+)?(\d+)\s*(?:to\s+)?(\d+)\s*(?:hour|hr)s?'
            ],
            'recurring_patterns': [
                r'\b(daily|every\s+day)\b',
                r'\b(weekly|every\s+week)\b',
                r'\b(monthly|every\s+month)\b',
                r'\b(yearly|annually|every\s+year)\b',
                r'\bevery\s+(\d+)\s+(days?|weeks?|months?|years?)\b'
            ]
        }
    
    async def execute(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Main execution method using the full AI pipeline"""
        return await self.execute_with_full_pipeline(state)
    
    async def _execute_task(self, decision: AgentDecision, 
                          state: EnhancedAssistantState, 
                          context: Dict[str, Any]) -> TaskResult:
        """Execute calendar-specific tasks with proper state management"""
        task_type = decision.parameters.get('task_type', TaskType.CALENDAR_ANALYSIS.value)
        
        # Record the decision in state
        record_agent_decision(state, decision)
        
        try:
            if task_type == TaskType.CALENDAR_SCHEDULING.value:
                return await self._handle_calendar_scheduling(decision, state, context)
            elif task_type == TaskType.CALENDAR_ANALYSIS.value:
                return await self._handle_calendar_analysis(decision, state, context)
            elif task_type == TaskType.AVAILABILITY_CHECKING.value:
                return await self._handle_availability_checking(decision, state, context)
            elif task_type == TaskType.CONFLICT_RESOLUTION.value:
                return await self._handle_conflict_resolution(decision, state, context)
            elif task_type == TaskType.EVENT_MANAGEMENT.value:
                return await self._handle_event_management(decision, state, context)
            else:
                error_msg = f"Unknown calendar task type: {task_type}"
                # Log error in state
                error_log = state.get('error_log', [])
                error_log.append({
                    'agent': self.agent_name,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                state['error_log'] = error_log
                
                return TaskResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    task_type=task_type,
                    agent=self.agent_name,
                    error=error_msg
                )
        except Exception as e:
            # Handle exceptions with proper state management
            error_record = {
                'agent': self.agent_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'task_type': task_type,
                'timestamp': datetime.now().isoformat()
            }
            
            error_log = state.get('error_log', [])
            error_log.append(error_record)
            state['error_log'] = error_log
            state['fallback_triggered'] = True
            
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=task_type,
                agent=self.agent_name,
                error=str(e),
                needs_escalation=True
            )
    
    async def _handle_calendar_scheduling(self, decision: AgentDecision,
                                        state: EnhancedAssistantState,
                                        context: Dict[str, Any]) -> TaskResult:
        """Handle intelligent calendar scheduling with optimization"""
        try:
            # Check rate limits before making Google API calls
            if is_rate_limited(state, 'gemini') or is_rate_limited(state, 'google_api'):
                return await self._handle_rate_limited_scheduling(decision, state, context)
                
            user_input = state.get('user_input', '')
            
            # 1. Parse event details from conversation
            event_parsing = await self._parse_event_details(user_input, state)
            
            # Update resource metrics
            update_resource_metrics(state, api_calls=1, processing_time=0.5)
            
            if event_parsing['clarification_needed']:
                clarification = generate_scheduling_clarification(event_parsing)
                return TaskResult(
                    success=True,
                    data=clarification,
                    confidence=0.8,
                    task_type=TaskType.CALENDAR_SCHEDULING.value,
                    agent=self.agent_name
                )
            
            # 2. Get current calendar context
            calendar_data = await self._get_calendar_context(state)
            
            # 3. Optimize scheduling
            optimization_result = await self._optimize_event_scheduling(
                event_parsing, calendar_data, state
            )
            
            # 4. Create event if optimization is confident
            if optimization_result['confidence'] > self.confidence_threshold:
                event_result = await self._create_optimized_event(
                    optimization_result, event_parsing, state
                )
                
                if event_result['success']:
                    response = self._format_scheduling_success_response(
                        event_result, optimization_result
                    )
                    
                    return TaskResult(
                        success=True,
                        data=response,
                        confidence=optimization_result['confidence'],
                        task_type=TaskType.CALENDAR_SCHEDULING.value,
                        agent=self.agent_name
                    )
            
            # 5. Present scheduling options for user decision
            options_response = format_scheduling_options_response(
                optimization_result, event_parsing
            )
            
            return TaskResult(
                success=True,
                data=options_response,
                confidence=optimization_result['confidence'],
                task_type=TaskType.CALENDAR_SCHEDULING.value,
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.CALENDAR_SCHEDULING.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    async def _handle_calendar_analysis(self, decision: AgentDecision,
                                      state: EnhancedAssistantState,
                                      context: Dict[str, Any]) -> TaskResult:
        """Handle comprehensive calendar analysis"""
        try:
            # Get calendar data with context
            calendar_data = await self._get_calendar_context(state)
            
            # Perform intelligent analysis
            analysis_result = await self._perform_comprehensive_calendar_analysis(
                calendar_data, state, context
            )
            
            # Generate insights response
            response = self._format_calendar_analysis_response(analysis_result)
            
            return TaskResult(
                success=True,
                data=response,
                confidence=analysis_result['confidence'],
                task_type=TaskType.CALENDAR_ANALYSIS.value,
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.CALENDAR_ANALYSIS.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    async def _handle_availability_checking(self, decision: AgentDecision,
                                          state: EnhancedAssistantState,
                                          context: Dict[str, Any]) -> TaskResult:
        """Handle intelligent availability analysis"""
        try:
            user_input = state.get('user_input', '')
            
            # Parse availability request
            availability_request = await parse_availability_request(user_input, state)
            
            # Get calendar context
            calendar_data = await self._get_calendar_context(state)
            
            # Find optimal availability slots
            availability_result = await find_optimal_availability(
                availability_request, calendar_data, state
            )
            
            # Generate availability response
            response = self._format_availability_response(
                availability_result, availability_request
            )
            
            return TaskResult(
                success=True,
                data=response,
                confidence=availability_result['confidence'],
                task_type=TaskType.AVAILABILITY_CHECKING.value,
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.AVAILABILITY_CHECKING.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    async def _handle_conflict_resolution(self, decision: AgentDecision,
                                        state: EnhancedAssistantState,
                                        context: Dict[str, Any]) -> TaskResult:
        """Handle intelligent conflict resolution"""
        try:
            # Get calendar data and identify conflicts
            calendar_data = await self._get_calendar_context(state)
            
            # Detect conflicts
            conflicts = await self._detect_calendar_conflicts(calendar_data)
            
            if not conflicts:
                return TaskResult(
                    success=True,
                    data="✅ **No scheduling conflicts detected**\n\nYour calendar looks well-organized with no overlapping events.",
                    confidence=0.95,
                    task_type=TaskType.CONFLICT_RESOLUTION.value,
                    agent=self.agent_name
                )
            
            # Generate resolution strategies
            resolution_result = await generate_conflict_resolutions(
                conflicts, calendar_data, state
            )
            
            # Format resolution response
            response = self._format_conflict_resolution_response(
                resolution_result, conflicts
            )
            
            return TaskResult(
                success=True,
                data=response,
                confidence=resolution_result['confidence'],
                task_type=TaskType.CONFLICT_RESOLUTION.value,
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.CONFLICT_RESOLUTION.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    async def _handle_event_management(self, decision: AgentDecision,
                                     state: EnhancedAssistantState,
                                     context: Dict[str, Any]) -> TaskResult:
        """Handle event management tasks"""
        try:
            user_input = state.get('user_input', '')
            
            # Determine event management operation
            operation = classify_event_operation(user_input)
            
            if operation == 'create':
                return await self._handle_calendar_scheduling(decision, state, context)
            elif operation == 'update':
                result = await handle_event_update(state, context)
                return self._convert_helper_result_to_task_result(result, TaskType.EVENT_MANAGEMENT.value)
            elif operation == 'delete':
                result = await handle_event_deletion(state, context)
                return self._convert_helper_result_to_task_result(result, TaskType.EVENT_MANAGEMENT.value)
            elif operation == 'list':
                result = await handle_event_listing(state, context)
                return self._convert_helper_result_to_task_result(result, TaskType.EVENT_MANAGEMENT.value)
            else:
                return TaskResult(
                    success=True,
                    data="I can help you with event management. What would you like to do?\n• Create new events\n• Update existing events\n• Delete events\n• List upcoming events",
                    confidence=0.8,
                    task_type=TaskType.EVENT_MANAGEMENT.value,
                    agent=self.agent_name
                )
                
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.EVENT_MANAGEMENT.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    # === CORE CALENDAR OPERATIONS ===
    
    async def _parse_event_details(self, user_input: str, 
                                 state: EnhancedAssistantState) -> Dict[str, Any]:
        """Parse event details using advanced LLM analysis"""
        conversation_history = state.get('conversation_history', [])
        history_text = self._format_conversation_for_analysis(conversation_history)
        
        prompt = self.prompt_templates['event_parsing'].format(
            user_input=user_input,
            conversation_history=history_text
        )
        
        try:
            response = await self.generate_response(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Event parsing failed: {e}")
            return self._get_fallback_event_parsing(user_input)
    
    async def _get_calendar_context(self, state: EnhancedAssistantState, 
                                  days_ahead: int = 14) -> Dict[str, Any]:
        """Get comprehensive calendar context with caching"""
        cache_key = f"calendar_context_{days_ahead}"
        
        # Check cache first
        if cache_key in self.calendar_cache:
            cache_entry = self.calendar_cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < 300:  # 5 min cache
                return cache_entry['data']
        
        try:
            # Get events using MCP tools
            events_result = await self.use_tool("list_calendar_events", {
                "days_ahead": days_ahead
            })
            
            calendar_data = {
                'events': events_result.get('events', []),
                'total_events': len(events_result.get('events', [])),
                'timeframe': f"next_{days_ahead}_days",
                'retrieved_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self.calendar_cache[cache_key] = {
                'data': calendar_data,
                'timestamp': datetime.now()
            }
            
            return calendar_data
            
        except Exception as e:
            self.logger.error(f"Calendar context retrieval failed: {e}")
            return {'events': [], 'total_events': 0, 'error': str(e)}
    
    async def _optimize_event_scheduling(self, event_parsing: Dict[str, Any],
                                       calendar_data: Dict[str, Any],
                                       state: EnhancedAssistantState) -> Dict[str, Any]:
        """Optimize event scheduling using advanced algorithms"""
        prompt = self.prompt_templates['scheduling_optimization'].format(
            scheduling_request=json.dumps(event_parsing, indent=2),
            current_events=json.dumps(calendar_data.get('events', [])[:10], indent=2),
            scheduling_constraints=json.dumps({
                'working_hours': self.working_hours,
                'buffer_time_minutes': 15,
                'max_daily_meetings': 6
            }, indent=2)
        )
        
        try:
            response = await self.generate_response(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Scheduling optimization failed: {e}")
            return self._get_fallback_scheduling_optimization(event_parsing)
    
    async def _perform_comprehensive_calendar_analysis(self, calendar_data: Dict[str, Any],
                                                     state: EnhancedAssistantState,
                                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive calendar analysis"""
        conversation_context = context.get('summary', '')
        
        prompt = self.prompt_templates['calendar_analysis'].format(
            calendar_events=json.dumps(calendar_data.get('events', [])[:20], indent=2),
            conversation_context=conversation_context
        )
        
        try:
            response = await self.generate_response(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Calendar analysis failed: {e}")
            return self._get_fallback_calendar_analysis(calendar_data)
    
    async def _detect_calendar_conflicts(self, calendar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect scheduling conflicts in calendar events"""
        events = calendar_data.get('events', [])
        conflicts = []
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda x: x.get('start_time', ''))
        
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            try:
                current_end = datetime.fromisoformat(
                    current.get('end_time', '').replace('Z', '+00:00')
                )
                next_start = datetime.fromisoformat(
                    next_event.get('start_time', '').replace('Z', '+00:00')
                )
                
                if current_end > next_start:
                    overlap_minutes = int((current_end - next_start).total_seconds() / 60)
                    conflicts.append({
                        'event1': {
                            'id': current.get('id', ''),
                            'title': current.get('title', 'Untitled'),
                            'start': current.get('start_time', ''),
                            'end': current.get('end_time', '')
                        },
                        'event2': {
                            'id': next_event.get('id', ''),
                            'title': next_event.get('title', 'Untitled'),
                            'start': next_event.get('start_time', ''),
                            'end': next_event.get('end_time', '')
                        },
                        'overlap_minutes': overlap_minutes,
                        'severity': 'high' if overlap_minutes > 30 else 'medium'
                    })
            except Exception as e:
                self.logger.warning(f"Conflict detection error: {e}")
                continue
        
        return conflicts
    
    # === RESPONSE FORMATTERS ===
    
    def _format_scheduling_success_response(self, event_result: Dict[str, Any],
                                          optimization_result: Dict[str, Any]) -> str:
        """Format successful scheduling response"""
        event_details = optimization_result.get('optimal_schedule', {})
        
        response = f"✅ **Event Scheduled Successfully!**\n\n"
        response += f"**{event_details.get('title', 'New Event')}**\n"
        response += f"📅 **Date & Time:** {self._format_datetime_display(event_details.get('recommended_datetime', ''))}\n"
        response += f"⏱️ **Duration:** {event_details.get('duration_recommendation', 60)} minutes\n"
        
        if event_details.get('location_suggestion'):
            response += f"📍 **Location:** {event_details['location_suggestion']}\n"
        
        response += f"\n**Scheduling Optimization:**\n"
        rationale = optimization_result.get('scheduling_rationale', {})
        response += f"• {rationale.get('conflict_avoidance', 'Optimal timing selected')}\n"
        response += f"• {rationale.get('efficiency_factors', 'Efficient scheduling achieved')}\n"
        
        impact = optimization_result.get('calendar_impact', {})
        if impact.get('focus_time_preserved'):
            response += f"• Focus time preserved: {impact['focus_time_preserved']}\n"
        
        response += f"\n📊 **Confidence Score:** {optimization_result.get('confidence', 0.0):.0%}"
        
        return response
    
    def _format_calendar_analysis_response(self, analysis_result: Dict[str, Any]) -> str:
        """Format comprehensive calendar analysis response"""
        overview = analysis_result.get('schedule_overview', {})
        conflicts = analysis_result.get('conflict_analysis', {})
        recommendations = analysis_result.get('optimization_recommendations', {})
        
        response = f"📊 **Calendar Analysis Report**\n\n"
        
        # Schedule Overview
        response += f"**📅 Schedule Overview:**\n"
        response += f"• Total Events: {overview.get('total_events', 0)}\n"
        response += f"• Weekly Busy Hours: {overview.get('busy_hours_per_week', 0)}\n"
        response += f"• Free Time Blocks: {len(overview.get('free_time_blocks', []))}\n\n"
        
        # Conflict Analysis
        conflicts_detected = conflicts.get('conflicts_detected', [])
        if conflicts_detected:
            response += f"⚠️ **Conflicts Detected:** {len(conflicts_detected)}\n"
            for conflict in conflicts_detected[:3]:  # Show top 3
                response += f"• {conflict}\n"
            response += "\n"
        
        # Optimization Recommendations
        response += f"🎯 **Optimization Recommendations:**\n"
        for category, suggestions in recommendations.items():
            if suggestions and isinstance(suggestions, list):
                response += f"• **{category.replace('_', ' ').title()}:**\n"
                for suggestion in suggestions[:2]:  # Show top 2 per category
                    response += f"  - {suggestion}\n"
        
        # Availability Insights
        availability = analysis_result.get('availability_insights', {})
        best_times = availability.get('best_meeting_times', [])
        if best_times:
            response += f"\n⏰ **Best Meeting Times:**\n"
            for time_slot in best_times[:3]:
                response += f"• {time_slot}\n"
        
        response += f"\n📊 **Analysis Confidence:** {analysis_result.get('confidence', 0.0):.0%}"
        
        return response
    
    def _format_availability_response(self, availability_result: Dict[str, Any],
                                    availability_request: Dict[str, Any]) -> str:
        """Format availability analysis response"""
        available_slots = availability_result.get('available_slots', [])
        
        if not available_slots:
            return "🔍 **No available slots found** for your requested timeframe. Consider extending the search period or adjusting your requirements."
        
        response = f"🗓️ **Available Time Slots**\n\n"
        response += f"Found {len(available_slots)} available slots:\n\n"
        
        for i, slot in enumerate(available_slots[:5], 1):
            start_time = self._format_datetime_display(slot.get('start', ''))
            duration = slot.get('duration_minutes', 60)
            response += f"**{i}.** {start_time} ({duration} min)\n"
            if slot.get('quality_score'):
                response += f"   Quality Score: {slot['quality_score']:.0%}\n"
        
        if len(available_slots) > 5:
            response += f"\n*...and {len(available_slots) - 5} more slots available*"
        
        # Add optimization insights
        if availability_result.get('optimization_insights'):
            response += f"\n\n💡 **Insights:**\n"
            for insight in availability_result['optimization_insights']:
                response += f"• {insight}\n"
        
        return response
    
    def _format_conflict_resolution_response(self, resolution_result: Dict[str, Any],
                                           conflicts: List[Dict[str, Any]]) -> str:
        """Format conflict resolution response"""
        response = f"⚠️ **{len(conflicts)} Scheduling Conflicts Found**\n\n"
        
        # Show conflicts
        for i, conflict in enumerate(conflicts, 1):
            event1 = conflict['event1']
            event2 = conflict['event2']
            response += f"**Conflict {i}:**\n"
            response += f"• {event1['title']} ↔️ {event2['title']}\n"
            response += f"• Overlap: {conflict['overlap_minutes']} minutes\n"
            response += f"• Severity: {conflict['severity'].upper()}\n\n"
        
        # Show resolution options
        options = resolution_result.get('resolution_options', [])
        if options:
            response += f"🔧 **Resolution Options:**\n\n"
            for i, option in enumerate(options[:3], 1):
                response += f"**Option {i}: {option.get('strategy', 'Unknown').title()}**\n"
                response += f"• Action: {option.get('description', 'No description')}\n"
                
                pros = option.get('pros', [])
                cons = option.get('cons', [])
                if pros:
                    response += f"• Pros: {', '.join(pros)}\n"
                if cons:
                    response += f"• Cons: {', '.join(cons)}\n"
                response += f"• Confidence: {option.get('confidence', 0.0):.0%}\n\n"
        
        # Show recommended action
        recommended = resolution_result.get('recommended_action')
        if recommended:
            response += f"💡 **Recommended Action:**\n{recommended}"
        
        return response
    
    # === UTILITY METHODS ===
    
    def _format_conversation_for_analysis(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for LLM analysis"""
        formatted = []
        for msg in conversation_history[-10:]:  # Last 10 messages
            if 'user' in msg:
                formatted.append(f"User: {msg['text']}")
            elif 'assistant' in msg:
                formatted.append(f"Assistant: {msg['text']}")
        return '\n'.join(formatted)
    
    def _format_datetime_display(self, datetime_str: str) -> str:
        """Format datetime for user-friendly display"""
        try:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime("%A, %B %d at %I:%M %p")
        except Exception:
            return datetime_str
    
    async def _create_optimized_event(self, optimization_result: Dict[str, Any],
                                    event_parsing: Dict[str, Any],
                                    state: EnhancedAssistantState) -> Dict[str, Any]:
        """Create calendar event using optimized scheduling"""
        optimal_schedule = optimization_result.get('optimal_schedule', {})
        event_details = event_parsing.get('event_details', {})
        
        try:
            event_result = await self.use_tool("create_calendar_event", {
                "title": event_details.get('title', 'New Event'),
                "start_time": optimal_schedule.get('recommended_datetime'),
                "end_time": self._calculate_end_time(
                    optimal_schedule.get('recommended_datetime'),
                    optimal_schedule.get('duration_recommendation', 60)
                ),
                "description": event_details.get('description', ''),
                "location": optimal_schedule.get('location_suggestion', '')
            })
            
            return {'success': True, 'result': event_result}
            
        except Exception as e:
            self.logger.error(f"Event creation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_end_time(self, start_time: str, duration_minutes: int) -> str:
        """Calculate end time given start time and duration"""
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = start_dt + timedelta(minutes=duration_minutes)
            return end_dt.isoformat()
        except Exception:
            return start_time
    
    # === FALLBACK METHODS ===
    
    def _get_fallback_event_parsing(self, user_input: str) -> Dict[str, Any]:
        """Fallback event parsing using pattern matching"""
        return {
            'event_details': {
                'title': user_input[:50] if user_input else 'New Event',
                'description': user_input,
                'duration_minutes': 60,
                'category': 'meeting'
            },
            'parsing_confidence': {'overall': 0.3},
            'clarification_needed': True,
            'missing_information': ['start_datetime', 'location'],
            'reasoning': 'Fallback pattern matching used'
        }
    
    def _get_fallback_scheduling_optimization(self, event_parsing: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback scheduling optimization"""
        # Suggest next business day at 2 PM
        tomorrow = datetime.now() + timedelta(days=1)
        next_business_day = tomorrow.replace(hour=14, minute=0, second=0, microsecond=0)
        
        return {
            'optimal_schedule': {
                'recommended_datetime': next_business_day.isoformat(),
                'duration_recommendation': 60,
                'location_suggestion': 'TBD'
            },
            'confidence': 0.5,
            'scheduling_score': 0.6
        }
    
    def _get_fallback_calendar_analysis(self, calendar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback calendar analysis"""
        total_events = calendar_data.get('total_events', 0)
        
        return {
            'schedule_overview': {
                'total_events': total_events,
                'busy_hours_per_week': total_events * 1.5,  # Rough estimate
                'free_time_blocks': [],
                'recurring_patterns': []
            },
            'optimization_recommendations': {
                'general': ['Consider time-blocking for focused work', 'Review recurring meetings for efficiency']
            },
            'confidence': 0.4,
            'reasoning': 'Fallback analysis based on event count'
        }
    
    def _convert_helper_result_to_task_result(self, helper_result: Dict[str, Any], task_type: str) -> TaskResult:
        """Convert helper function result to TaskResult"""
        return TaskResult(
            success=helper_result.get('success', True),
            data=helper_result.get('data', 'Operation completed'),
            confidence=helper_result.get('confidence', 0.7),
            task_type=task_type,
            agent=self.agent_name,
            error=helper_result.get('error'),
            needs_escalation=helper_result.get('needs_escalation', False)
        )

    def _get_task_patterns(self) -> Dict[str, List[str]]:
        """Get calendar-specific task patterns"""
        return {
            'calendar_scheduling': ['schedule', 'book', 'create event', 'plan meeting', 'set up'],
            'calendar_analysis': ['analyze', 'review schedule', 'calendar insights', 'time management'],
            'availability_checking': ['available', 'free time', 'when can', 'find time'],
            'conflict_resolution': ['conflicts', 'overlapping', 'double-booked', 'resolve'],
            'event_management': ['update event', 'cancel', 'reschedule', 'modify']
        }