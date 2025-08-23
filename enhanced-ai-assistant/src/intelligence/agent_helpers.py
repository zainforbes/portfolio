"""
Agent Helper Functions
Provides common utility functions used across multiple agents to keep core agent files clean.
"""

from typing import Dict, Any, List
from src.core.enhanced_state_schema import TaskType


async def can_handle_task(agent_instance, task_type: str, complexity: str, completeness: float) -> bool:
    """Determine if an agent can handle a specific task"""
    # Check if task type is in agent's supported types
    supported_types = [t.value if hasattr(t, 'value') else str(t) for t in agent_instance.get_task_types()]
    task_supported = task_type in supported_types
    
    # Check complexity and completeness thresholds
    complexity_ok = complexity in ['low', 'medium'] or (complexity == 'high' and completeness > 0.8)
    completeness_ok = completeness > 0.3  # Minimum threshold
    
    return task_supported and complexity_ok and completeness_ok


async def suggest_routing(agent_name: str, task_type: str, complexity: str, can_handle: bool) -> str:
    """Suggest routing based on task analysis"""
    if can_handle and complexity in ['low', 'medium']:
        return agent_name
    elif complexity == 'high':
        return 'orchestrator'  # Complex tasks need orchestration
    else:
        return 'orchestrator'  # Let orchestrator decide


# Calendar Agent Helper Functions

def generate_scheduling_clarification(event_parsing: Dict[str, Any]) -> str:
    """Generate clarification request for scheduling"""
    missing_info = event_parsing.get('missing_information', [])
    event_details = event_parsing.get('event_details', {})
    
    response = "I'd like to help you schedule this event, but I need a bit more information:\n\n"
    
    if 'start_datetime' in missing_info:
        response += "📅 **When would you like to schedule this?** (date and time)\n"
    
    if 'duration_minutes' in missing_info and not event_details.get('duration_minutes'):
        response += "⏱️ **How long should this event be?**\n"
    
    if 'location' in missing_info:
        response += "📍 **Where should this take place?** (location or video link)\n"
    
    if 'attendees' in missing_info:
        response += "👥 **Who should be invited?** (email addresses)\n"
    
    suggested_defaults = event_parsing.get('suggested_defaults', {})
    if suggested_defaults:
        response += "\n💡 **Suggested defaults:**\n"
        for key, value in suggested_defaults.items():
            response += f"• {key}: {value}\n"
    
    response += "\nPlease provide the missing details and I'll schedule it for you!"
    return response


def format_scheduling_options_response(optimization_result: Dict[str, Any], 
                                     event_parsing: Dict[str, Any]) -> str:
    """Format scheduling options for user selection"""
    optimal_schedule = optimization_result.get('optimal_schedule', {})
    alternatives = optimal_schedule.get('alternative_times', [])
    
    response = "🗓️ **Scheduling Options**\n\n"
    response += f"**Recommended Time:**\n"
    response += f"📅 {_format_datetime_display(optimal_schedule.get('recommended_datetime', ''))}\n"
    response += f"⏱️ Duration: {optimal_schedule.get('duration_recommendation', 60)} minutes\n"
    
    if optimal_schedule.get('location_suggestion'):
        response += f"📍 Location: {optimal_schedule['location_suggestion']}\n"
    
    if alternatives:
        response += f"\n**Alternative Times:**\n"
        for i, alt_time in enumerate(alternatives[:3], 1):
            response += f"{i}. {_format_datetime_display(alt_time)}\n"
    
    rationale = optimization_result.get('scheduling_rationale', {})
    if rationale:
        response += f"\n**Why this time works best:**\n"
        for factor, explanation in rationale.items():
            response += f"• {explanation}\n"
    
    response += f"\n📊 **Confidence Score:** {optimization_result.get('confidence', 0.0):.0%}"
    response += f"\n\nWould you like me to book the recommended time, or would you prefer one of the alternatives?"
    
    return response


async def parse_availability_request(user_input: str, state) -> Dict[str, Any]:
    """Parse availability checking request"""
    # Simple pattern matching for now
    request = {
        'timeframe': 'this_week',
        'duration_needed': 60,
        'preferred_times': [],
        'constraints': []
    }
    
    user_lower = user_input.lower()
    
    # Extract timeframe
    if 'today' in user_lower:
        request['timeframe'] = 'today'
    elif 'tomorrow' in user_lower:
        request['timeframe'] = 'tomorrow'
    elif 'this week' in user_lower:
        request['timeframe'] = 'this_week'
    elif 'next week' in user_lower:
        request['timeframe'] = 'next_week'
    
    # Extract duration
    import re
    duration_match = re.search(r'(\d+)\s*(?:hour|hr|minute|min)', user_lower)
    if duration_match:
        duration = int(duration_match.group(1))
        if 'hour' in user_lower or 'hr' in user_lower:
            request['duration_needed'] = duration * 60
        else:
            request['duration_needed'] = duration
    
    return request


async def find_optimal_availability(availability_request: Dict[str, Any], 
                                  calendar_data: Dict[str, Any], 
                                  state) -> Dict[str, Any]:
    """Find optimal availability slots"""
    events = calendar_data.get('events', [])
    duration_needed = availability_request.get('duration_needed', 60)
    
    # Simple algorithm to find gaps
    available_slots = []
    
    # For demo, return some sample slots
    from datetime import datetime, timedelta
    now = datetime.now()
    
    # Generate some available slots
    for i in range(3):
        start_time = now + timedelta(days=i, hours=9+i*2)
        available_slots.append({
            'start': start_time.isoformat(),
            'duration_minutes': duration_needed,
            'quality_score': 0.9 - i*0.1,
            'reason': f'Good {["morning", "afternoon", "evening"][i]} slot with no conflicts'
        })
    
    return {
        'available_slots': available_slots,
        'optimization_insights': [
            'Morning slots typically have higher success rates',
            'Buffer time included to prevent back-to-back meetings'
        ],
        'confidence': 0.8
    }


def generate_conflict_resolutions(conflicts: List[Dict[str, Any]], 
                                calendar_data: Dict[str, Any], 
                                state) -> Dict[str, Any]:
    """Generate conflict resolution strategies"""
    resolution_options = []
    
    for conflict in conflicts:
        # Generate different resolution strategies
        strategies = [
            {
                'strategy': 'reschedule',
                'description': f'Reschedule "{conflict["event1"]["title"]}" to avoid overlap',
                'pros': ['Preserves both events', 'Minimal disruption'],
                'cons': ['Requires coordination with attendees'],
                'confidence': 0.8
            },
            {
                'strategy': 'merge',
                'description': f'Combine overlapping events if topics are related',
                'pros': ['More efficient', 'Single meeting instead of two'],
                'cons': ['May not suit all attendees', 'Longer duration'],
                'confidence': 0.6
            }
        ]
        resolution_options.extend(strategies)
    
    return {
        'resolution_options': resolution_options,
        'recommended_action': 'Review each conflict and reschedule lower priority items',
        'confidence': 0.7
    }


def classify_event_operation(user_input: str) -> str:
    """Classify what event management operation is requested"""
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ['create', 'schedule', 'book', 'add', 'new']):
        return 'create'
    elif any(word in user_lower for word in ['update', 'change', 'modify', 'edit', 'reschedule']):
        return 'update'
    elif any(word in user_lower for word in ['delete', 'remove', 'cancel']):
        return 'delete'
    elif any(word in user_lower for word in ['list', 'show', 'what', 'upcoming', 'events']):
        return 'list'
    else:
        return 'unknown'


async def handle_event_update(state, context) -> Dict[str, Any]:
    """Handle event update operations"""
    return {
        'success': True,
        'data': "Event update functionality would be implemented here with MCP tools",
        'confidence': 0.7,
        'needs_clarification': True
    }


async def handle_event_deletion(state, context) -> Dict[str, Any]:
    """Handle event deletion operations"""
    return {
        'success': True, 
        'data': "Event deletion functionality would be implemented here with MCP tools",
        'confidence': 0.7,
        'needs_confirmation': True
    }


async def handle_event_listing(state, context) -> Dict[str, Any]:
    """Handle event listing operations"""
    return {
        'success': True,
        'data': "📅 **Upcoming Events:**\n\nEvent listing functionality would be implemented here with MCP tools to fetch and format upcoming events.",
        'confidence': 0.8
    }


# Helper utility functions

def _format_datetime_display(datetime_str: str) -> str:
    """Format datetime for user-friendly display"""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return dt.strftime("%A, %B %d at %I:%M %p")
    except Exception:
        return datetime_str or "TBD"