import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.core.state_schema import AssistantState

class CoreOrchestrator:
    """
    Central orchestrator for the AI Assistant system.
    Handles request routing, response verification, fallback logic,
    and complex multi-agent orchestration.
    """

    def __init__(self, gemini_mcp_client):
        self.gemini_mcp_client = gemini_mcp_client

        self.logger = logging.getLogger("orchestrator")
        self.logger.setLevel(logging.INFO)

        self.route_patterns = {
            'email': [
                'email', 'mail', 'inbox', 'compose', 'send', 'reply',
                'forward', 'attachment', 'subject', 'sender', 'recipient',
                'gmail', 'outlook', 'message', 'correspondence'
            ],
            'calendar': [
                'schedule', 'meeting', 'appointment', 'calendar', 'event',
                'book', 'reschedule', 'cancel', 'availability', 'time',
                'date', 'tomorrow', 'today', 'next week', 'free', 'busy'
            ],
            'search': [
                'search', 'find', 'look up', 'research', 'investigate',
                'what is', 'who is', 'how to', 'when did', 'where is',
                'what are', 'what was', 'capital of', 'capital city',
                'google', 'browse', 'web', 'internet', 'online',
                'information about', 'details on', 'facts about',
                'brave search', 'web search', 'define', 'explain'
            ]
        }

        self.routing_stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'fallback_count': 0,
            'average_confidence': 0.0
        }

    async def _analyze_conversation_context(self, user_input: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to intelligently analyze conversation context and determine next action."""
        try:
            # Format conversation history for LLM analysis
            conversation_text = []
            for msg in conversation_history[-10:]:  # Last 10 messages for context
                if 'user' in msg:
                    conversation_text.append(f"User: {msg['text']}")
                elif 'assistant' in msg:
                    conversation_text.append(f"Assistant: {msg['text']}")
            
            conversation_str = "\n".join(conversation_text)
            
            prompt = f"""Analyze this conversation to understand what the user wants to do next:

CONVERSATION HISTORY:
{conversation_str}

CURRENT USER INPUT: "{user_input}"

Based on the conversation context, determine:
1. What is the user trying to accomplish? 
2. What should happen next?
3. Which agent/capability should handle this: email, calendar, search, or orchestrator?
4. What specific action should be taken?

Respond in this exact JSON format:
{{
    "analysis": "Brief explanation of what's happening in the conversation",
    "user_intent": "What the user wants to accomplish", 
    "next_action": "Specific action to take",
    "route": "email|calendar|search|orchestrator",
    "task_type": "specific task identifier",
    "confidence": 0.0-1.0,
    "parameters": {{}}
}}

INTELLIGENT ROUTING: Make autonomous decisions about:
1. Which agent should handle this request
2. What specific task needs to be done
3. What parameters are needed
4. Whether agents need to collaborate
5. If escalation to orchestrator is needed

TASK TYPES AND PARAMETERS:
EMAIL TASKS:
- "email_summarization" - summarize emails
  Parameters: {{"count": number, "timeframe": "recent|today|week", "priority": "high|all"}}
- "email_classification" - classify emails by priority/category
  Parameters: {{"criteria": "priority|sender|category", "count": number}}
- "email_search" - search emails with query
  Parameters: {{"query": "search terms", "sender": "email", "timeframe": "days"}}
- "email_composition" - compose or reply to emails
  Parameters: {{"type": "reply|compose|forward", "recipient": "email", "subject": "text"}}
- "inbox_management" - organize and manage inbox
  Parameters: {{"action": "organize|clean|archive", "criteria": "age|importance"}}
- "list_emails" - show email options when user is unclear

CALENDAR TASKS:
- "list_events" - show upcoming events
  Parameters: {{"timeframe": "today|tomorrow|week|month", "count": number}}
- "create_event" - create calendar event
  Parameters: {{"title": "text", "time": "datetime", "duration": "minutes"}}
- "analyze_schedule" - analyze schedule patterns
  Parameters: {{"period": "week|month", "focus": "conflicts|productivity|balance"}}
- "check_availability" - find free time
  Parameters: {{"duration": "minutes", "timeframe": "today|week", "preferences": "morning|afternoon"}}
- "resolve_conflicts" - fix scheduling conflicts
  Parameters: {{"period": "week|month", "priority": "work|personal"}}
- "optimize_schedule" - suggest improvements
  Parameters: {{"focus": "efficiency|balance|meetings", "period": "week"}}

SEARCH TASKS:
- "web_search" - search for information
  Parameters: {{"query": "search terms", "type": "factual|research|recent"}}
- "analyze_results" - analyze search patterns
- "optimize_query" - improve search terms

AUTONOMOUS DECISION EXAMPLES:
- "How many emails should I summarize?" → Intelligently ask for count, default to 5-10 if user is vague
- "Search for important emails" → Use parameters={{"query": "important OR urgent", "priority": "high"}}
- "Check my schedule for conflicts" → Use task_type="resolve_conflicts", parameters={{"period": "week"}}
- "Find time for a 30-minute meeting tomorrow" → Use task_type="check_availability", parameters={{"duration": 30, "timeframe": "tomorrow"}}

CONVERSATION CONTEXT HANDLING:
- If assistant just drafted an email and user says "send that/it/this" → route="email", task_type="email_composition" 
- If assistant showed calendar events and user asks "schedule meeting" → route="calendar", task_type="create_event"
- If assistant provided search results and user asks "search more" → route="search", task_type="web_search"
- Look for follow-up actions that build on previous assistant responses

AGENT COLLABORATION:
- If task requires multiple agents, set route="orchestrator" with collaboration plan
- Agents can request help: parameters={{"needs_help_from": "calendar|email|search", "reason": "explanation"}}
- Escalation triggers: complex multi-step tasks, cross-domain requests, user ambiguity
"""

            response = await self.gemini_mcp_client.chat(prompt)
            
            # Parse the JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                return {
                    'route': analysis.get('route', 'fallback'),
                    'confidence': analysis.get('confidence', 0.8),
                    'reason': analysis.get('analysis', 'LLM-driven context analysis'),
                    'task_type': analysis.get('task_type', 'general'),
                    'user_intent': analysis.get('user_intent', ''),
                    'next_action': analysis.get('next_action', ''),
                    'parameters': analysis.get('parameters', {}),
                    'keyword_analysis': {'method': 'llm_context'},
                    'ai_analysis': {'method': 'llm_context'}
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    'route': 'fallback',
                    'confidence': 0.3,
                    'reason': 'LLM context analysis failed to parse'
                }
                
        except Exception as e:
            self.logger.error(f"Conversation context analysis failed: {e}")
            return {
                'route': 'fallback', 
                'confidence': 0.2,
                'reason': f'Context analysis error: {str(e)}'
            }

    async def route_request(self, user_input: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        self.routing_stats['total_requests'] += 1

        keyword_analysis = await self._analyze_keywords(user_input)
        ai_analysis = await self._analyze_with_gemini(user_input, conversation_history)

        route_decision = await self._make_routing_decision(user_input, keyword_analysis, ai_analysis, conversation_history)

        if route_decision['route'] != 'fallback':
            self.routing_stats['successful_routes'] += 1
        else:
            self.routing_stats['fallback_count'] += 1

        total = self.routing_stats['total_requests']
        current_avg = self.routing_stats['average_confidence']
        new_confidence = route_decision['confidence']
        self.routing_stats['average_confidence'] = (
            (current_avg * (total - 1) + new_confidence) / total
        )

        self.logger.info(f"Routed request to: {route_decision['route']} (confidence: {route_decision['confidence']:.2f})")

        return route_decision

    async def _analyze_keywords(self, user_input: str) -> Dict[str, Any]:
        input_lower = user_input.lower()
        scores = {}

        for route, keywords in self.route_patterns.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword in input_lower:
                    weight = 2 if keyword == input_lower.strip() else 1
                    score += weight
                    matched_keywords.append(keyword)

            if keywords:
                base_score = score / len(keywords)
                boosted_score = min(base_score * 1.5 + (0.3 if score > 0 else 0), 1.0)
                scores[route] = {
                    'score': boosted_score,
                    'matched_keywords': matched_keywords,
                    'match_count': score
                }

        if scores:
            best_route = max(scores.keys(), key=lambda x: scores[x]['score'])
            best_score = scores[best_route]['score']

            return {
                'best_route': best_route,
                'confidence': min(best_score * 2, 1.0),
                'all_scores': scores,
                'method': 'keyword_analysis'
            }

        return {
            'best_route': 'fallback',
            'confidence': 0.0,
            'all_scores': {},
            'method': 'keyword_analysis'
        }

    async def _analyze_with_gemini(self, user_input: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Build context from conversation history
        context = ""
        if conversation_history and len(conversation_history) > 1:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            context = "\nConversation context:\n"
            for msg in recent_history[:-1]:  # Exclude current message
                if 'user' in msg:
                    context += f"User: {msg['text']}\n"
                elif 'assistant' in msg:
                    context += f"Assistant: {msg['text']}\n"
            context += "\nCurrent request: " + user_input
        
        prompt = (
            """Classify the following user request into one of the following categories:
            - email
            - calendar
            - search
            - multi_agent
            
            Consider the conversation context to understand if this is a follow-up request.
            For example, if the user previously asked about emails and now says "summarize", 
            this should be classified as "email" not "search".
            
            Respond with only the category name.
            """ + (context if context else f"\nRequest: {user_input}")
        )
        try:
            response = await self.gemini_mcp_client.chat(prompt)
            ai_category = response.lower().strip()

            return {
                'best_route': ai_category if ai_category in ['email', 'calendar', 'search', 'multi_agent'] else 'fallback',
                'confidence': 0.75 if ai_category != 'fallback' else 0.0,
                'reasoning': f"Gemini classified as '{ai_category}'",
                'method': 'ai_analysis',
                'task_type': 'general'
            }
        except Exception as e:
            self.logger.warning(f"Gemini classification failed: {e}")
            return {
                'best_route': 'fallback',
                'confidence': 0.0,
                'reasoning': 'Gemini classification error',
                'method': 'ai_analysis',
                'task_type': 'general'
            }

    async def _make_routing_decision(self, user_input: str, keyword_analysis: Dict[str, Any], ai_analysis: Dict[str, Any], conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        keyword_route = keyword_analysis['best_route']
        keyword_confidence = keyword_analysis['confidence']
        ai_route = ai_analysis['best_route']
        ai_confidence = ai_analysis['confidence']
        
        # Use LLM for intelligent conversation context analysis
        if conversation_history and len(conversation_history) > 1:
            context_analysis = await self._analyze_conversation_context(user_input, conversation_history)
            if context_analysis['confidence'] > 0.7:
                return context_analysis

        if keyword_route == ai_route and keyword_route != 'fallback':
            final_confidence = min((keyword_confidence + ai_confidence) / 2 * 1.5, 1.0)
            final_route = keyword_route
            reasoning = f"Both keyword and AI analysis agree on {final_route}"

        elif ai_confidence > 0.7:
            final_route = ai_route
            final_confidence = ai_confidence
            reasoning = f"High confidence AI classification: {ai_analysis.get('reasoning', '')}"

        elif keyword_confidence > 0.5:
            final_route = keyword_route
            final_confidence = keyword_confidence
            reasoning = f"Keyword analysis suggests {keyword_route}"

        else:
            if self._is_complex_request(user_input):
                final_route = 'orchestrator'
                final_confidence = 0.6
                reasoning = "Complex request requiring multiple agents"
            else:
                final_route = 'fallback'
                final_confidence = 0.0
                reasoning = "No clear routing path found"

        return {
            'route': final_route,
            'confidence': final_confidence,
            'reason': reasoning,
            'task_type': ai_analysis.get('task_type', 'general'),
            'keyword_analysis': keyword_analysis,
            'ai_analysis': ai_analysis
        }

    def _is_complex_request(self, user_input: str) -> bool:
        input_lower = user_input.lower()
        complexity_indicators = [
            'and then', 'after that', 'also', 'plus',
            'schedule and send', 'email and calendar',
            'organize and prioritize', 'multiple',
            'both', 'all my', 'everything'
        ]

        domain_count = 0
        for domain_keywords in self.route_patterns.values():
            if any(keyword in input_lower for keyword in domain_keywords):
                domain_count += 1

        return (
            domain_count >= 2 or
            any(indicator in input_lower for indicator in complexity_indicators) or
            len(user_input.split()) > 20
        )

    async def handle_agent_escalation(self, state: AssistantState) -> Dict[str, Any]:
        """Handle escalation requests from agents with intelligent coordination."""
        try:
            escalation_request = state.get('escalation_request', {})
            agent_help_request = state.get('agent_help_request', {})
            
            if escalation_request:
                # Handle escalation from agent
                escalated_from = escalation_request.get('escalated_from', 'unknown')
                reason = escalation_request.get('reason', 'Unknown reason')
                user_input = escalation_request.get('user_input', '')
                
                prompt = f"""As an intelligent orchestrator, handle this escalation from {escalated_from}:

ESCALATION REASON: {reason}
ORIGINAL USER REQUEST: {user_input}

ORCHESTRATOR CAPABILITIES:
1. Coordinate multiple agents for complex tasks
2. Break down complex requests into agent-specific tasks
3. Handle rate limiting and error recovery strategies
4. Manage agent communication and data sharing
5. Provide intelligent fallback responses

DECISION OPTIONS:
- "coordinate_agents": Plan multi-agent collaboration
- "retry_with_strategy": Implement intelligent retry/backoff
- "provide_alternative": Offer alternative approach
- "request_user_clarification": Need more information from user

Respond in JSON format:
{{
    "action": "coordinate_agents|retry_with_strategy|provide_alternative|request_user_clarification",
    "plan": "Detailed plan for handling the escalation",
    "agents_needed": ["email", "calendar", "search"],
    "task_sequence": ["task1", "task2", "task3"],
    "user_message": "Message to show user",
    "parameters": {{"key": "value"}}
}}"""

                response = await self.gemini_mcp_client.chat(prompt)
                return self._parse_orchestrator_response(response)
                
            elif agent_help_request:
                # Handle agent collaboration request
                requesting_agent = agent_help_request.get('requesting_agent', 'unknown')
                help_from = agent_help_request.get('help_from', 'unknown')
                reason = agent_help_request.get('reason', 'Unknown reason')
                
                return await self._coordinate_agent_collaboration(requesting_agent, help_from, reason, state)
            
            return {
                'response': 'No escalation or help request found',
                'metadata': {'handled_by': 'orchestrator'}
            }
            
        except Exception as e:
            self.logger.error(f"Escalation handling failed: {e}")
            return {
                'response': 'I encountered an issue handling the escalation request.',
                'metadata': {'error': str(e)}
            }

    async def _coordinate_agent_collaboration(self, requesting_agent: str, help_from: str, reason: str, state: AssistantState) -> Dict[str, Any]:
        """Coordinate collaboration between agents."""
        try:
            user_input = state.get('user_input', '')
            
            prompt = f"""Coordinate collaboration between agents:

REQUESTING AGENT: {requesting_agent}
NEEDS HELP FROM: {help_from}
REASON: {reason}
USER REQUEST: {user_input}

COLLABORATION STRATEGIES:
1. Sequential execution: One agent completes, then passes to next
2. Parallel execution: Agents work simultaneously on different aspects
3. Data sharing: One agent provides data for another to use
4. Confirmation handoff: First agent seeks second agent's verification

Plan the optimal collaboration approach:
{{
    "collaboration_type": "sequential|parallel|data_sharing|confirmation",
    "execution_plan": "Step by step plan",
    "first_agent_task": "What the requesting agent should do",
    "second_agent_task": "What the helper agent should do", 
    "data_to_share": "What information needs to be passed",
    "user_message": "Status update for user"
}}"""

            response = await self.gemini_mcp_client.chat(prompt)
            return self._parse_orchestrator_response(response)
            
        except Exception as e:
            return {
                'response': f'Collaboration coordination failed: {str(e)}',
                'metadata': {'error': str(e)}
            }

    def _parse_orchestrator_response(self, response: str) -> Dict[str, Any]:
        """Parse orchestrator LLM response."""
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    'response': parsed.get('user_message', response),
                    'metadata': {
                        'action': parsed.get('action', 'unknown'),
                        'plan': parsed.get('plan', ''),
                        'orchestrator_decision': parsed
                    }
                }
        except:
            pass
        
        return {
            'response': response,
            'metadata': {'parsed': False}
        }

    async def process_complex_request(self, user_input: str, task_type: str = 'general') -> Dict[str, Any]:
        # For search queries, directly execute search and return concise results
        if task_type == 'web_search' or any(word in user_input.lower() for word in ['what is', 'who is', 'capital of', 'search']):
            try:
                # Try direct search through search tools first
                search_results = None
                if hasattr(self.gemini_mcp_client, 'search_tools') and self.gemini_mcp_client.search_tools:
                    try:
                        search_results = await self.gemini_mcp_client.search_tools.web_search(user_input, 5)
                    except Exception as e:
                        self.logger.warning(f"Search tools failed in orchestrator: {e}")
                
                # Fallback to MCP client method
                if not search_results or not search_results.get('results'):
                    search_results = await self.gemini_mcp_client._search_web(user_input, 5)
                
                if search_results and 'results' in search_results and search_results['results']:
                    # Extract direct answer from top result
                    top_result = search_results['results'][0]
                    answer_prompt = f"""Based on this search result, provide a direct, concise answer (1-2 sentences max) to: {user_input}
                    
Result: {top_result.get('title', '')} - {top_result.get('description', '')}
                    
Provide only the factual answer, no explanations."""
                    
                    concise_answer = await self.gemini_mcp_client.chat(answer_prompt)
                    return {
                        'response': concise_answer,
                        'metadata': {'task_type': task_type, 'source': 'search_direct'}
                    }
            except Exception as e:
                self.logger.warning(f"Direct search failed: {e}")
        
        return {
            'response': f"I'll coordinate actions across multiple agents for: {user_input}",
            'metadata': {'task_type': task_type}
        }

    async def verify_response(self, user_input: str, final_response: str, current_agent: str) -> Dict[str, float]:
        try:
            prompt = (
                f"""
                Evaluate the following response from the assistant.
                User request: {user_input}
                Assistant response: {final_response}

                Rate the response on:
                - Quality (how clear and helpful it is) [0-1 scale]
                - Completeness (did it address the request?) [0-1 scale]

                Return as JSON with keys 'quality' and 'completeness'.
                """
            )
            rating = await self.gemini_mcp_client.chat(prompt)
            return eval(rating)  # Simplified for now, can switch to `json.loads`
        except:
            return {'quality': 0.0, 'completeness': 0.0}

    async def handle_fallback(self, user_input: str, error_log: List[Dict[str, Any]]) -> str:
        """Concise fallback handler for unroutable requests."""
        try:
            # First try direct search for informational queries
            if any(word in user_input.lower() for word in ['what', 'who', 'when', 'where', 'how', 'capital']):
                try:
                    # Try search tools first
                    search_results = None
                    if hasattr(self.gemini_mcp_client, 'search_tools') and self.gemini_mcp_client.search_tools:
                        try:
                            search_results = await self.gemini_mcp_client.search_tools.web_search(user_input, 3)
                        except Exception as e:
                            self.logger.warning(f"Search tools failed in fallback: {e}")
                    
                    # Fallback to MCP client method
                    if not search_results or not search_results.get('results'):
                        search_results = await self.gemini_mcp_client._search_web(user_input, 3)
                    
                    if search_results and 'results' in search_results and search_results['results']:
                        top_result = search_results['results'][0]
                        answer_prompt = f"""Provide a direct, concise answer (1-2 sentences max) to: {user_input}
                        
Based on: {top_result.get('description', '')}
                        
Answer only the question asked."""
                        
                        answer = await self.gemini_mcp_client.chat(answer_prompt)
                        return answer
                except Exception:
                    pass
            
            # For other requests, provide brief guidance
            return f"I can't handle '{user_input}' directly. Try: 'search for [topic]', 'check emails', or 'show calendar'."
            
        except Exception as e:
            self.logger.error(f"Fallback handler error: {e}")
            return "Request not understood. Try 'search for [topic]', 'check emails', or 'show calendar'."

    async def health_check(self) -> bool:
        try:
            test = await self.gemini_mcp_client.chat("Ping")
            return True if test else False
        except:
            return False
