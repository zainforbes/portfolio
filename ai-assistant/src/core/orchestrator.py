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
                'date', 'tomorrow', 'today', 'next week'
            ],
            'search': [
                'search', 'find', 'look up', 'research', 'investigate',
                'what is', 'who is', 'how to', 'when did', 'where is',
                'google', 'browse', 'web', 'internet', 'online',
                'information about', 'details on', 'facts about',
                'brave search', 'web search'
            ]
        }

        self.routing_stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'fallback_count': 0,
            'average_confidence': 0.0
        }

    async def route_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.routing_stats['total_requests'] += 1

        keyword_analysis = await self._analyze_keywords(user_input)
        ai_analysis = await self._analyze_with_gemini(user_input)

        route_decision = await self._make_routing_decision(user_input, keyword_analysis, ai_analysis)

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

    async def _analyze_with_gemini(self, user_input: str) -> Dict[str, Any]:
        prompt = (
            """Classify the following user request into one of the following categories:
            - email
            - calendar
            - search
            - multi_agent
            Respond with only the category name.

            Request: """ + user_input + """
            """
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

    async def _make_routing_decision(self, user_input: str, keyword_analysis: Dict[str, Any], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        keyword_route = keyword_analysis['best_route']
        keyword_confidence = keyword_analysis['confidence']
        ai_route = ai_analysis['best_route']
        ai_confidence = ai_analysis['confidence']

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

    async def process_complex_request(self, user_input: str, task_type: str = 'general') -> Dict[str, Any]:
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
        """Enhanced fallback handler that provides helpful responses based on input."""
        try:
            # Analyze the user input to provide contextual help
            fallback_prompt = f"""
            The user asked: "{user_input}"
            
            I couldn't route this to a specific agent, but I should still be helpful.
            Please provide a useful response that:
            1. Acknowledges what they're trying to do
            2. Explains what I can help with instead
            3. Suggests how they could rephrase their request
            4. Offers specific examples of what I can do
            
            My capabilities include:
            - Email management (reading, organizing, composing emails)
            - Calendar management (scheduling, viewing events, finding availability)
            - Web search and research (finding information online)
            
            Be friendly and helpful while guiding them toward a request I can handle.
            """
            
            helpful_response = await self.gemini_mcp_client.chat(fallback_prompt)
            return helpful_response
            
        except Exception as e:
            self.logger.error(f"Fallback handler error: {e}")
            return f"""I'm sorry, I couldn't process your request: "{user_input}"

However, I can help you with:
• 📧 **Email tasks**: "show my unread emails" or "classify my emails by priority"  
• 📅 **Calendar tasks**: "show my upcoming events" or "when am I free tomorrow?"
• 🔍 **Search tasks**: "search for information about [topic]"

Could you try rephrasing your request using one of these examples?"""

    async def health_check(self) -> bool:
        try:
            test = await self.gemini_mcp_client.chat("Ping")
            return True if test else False
        except:
            return False
