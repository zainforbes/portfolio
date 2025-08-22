# src/core/orchestrator.py
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.core.state_schema import AssistantState

class CoreOrchestrator:
    """
    Central orchestrator for the AI Assistant system.
    Handles request routing, response verification, and fallback logic.
    """
    
    def __init__(self, gemini_client, mcp_client):
        """
        Initialize the core orchestrator.
        
        Args:
            gemini_client: Gemini API client for AI operations
            mcp_client: MCP client for tool access
        """
        self.gemini_client = gemini_client
        self.mcp_client = mcp_client
        
        # Setup logging
        self.logger = logging.getLogger("orchestrator")
        self.logger.setLevel(logging.INFO)
        
        # Route classification patterns
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
            'task': [
                'task', 'todo', 'priority', 'deadline', 'project',
                'complete', 'finish', 'work on', 'assignment',
                'deliverable', 'milestone', 'progress'
            ]
        }
        
        # Performance tracking
        self.routing_stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'fallback_count': 0,
            'average_confidence': 0.0
        }

    async def route_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route user request to the appropriate agent.
        
        Args:
            user_input: User's request text
            context: Optional context information
            
        Returns:
            Routing decision with confidence and reasoning
        """
        try:
            self.routing_stats['total_requests'] += 1
            
            # Analyze request using multiple methods
            keyword_analysis = await self._analyze_keywords(user_input)
            ai_analysis = await self._analyze_with_ai(user_input, context)
            
            # Combine analyses for final routing decision
            route_decision = await self._make_routing_decision(
                user_input, keyword_analysis, ai_analysis
            )
            
            # Update stats
            if route_decision['route'] != 'fallback':
                self.routing_stats['successful_routes'] += 1
            else:
                self.routing_stats['fallback_count'] += 1
            
            # Update average confidence
            total = self.routing_stats['total_requests']
            current_avg = self.routing_stats['average_confidence']
            new_confidence = route_decision['confidence']
            self.routing_stats['average_confidence'] = (
                (current_avg * (total - 1) + new_confidence) / total
            )
            
            self.logger.info(f"Routed request to: {route_decision['route']} "
                           f"(confidence: {route_decision['confidence']:.2f})")
            
            return route_decision
            
        except Exception as e:
            self.logger.error(f"Routing failed: {e}")
            return {
                'route': 'fallback',
                'confidence': 0.0,
                'reason': f'Routing error: {str(e)}',
                'task_type': 'error_recovery'
            }

    async def _analyze_keywords(self, user_input: str) -> Dict[str, Any]:
        """Analyze request using keyword matching."""
        input_lower = user_input.lower()
        scores = {}
        
        # Score each route based on keyword matches
        for route, keywords in self.route_patterns.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in input_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Normalize score
            if keywords:
                scores[route] = {
                    'score': score / len(keywords),
                    'matched_keywords': matched_keywords,
                    'match_count': score
                }
        
        # Find best match
        if scores:
            best_route = max(scores.keys(), key=lambda x: scores[x]['score'])
            best_score = scores[best_route]['score']
            
            return {
                'best_route': best_route,
                'confidence': min(best_score * 2, 1.0),  # Cap at 1.0
                'all_scores': scores,
                'method': 'keyword_analysis'
            }
        
        return {
            'best_route': 'fallback',
            'confidence': 0.0,
            'all_scores': {},
            'method': 'keyword_analysis'
        }

    async def _analyze_with_ai(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze request using AI classification."""
        try:
            # Create classification prompt
            prompt = f"""
            Analyze this user request and classify it into one of these categories:
            - email: Email management, composition, reading, organizing
            - calendar: Scheduling, meetings, appointments, time management
            - task: Task management, priorities, project work, deadlines
            - multi_agent: Complex requests requiring multiple agents
            - fallback: Unclear or unsupported requests
            
            User request: "{user_input}"
            
            Provide your analysis as JSON with:
            - category: the main category
            - confidence: float 0-1
            - reasoning: brief explanation
            - task_type: specific type of task
            """
            
            if context:
                prompt += f"\nAdditional context: {context}"
            
            # Use Gemini for classification via MCP
            response = await self._call_gemini_tool(
                "gemini_generate",
                {
                    "prompt": prompt,
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "max_output_tokens": 200
                }
            )
            
            # Parse AI response
            ai_result = self._parse_ai_classification(response)
            ai_result['method'] = 'ai_analysis'
            
            return ai_result
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return {
                'best_route': 'fallback',
                'confidence': 0.0,
                'reasoning': f'AI analysis error: {str(e)}',
                'method': 'ai_analysis'
            }

    async def _call_gemini_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Call Gemini via MCP tools."""
        try:
            result = await self.mcp_client.call_tool(tool_name, parameters)
            return result.get('text', result.get('content', ''))
        except Exception as e:
            # Fallback to direct Gemini client if MCP fails
            if hasattr(self.gemini_client, 'generate_content_async'):
                response = await self.gemini_client.generate_content_async(
                    parameters.get('prompt', ''),
                    generation_config={
                        'temperature': parameters.get('temperature', 0.7),
                        'max_output_tokens': parameters.get('max_output_tokens', 1000)
                    }
                )
                return response.text
            raise e

    def _parse_ai_classification(self, response: str) -> Dict[str, Any]:
        """Parse AI classification response."""
        try:
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'best_route': data.get('category', 'fallback'),
                    'confidence': float(data.get('confidence', 0.0)),
                    'reasoning': data.get('reasoning', 'AI classification'),
                    'task_type': data.get('task_type', 'general')
                }
        except:
            pass
        
        # Fallback parsing using keywords
        response_lower = response.lower()
        
        for route in self.route_patterns.keys():
            if route in response_lower:
                return {
                    'best_route': route,
                    'confidence': 0.6,
                    'reasoning': f'Found {route} keyword in AI response',
                    'task_type': 'general'
                }
        
        return {
            'best_route': 'fallback',
            'confidence': 0.0,
            'reasoning': 'Could not parse AI response',
            'task_type': 'general'
        }

    async def _make_routing_decision(self, 
                                   user_input: str,
                                   keyword_analysis: Dict[str, Any],
                                   ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make final routing decision by combining analyses."""
        
        # Extract results
        keyword_route = keyword_analysis['best_route']
        keyword_confidence = keyword_analysis['confidence']
        
        ai_route = ai_analysis['best_route']
        ai_confidence = ai_analysis['confidence']
        
        # Decision logic
        if keyword_route == ai_route and keyword_route != 'fallback':
            # Both agree on a specific route
            final_confidence = min((keyword_confidence + ai_confidence) / 2 * 1.5, 1.0)
            final_route = keyword_route
            reasoning = f"Both keyword and AI analysis agree on {final_route}"
        
        elif ai_confidence > 0.7:
            # High confidence AI prediction
            final_route = ai_route
            final_confidence = ai_confidence
            reasoning = f"High confidence AI classification: {ai_analysis.get('reasoning', '')}"
        
        elif keyword_confidence > 0.5:
            # Decent keyword match
            final_route = keyword_route
            final_confidence = keyword_confidence
            reasoning = f"Keyword analysis suggests {keyword_route}"
        
        else:
            # No clear route - check for multi-agent scenarios
            if self._is_complex_request(user_input):
                final_route = 'coordinator'
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
        """Check if request requires multiple agents."""
        input_lower = user_input.lower()
        
        # Look for complexity indicators
        complexity_indicators = [
            'and then', 'after that', 'also', 'plus',
            'schedule and send', 'email and calendar',
            'organize and prioritize', 'multiple',
            'both', 'all my', 'everything'
        ]
        
        # Count different domain mentions
        domain_count = 0
        for domain_keywords in self.route_patterns.values():
            if any(keyword in input_lower for keyword in domain_keywords):
                domain_count += 1
        
        return (
            domain_count >= 2 or 
            any(indicator in input_lower for indicator in complexity_indicators) or
            len(user_input.split()) > 20  # Very long requests might be complex
        )

    async def verify_response(self, 
                            user_input: str, 
                            response: str, 
                            agent_name: str) -> Dict[str, Any]:
        """
        Verify the quality and completeness of an agent response.
        
        Args:
            user_input: Original user request
            response: Agent's response
            agent_name: Name of the agent that generated the response
            
        Returns:
            Verification scores and analysis
        """
        try:
            # Create verification prompt
            verification_prompt = f"""
            Evaluate this AI assistant response for quality and completeness:
            
            User Request: "{user_input}"
            Agent Response: "{response}"
            Agent: {agent_name}
            
            Rate the response on:
            1. Quality (0-1): How well-written and professional is it?
            2. Completeness (0-1): Does it fully address the user's request?
            3. Accuracy (0-1): Is the information correct and relevant?
            4. Helpfulness (0-1): How useful is this response to the user?
            
            Provide scores as JSON with brief explanations.
            """
            
            verification_response = await self._call_gemini_tool(
                "gemini_generate",
                {
                    "prompt": verification_prompt,
                    "temperature": 0.1,
                    "max_output_tokens": 300
                }
            )
            
            # Parse verification scores
            scores = self._parse_verification_response(verification_response)
            
            # Add metadata
            scores.update({
                'agent': agent_name,
                'response_length': len(response),
                'verification_timestamp': datetime.utcnow().isoformat()
            })
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Response verification failed: {e}")
            return {
                'quality': 0.5,
                'completeness': 0.5,
                'accuracy': 0.5,
                'helpfulness': 0.5,
                'error': str(e)
            }

    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse verification response into scores."""
        try:
            import json
            import re
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'quality': float(data.get('quality', 0.5)),
                    'completeness': float(data.get('completeness', 0.5)),
                    'accuracy': float(data.get('accuracy', 0.5)),
                    'helpfulness': float(data.get('helpfulness', 0.5)),
                    'explanation': data.get('explanation', '')
                }
        except:
            pass
        
        # Fallback: look for score patterns in text
        scores = {}
        for metric in ['quality', 'completeness', 'accuracy', 'helpfulness']:
            pattern = rf'{metric}[:\s]*([0-9]*\.?[0-9]+)'
            match = re.search(pattern, response.lower())
            if match:
                scores[metric] = min(float(match.group(1)), 1.0)
            else:
                scores[metric] = 0.5  # Default neutral score
        
        return scores

    async def handle_fallback(self, 
                            user_input: str, 
                            error_log: List[Dict[str, Any]] = None) -> str:
        """
        Handle fallback scenarios when normal routing fails.
        
        Args:
            user_input: Original user request
            error_log: List of errors that occurred
            
        Returns:
            Fallback response
        """
        try:
            # Analyze why we're in fallback
            error_context = ""
            if error_log:
                recent_errors = error_log[-3:]  # Last 3 errors
                error_context = f"Recent errors: {[e.get('error', '') for e in recent_errors]}"
            
            # Create fallback prompt
            fallback_prompt = f"""
            The user made this request but our specialized agents couldn't handle it:
            
            User Request: "{user_input}"
            {error_context}
            
            Provide a helpful general response that:
            1. Acknowledges their request
            2. Explains what you can help with
            3. Suggests alternative approaches
            4. Remains professional and supportive
            
            Keep it concise but helpful.
            """
            
            fallback_response = await self._call_gemini_tool(
                "gemini_generate",
                {
                    "prompt": fallback_prompt,
                    "temperature": 0.7,
                    "max_output_tokens": 500
                }
            )
            
            return fallback_response or self._get_default_fallback_response()
            
        except Exception as e:
            self.logger.error(f"Fallback handling failed: {e}")
            return self._get_default_fallback_response()

    def _get_default_fallback_response(self) -> str:
        """Get a safe default fallback response."""
        return """I understand you're looking for assistance, but I'm having difficulty processing your specific request right now. 

I can help you with:
• Email management (organizing, composing, prioritizing)
• Calendar scheduling (meetings, appointments, availability)
• Task management (priorities, deadlines, organization)

Could you try rephrasing your request or let me know which of these areas you'd like help with?"""

    async def health_check(self) -> bool:
        """
        Perform a health check on the orchestrator.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Test basic functionality
            test_result = await self.route_request("test health check")
            
            # Check if we got a valid routing result
            required_fields = ['route', 'confidence', 'reason']
            is_healthy = all(field in test_result for field in required_fields)
            
            if is_healthy:
                self.logger.info("Orchestrator health check passed")
            else:
                self.logger.warning("Orchestrator health check failed: missing required fields")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Orchestrator health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        return {
            'routing_stats': self.routing_stats.copy(),
            'route_patterns': {k: len(v) for k, v in self.route_patterns.items()},
            'status': 'active'
        }