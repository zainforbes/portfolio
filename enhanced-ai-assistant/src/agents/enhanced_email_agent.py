import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
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
from src.intelligence.agent_helpers import can_handle_task, suggest_routing

class EnhancedEmailAgent(EnhancedBaseAgent):
    """
    Enhanced Email Agent with full AI capabilities:
    - Intelligent conversation state management
    - Advanced prompt engineering for email composition
    - Hallucination mitigation and verification
    - Autonomous decision making for email tasks
    - Resource optimization and caching
    - Error recovery and fallback strategies
    """
    
    def __init__(self, mcp_client, agent_name: str = "EnhancedEmailAgent"):
        capabilities = [
            'email_composition', 'email_search', 'email_summarization',
            'email_classification', 'inbox_management', 'conversation_state_management',
            'draft_management', 'send_confirmation_detection'
        ]
        
        super().__init__(mcp_client, agent_name, capabilities)
        
        # Email-specific configuration
        self.draft_storage: Dict[str, Any] = {}
        self.conversation_patterns = self._initialize_conversation_patterns()
        
    def get_task_types(self) -> List[TaskType]:
        """Return email task types this agent can handle"""
        return [
            TaskType.EMAIL_COMPOSITION,
            TaskType.EMAIL_SEARCH,
            TaskType.EMAIL_SUMMARIZATION,
            TaskType.EMAIL_CLASSIFICATION,
            TaskType.INBOX_MANAGEMENT
        ]
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize email-specific prompt templates"""
        return {
            'conversation_analysis': """ADVANCED EMAIL CONVERSATION ANALYSIS

Analyze this conversation for email composition context:

CONVERSATION HISTORY:
{conversation_history}

CURRENT USER INPUT: "{user_input}"

EXTRACT WITH HIGH PRECISION:
1. Email recipients (names, emails, pronouns → referents)
2. Subject/topic information 
3. Message content and intent
4. Tone/style preferences
5. Send confirmation signals

CONVERSATION STATE MANAGEMENT:
- Track entities across turns (e.g., "her" → previously mentioned email)
- Identify partial information from previous exchanges
- Detect completion vs. continuation signals
- Recognize send confirmations ("send it", "yes", "go ahead")

RESPOND WITH STRUCTURED JSON:
{{
    "recipients": {{"email_or_name": "confidence_score_0_to_1"}},
    "subject": {{"potential_subject": "confidence_score"}},
    "content": {{"message_intent": "confidence_score"}},
    "tone": {{"tone_preference": "confidence_score"}},
    "send_confirmation": {{"detected": true/false, "confidence": 0.0-1.0}},
    "conversation_state": {{"complete": true/false, "missing": ["list", "of", "missing", "info"]}},
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of analysis"
}}

HALLUCINATION PREVENTION:
- Only extract explicitly mentioned information
- Mark inferred items with lower confidence
- Distinguish between certain and uncertain extractions""",

            'email_composition': """INTELLIGENT EMAIL COMPOSITION

CONTEXT ANALYSIS:
{context_analysis}

TASK: Create a complete, professional email based on extracted context

COMPOSITION GUIDELINES:
- Use ONLY information from context analysis
- Fill gaps with appropriate professional defaults
- Match specified tone (professional/casual/friendly)
- Include proper greeting, body, and closing
- Keep concise but complete

QUALITY REQUIREMENTS:
- No hallucinated information
- Contextually appropriate content
- Professional formatting
- Clear subject line

RESPOND WITH JSON:
{{
    "email": {{
        "to": "recipient_email_or_placeholder",
        "subject": "clear_subject_line",
        "body": "complete_email_body_with_formatting"
    }},
    "confidence": 0.0-1.0,
    "reasoning": "why this composition was chosen",
    "missing_info": ["list", "of", "any", "missing", "information"],
    "tone_applied": "professional|casual|friendly"
}}""",

            'verification': """EMAIL VERIFICATION AND QUALITY CHECK

ORIGINAL REQUEST: "{original_request}"
GENERATED EMAIL: {generated_email}
CONVERSATION CONTEXT: {conversation_context}

VERIFICATION CHECKLIST:
1. Recipient accuracy (matches conversation)
2. Subject relevance (matches intent)
3. Content alignment (addresses request)
4. Tone appropriateness (matches context)
5. Professional quality (grammar, structure)
6. No hallucinated information
7. Conversation continuity

CONFIDENCE FACTORS:
- Information explicitly provided: +0.3
- Logical inference from context: +0.2
- Professional template usage: +0.1
- Assumptions made: -0.2
- Missing critical info: -0.3

RESPOND WITH JSON:
{{
    "verification_passed": true/false,
    "confidence_score": 0.0-1.0,
    "issues_found": ["list", "of", "issues"],
    "quality_scores": {{
        "recipient_accuracy": 0.0-1.0,
        "content_alignment": 0.0-1.0,
        "tone_appropriateness": 0.0-1.0,
        "professional_quality": 0.0-1.0
    }},
    "recommendations": ["list", "of", "improvements"]
}}"""
        }
    
    def _initialize_conversation_patterns(self) -> Dict[str, Any]:
        """Initialize conversation pattern recognition"""
        return {
            'send_confirmations': [
                'send it', 'send that', 'send the email', 'send this',
                'yes send', 'yes please send', 'go ahead', 'proceed',
                'send now', 'send the message', 'yes', 'confirm send'
            ],
            'recipient_patterns': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'send (?:it )?to (.+?)(?:\s|$|,)',  # "send it to John"
                r'email (.+?)(?:\s|$|,)',  # "email John"
            ],
            'subject_patterns': [
                r'subject.*?["\']([^"\']+)["\']',  # subject "..."
                r'about (.+?)(?:\s|$|,)',  # about vacation
                r'regarding (.+?)(?:\s|$|,)',  # regarding meeting
            ],
            'content_patterns': [
                r'tell (?:her|him|them) (.+?)(?:\s|$|\.|!|\?)',
                r'say (.+?)(?:\s|$|\.|!|\?)',
                r'message (.+?)(?:\s|$|\.|!|\?)',
                r'let (?:her|him|them) know (.+?)(?:\s|$|\.|!|\?)'
            ]
        }
    
    async def execute(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Main execution method using the full AI pipeline"""
        return await self.execute_with_full_pipeline(state)
    
    async def _execute_task(self, decision: AgentDecision, 
                          state: EnhancedAssistantState, 
                          context: Dict[str, Any]) -> TaskResult:
        """Execute email-specific tasks with proper state management"""
        task_type = decision.parameters.get('task_type', TaskType.EMAIL_COMPOSITION.value)
        
        # Record the decision in state
        record_agent_decision(state, decision)
        
        try:
            if task_type == TaskType.EMAIL_COMPOSITION.value:
                return await self._handle_email_composition(decision, state, context)
            elif task_type == TaskType.EMAIL_SEARCH.value:
                return await self._handle_email_search(decision, state, context)
            elif task_type == TaskType.EMAIL_SUMMARIZATION.value:
                return await self._handle_email_summarization(decision, state, context)
            elif task_type == TaskType.EMAIL_CLASSIFICATION.value:
                return await self._handle_email_classification(decision, state, context)
            elif task_type == TaskType.INBOX_MANAGEMENT.value:
                return await self._handle_inbox_management(decision, state, context)
            else:
                error_msg = f"Unknown task type: {task_type}"
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
    
    async def _handle_email_composition(self, decision: AgentDecision,
                                      state: EnhancedAssistantState,
                                      context: Dict[str, Any]) -> TaskResult:
        """Handle email composition with advanced conversation state management"""
        try:
            # Check rate limits before making API calls
            if is_rate_limited(state, 'gemini') or is_rate_limited(state, 'gmail_api'):
                return await self._handle_rate_limited_composition(decision, state, context)
            
            # 1. Check for pending draft and send confirmation
            pending_draft = self._get_pending_draft(state)
            if pending_draft and self._is_send_confirmation(state.get('user_input', '')):
                return await self._send_pending_draft(state, pending_draft)
            
            # 2. Analyze conversation for email context
            conversation_analysis = await self._analyze_email_conversation(state)
            
            # 3. Determine if we have enough information
            if conversation_analysis['conversation_state']['complete']:
                # Compose email with available information
                email_result = await self._compose_email(conversation_analysis, state)
                
                if email_result['confidence'] > self.confidence_threshold:
                    # Store draft and present to user
                    draft = email_result['email']
                    self._store_draft(state, draft, email_result['confidence'])
                    
                    response = self._format_draft_response(draft, email_result)
                    
                    return TaskResult(
                        success=True,
                        data=response,
                        confidence=email_result['confidence'],
                        task_type='email_composition',
                        agent=self.agent_name
                    )
            
            # 4. Request missing information
            missing_info = conversation_analysis['conversation_state']['missing']
            clarification = self._generate_clarification_request(missing_info, conversation_analysis)
            
            return TaskResult(
                success=True,
                data=clarification,
                confidence=0.8,
                task_type='email_composition',
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type='email_composition',
                agent=self.agent_name,
                error=str(e)
            )
    
    async def _analyze_email_conversation(self, state: EnhancedAssistantState) -> Dict[str, Any]:
        """Analyze conversation for email composition context"""
        conversation_history = state.get('conversation_history', [])
        user_input = state.get('user_input', '')
        
        # Format conversation for analysis
        history_text = self._format_conversation_for_analysis(conversation_history)
        
        # Use LLM for advanced analysis
        prompt = self.prompt_templates['conversation_analysis'].format(
            conversation_history=history_text,
            user_input=user_input
        )
        
        try:
            response = await self.generate_response(prompt)
            analysis = self._parse_json_response(response)
            
            # Validate and enhance analysis
            analysis = self._validate_conversation_analysis(analysis, state)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Conversation analysis failed: {e}")
            return self._get_fallback_conversation_analysis(conversation_history, user_input)
    
    async def _compose_email(self, conversation_analysis: Dict[str, Any], 
                           state: EnhancedAssistantState) -> Dict[str, Any]:
        """Compose email using conversation analysis"""
        prompt = self.prompt_templates['email_composition'].format(
            context_analysis=json.dumps(conversation_analysis, indent=2)
        )
        
        try:
            response = await self.generate_response(prompt)
            composition_result = self._parse_json_response(response)
            
            # Validate composition
            if self.verification_enabled:
                composition_result = await self._verify_email_composition(
                    composition_result, conversation_analysis, state
                )
            
            return composition_result
            
        except Exception as e:
            self.logger.error(f"Email composition failed: {e}")
            return {
                'email': {
                    'to': '[Enter recipient]',
                    'subject': '[Enter subject]',
                    'body': '[Enter message]'
                },
                'confidence': 0.3,
                'reasoning': f'Composition failed: {str(e)}'
            }
    
    async def _verify_email_composition(self, composition: Dict[str, Any],
                                      conversation_analysis: Dict[str, Any],
                                      state: EnhancedAssistantState) -> Dict[str, Any]:
        """Verify email composition quality and accuracy"""
        prompt = self.prompt_templates['verification'].format(
            original_request=state.get('user_input', ''),
            generated_email=json.dumps(composition.get('email', {}), indent=2),
            conversation_context=json.dumps(conversation_analysis, indent=2)
        )
        
        try:
            response = await self.generate_response(prompt)
            verification = self._parse_json_response(response)
            
            # Apply verification results
            if not verification.get('verification_passed', False):
                composition['confidence'] *= 0.6  # Reduce confidence
                composition['verification_issues'] = verification.get('issues_found', [])
            
            return composition
            
        except Exception as e:
            self.logger.warning(f"Email verification failed: {e}")
            return composition
    
    def _is_send_confirmation(self, user_input: str) -> bool:
        """Enhanced send confirmation detection"""
        user_lower = user_input.lower().strip()
        return any(phrase in user_lower for phrase in self.conversation_patterns['send_confirmations'])
    
    def _get_pending_draft(self, state: EnhancedAssistantState) -> Optional[Dict[str, Any]]:
        """Get pending email draft from state"""
        return state.get('active_context', {}).get('pending_email_draft')
    
    def _store_draft(self, state: EnhancedAssistantState, draft: Dict[str, Any], confidence: float) -> None:
        """Store email draft in state"""
        if 'active_context' not in state:
            state['active_context'] = {}
        
        state['active_context']['pending_email_draft'] = {
            **draft,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _send_pending_draft(self, state: EnhancedAssistantState, 
                                draft: Dict[str, Any]) -> TaskResult:
        """Send the pending email draft"""
        try:
            to = draft.get('to', '')
            subject = draft.get('subject', '')
            body = draft.get('body', '')
            
            if not to or '[Enter recipient]' in to:
                return TaskResult(
                    success=False,
                    data="Cannot send email: No recipient specified",
                    confidence=0.0,
                    task_type='email_composition',
                    agent=self.agent_name,
                    error="No recipient specified"
                )
            
            # Send email using MCP client
            result = await self.use_tool("send_email", {
                "to": to,
                "subject": subject,
                "body": body
            })
            
            # Clear pending draft
            if 'active_context' in state and 'pending_email_draft' in state['active_context']:
                del state['active_context']['pending_email_draft']
            
            success_message = f"✅ **Email sent successfully!**\n\n**To:** {to}\n**Subject:** {subject}\n\nYour message has been delivered."
            
            return TaskResult(
                success=True,
                data=success_message,
                confidence=0.95,
                task_type='email_composition',
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type='email_composition',
                agent=self.agent_name,
                error=f"Failed to send email: {str(e)}"
            )
    
    def _format_draft_response(self, draft: Dict[str, Any], composition_result: Dict[str, Any]) -> str:
        """Format draft presentation to user"""
        response = f"📧 **I've created an email draft for you:**\n\n"
        response += f"**To:** {draft.get('to', '[Enter recipient]')}\n"
        response += f"**Subject:** {draft.get('subject', '[Enter subject]')}\n\n"
        response += f"**Message:**\n{draft.get('body', '[Enter message]')}\n\n"
        
        confidence = composition_result.get('confidence', 0.0)
        if confidence > 0.8:
            response += "Would you like me to send this email?"
        else:
            response += "Please review and let me know if you'd like me to send this or make any changes."
        
        return response
    
    def _generate_clarification_request(self, missing_info: List[str], 
                                     conversation_analysis: Dict[str, Any]) -> str:
        """Generate intelligent clarification request"""
        available_info = []
        
        # Show what we already have
        recipients = conversation_analysis.get('recipients', {})
        if recipients:
            best_recipient = max(recipients.items(), key=lambda x: x[1])[0]
            available_info.append(f"• Recipient: {best_recipient}")
        
        subjects = conversation_analysis.get('subject', {})
        if subjects:
            best_subject = max(subjects.items(), key=lambda x: x[1])[0]
            available_info.append(f"• Subject: {best_subject}")
        
        content = conversation_analysis.get('content', {})
        if content:
            best_content = max(content.items(), key=lambda x: x[1])[0]
            available_info.append(f"• Message: {best_content}")
        
        # Build clarification request
        clarification = "I need a bit more information to compose your email:\n\n"
        
        missing_items = []
        if 'recipient' in missing_info or not recipients:
            missing_items.append("• Who should receive the email?")
        if 'subject' in missing_info or not subjects:
            missing_items.append("• What should the subject line be?")
        if 'content' in missing_info or not content:
            missing_items.append("• What message would you like to send?")
        
        if missing_items:
            clarification += "\n".join(missing_items)
        
        if available_info:
            clarification += f"\n\n**I already have:**\n" + "\n".join(available_info)
        
        return clarification
    
    def _validate_conversation_analysis(self, analysis: Dict[str, Any], 
                                      state: EnhancedAssistantState) -> Dict[str, Any]:
        """Validate and enhance conversation analysis"""
        # Ensure required structure
        if 'conversation_state' not in analysis:
            analysis['conversation_state'] = {'complete': False, 'missing': ['recipient', 'subject', 'content']}
        
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5
        
        # Validate send confirmation detection
        user_input = state.get('user_input', '')
        if self._is_send_confirmation(user_input):
            analysis['send_confirmation'] = {'detected': True, 'confidence': 0.9}
        
        return analysis
    
    def _get_fallback_conversation_analysis(self, conversation_history: List[Dict], 
                                          user_input: str) -> Dict[str, Any]:
        """Fallback conversation analysis using pattern matching"""
        analysis = {
            'recipients': {},
            'subject': {},
            'content': {},
            'tone': {},
            'send_confirmation': {'detected': False, 'confidence': 0.0},
            'conversation_state': {'complete': False, 'missing': ['recipient', 'subject', 'content']},
            'confidence': 0.3,
            'reasoning': 'Fallback pattern matching used'
        }
        
        # Simple pattern matching
        all_text = ' '.join([msg.get('text', '') for msg in conversation_history] + [user_input])
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, all_text)
        for email in emails:
            analysis['recipients'][email] = 0.8
        
        # Extract content intent
        for pattern in self.conversation_patterns['content_patterns']:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                analysis['content'][match.strip()] = 0.6
        
        # Check send confirmation
        if self._is_send_confirmation(user_input):
            analysis['send_confirmation'] = {'detected': True, 'confidence': 0.9}
        
        return analysis
    
    def _format_conversation_for_analysis(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for LLM analysis"""
        formatted = []
        for msg in conversation_history[-10:]:  # Last 10 messages
            if 'user' in msg:
                formatted.append(f"User: {msg['text']}")
            elif 'assistant' in msg:
                formatted.append(f"Assistant: {msg['text']}")
        return '\n'.join(formatted)
    
    # Placeholder implementations for other email tasks
    async def _handle_email_search(self, decision: AgentDecision,
                                 state: EnhancedAssistantState,
                                 context: Dict[str, Any]) -> TaskResult:
        """Handle email search tasks"""
        return TaskResult(
            success=True,
            data="Email search functionality will be implemented soon.",
            confidence=0.8,
            task_type='email_search',
            agent=self.agent_name
        )
    
    async def _handle_email_summarization(self, decision: AgentDecision,
                                        state: EnhancedAssistantState,
                                        context: Dict[str, Any]) -> TaskResult:
        """Handle email summarization tasks"""
        try:
            # Update resource metrics
            update_resource_metrics(state, processing_time=0.1)
            
            return TaskResult(
                success=True,
                data="Email summarization functionality will be implemented soon.",
                confidence=0.8,
                task_type=TaskType.EMAIL_SUMMARIZATION.value,
                agent=self.agent_name
            )
        except Exception as e:
            return self._create_error_result(e, TaskType.EMAIL_SUMMARIZATION.value, state)
    
    async def _handle_email_classification(self, decision: AgentDecision,
                                         state: EnhancedAssistantState,
                                         context: Dict[str, Any]) -> TaskResult:
        """Handle email classification tasks"""
        try:
            # Update resource metrics
            update_resource_metrics(state, processing_time=0.1)
            
            return TaskResult(
                success=True,
                data="Email classification functionality will be implemented soon.",
                confidence=0.8,
                task_type=TaskType.EMAIL_CLASSIFICATION.value,
                agent=self.agent_name
            )
        except Exception as e:
            return self._create_error_result(e, TaskType.EMAIL_CLASSIFICATION.value, state)
    
    async def _handle_inbox_management(self, decision: AgentDecision,
                                     state: EnhancedAssistantState,
                                     context: Dict[str, Any]) -> TaskResult:
        """Handle inbox management tasks"""
        return TaskResult(
            success=True,
            data="Inbox management functionality will be implemented soon.",
            confidence=0.8,
            task_type='inbox_management',
            agent=self.agent_name
        )
    
    def _get_task_patterns(self) -> Dict[str, List[str]]:
        """Get email-specific task patterns"""
        return {
            'email_composition': ['send', 'compose', 'write', 'email', 'message'],
            'email_search': ['find email', 'search email', 'look for email'],
            'email_summarization': ['summarize email', 'email summary'],
            'inbox_management': ['organize inbox', 'manage email', 'clean inbox']
        }
    
    def _create_error_result(self, error: Exception, task_type: str, 
                           state: EnhancedAssistantState) -> TaskResult:
        """Create standardized error result with state logging"""
        error_record = {
            'agent': self.agent_name,
            'error': str(error),
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
            error=str(error)
        )
    
    async def _handle_rate_limited_composition(self, decision: AgentDecision, 
                                             state: EnhancedAssistantState, 
                                             context: Dict[str, Any]) -> TaskResult:
        """Handle rate-limited email composition"""
        fallback_response = "I'm temporarily rate-limited. Please try again in a moment, or provide more details for your email."
        
        return TaskResult(
            success=True,
            data=fallback_response,
            confidence=0.4,
            task_type=TaskType.EMAIL_COMPOSITION.value,
            agent=self.agent_name
        )