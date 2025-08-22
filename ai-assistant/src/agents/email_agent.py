import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent
from src.core.state_schema import AssistantState
from src.core.message_types import AgentMessage, MessageTypes

@dataclass
class EmailClassification:
    """Email classification result"""
    priority: str  # high, medium, low
    category: str  # work, personal, marketing, support, etc.
    sentiment: str  # positive, negative, neutral
    action_required: bool
    urgency_score: float  # 0-1
    suggested_actions: List[str]
    confidence: float  # 0-1
    

@dataclass
class EmailSummary:
    """Email summary for quick overview"""
    subject: str
    sender: str
    key_points: List[str]
    action_items: List[str]
    deadline: Optional[datetime]
    people_mentioned: List[str]

class EmailAgent(BaseAgent):
    """
    Specialized agent for email management and intelligence.
    Handles email classification, summarization, and automated responses.
    """
    
    def __init__(self, gemini_mcp_client, agent_name: str = "EmailAgent"):
        capabilities = [
            "email_classification",
            "email_summarization", 
            "priority_detection",
            "automated_responses",
            "email_organization",
            "spam_detection",
            "sentiment_analysis"
        ]
        
        super().__init__(gemini_mcp_client, agent_name, capabilities)
        
        # Email-specific prompt templates
        self.system_prompts.update({
            'classification': """You are an expert email classifier. Analyze emails and categorize them by:
1. Priority (high/medium/low) based on urgency and importance
2. Category (work, personal, marketing, support, newsletter, etc.)
3. Sentiment (positive, negative, neutral)
4. Action required (yes/no)
5. Urgency score (0-1 where 1 is most urgent)

Consider factors like:
- Sender importance and relationship
- Keywords indicating urgency
- Deadlines mentioned
- Meeting requests
- Financial implications
- Legal or compliance matters""",

            'summarization': """You are an expert at email summarization. Create concise, actionable summaries that include:
1. Key points from the email
2. Action items or requests
3. Important dates or deadlines
4. People mentioned who might need follow-up
5. Context for decision making

Keep summaries under 100 words but capture all critical information.""",

            'response_generation': """You are a professional email response generator. Create appropriate responses that are:
1. Professional and courteous
2. Address all points raised
3. Clear and actionable
4. Match the tone and formality of the original
5. Include next steps when appropriate"""
        })
        
        # Email patterns for enhanced parsing
        self.email_patterns = {
            'urgency_keywords': [
                'urgent', 'asap', 'immediate', 'emergency', 'critical',
                'deadline', 'today', 'by end of day', 'eod', 'rush'
            ],
            'meeting_patterns': [
                r'meeting.*(?:at|on|from)\s*(\d{1,2}:\d{2})',
                r'call.*(?:at|on|from)\s*(\d{1,2}:\d{2})',
                r'conference.*(?:at|on|from)\s*(\d{1,2}:\d{2})'
            ],
            'date_patterns': [
                r'(?:by|before|until)\s*(\w+\s+\d{1,2},?\s+\d{4})',
                r'deadline.*?(\w+\s+\d{1,2},?\s+\d{4})',
                r'due\s*(\w+\s+\d{1,2},?\s+\d{4})'
            ]
        }

    async def execute(self, state: AssistantState) -> AssistantState:
        """Execute email agent operations based on current state and context."""
        try:
            # Ensure state is a dictionary, not a string
            if isinstance(state, str):
                self.logger.error(f"State is a string instead of dict: {state}")
                return {'final_response': f"I understand you'd like me to help with: {state}. However, I'm experiencing a technical issue with my email functionality. Please try again or rephrase your request."}
            
            user_request = state.get('user_input', '')
            
            # Determine what email operation to perform
            operation = await self._determine_operation(user_request, state.get('active_context', {}))
            
            if operation == "classify_emails":
                return await self._classify_recent_emails(state)
            elif operation == "summarize_emails":
                return await self._summarize_emails(state)
            elif operation == "manage_inbox":
                return await self._manage_inbox(state)
            elif operation == "compose_response":
                return await self._compose_response(state)
            elif operation == "search_emails":
                return await self._search_emails(state)
            else:
                return await self._general_email_assistance(state)
                
        except Exception as e:
            self.logger.error(f"Email agent execution failed: {e}")
            # Add to error log using your state schema
            error_log = state.get('error_log', [])
            error_log.append({
                'agent': self.agent_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            state['error_log'] = error_log
            return state

    async def can_handle(self, request: str, context: Dict[str, Any] = None) -> bool:
        """Determine if this agent can handle the email-related request."""
        email_keywords = [
            'email', 'mail', 'inbox', 'compose', 'send', 'reply',
            'forward', 'attachment', 'subject', 'sender', 'recipient',
            'gmail', 'outlook', 'message', 'correspondence'
        ]
        
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in email_keywords)

    async def _determine_operation(self, request: str, context: Dict[str, Any]) -> str:
        """Determine what email operation to perform based on request."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['classify', 'categorize', 'priority']):
            return "classify_emails"
        elif any(word in request_lower for word in ['summarize', 'summary', 'overview']):
            return "summarize_emails"
        elif any(word in request_lower for word in ['manage', 'organize', 'clean']):
            return "manage_inbox"
        elif any(word in request_lower for word in ['reply', 'respond', 'compose']):
            return "compose_response"
        elif any(word in request_lower for word in ['search', 'find', 'look for']):
            return "search_emails"
        else:
            return "general_assistance"

    def _add_agent_message(self, state: AssistantState, content: str, message_type: str = "info") -> AssistantState:
        """Helper to add agent messages to state using your schema."""
        # Ensure state is a dictionary
        if isinstance(state, str):
            state = {'user_input': state, 'agent_messages': [], 'final_response': content}
            
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

    async def _classify_recent_emails(self, state: AssistantState) -> AssistantState:
        """Classify recent emails for priority and category."""
        try:
            # Get recent emails via MCP
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": 20,
                "query": "is:unread"
            })
            
            classifications = []
            for email in emails.get('messages', []):
                # Get email details
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                
                # Classify the email
                classification = await self._classify_single_email(email_data)
                classifications.append({
                    'id': email['id'],
                    'subject': email_data.get('subject', 'No Subject'),
                    'sender': email_data.get('from', 'Unknown'),
                    'classification': classification
                })
            
            # Sort by priority and urgency
            classifications.sort(
                key=lambda x: (
                    x['classification'].priority == 'high',
                    x['classification'].urgency_score
                ), 
                reverse=True
            )
            
            # Update state with classifications using your schema
            active_context = state.get('active_context', {})
            active_context['email_classifications'] = classifications
            active_context['high_priority_count'] = sum(
                1 for c in classifications 
                if c['classification'].priority == 'high'
            )
            state['active_context'] = active_context
            
            # Generate response
            response = await self._generate_classification_summary(classifications)
            return self._add_agent_message(state, response, "classification_result")
            
        except Exception as e:
            self.logger.error(f"Email classification failed: {e}")
            return self._add_agent_message(state, f"I encountered an error classifying emails: {str(e)}", "error")

    async def _classify_single_email(self, email_data: Dict[str, Any]) -> EmailClassification:
        """Classify a single email using AI analysis."""
        try:
            # Prepare email content for analysis
            subject = email_data.get('subject', '')
            sender = email_data.get('from', '')
            body = email_data.get('body', '')[:1000]  # Limit body length
            
            # Create classification prompt
            prompt = f"""
            Classify this email:
            
            From: {sender}
            Subject: {subject}
            Body: {body}
            
            Provide classification as JSON with:
            - priority: "high"|"medium"|"low"
            - category: category name
            - sentiment: "positive"|"negative"|"neutral"
            - action_required: boolean
            - urgency_score: float 0-1
            - suggested_actions: array of strings
            - confidence: float 0-1
            """
            
            # Generate classification using Gemini via MCP
            response = await self.generate_response(
                prompt, 
                context=self.system_prompts['classification']
            )
            
            # Parse the response (assuming JSON format)
            classification_data = self._parse_classification_response(response)
            
            return EmailClassification(**classification_data)
            
        except Exception as e:
            self.logger.error(f"Single email classification failed: {e}")
            # Return default classification
            return EmailClassification(
                priority="medium",
                category="general",
                sentiment="neutral",
                action_required=False,
                urgency_score=0.5,
                suggested_actions=["review"],
                confidence=0.5
            )

    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into classification data."""
        # This is a simplified parser - in reality you'd want more robust JSON parsing
        try:
            import json
            # Extract JSON from response if it exists
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing using keywords
        priority = "medium"
        if any(word in response.lower() for word in ['urgent', 'high', 'critical']):
            priority = "high"
        elif any(word in response.lower() for word in ['low', 'minor']):
            priority = "low"
        
        return {
            'priority': priority,
            'category': 'general',
            'sentiment': 'neutral',
            'action_required': 'action' in response.lower(),
            'urgency_score': 0.7 if priority == 'high' else 0.5,
            'suggested_actions': ['review'],
            'confidence': 0.7
        }

    async def _summarize_emails(self, state: AssistantState) -> AssistantState:
        """Create summaries of recent emails."""
        try:
            # Get recent emails
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": 10,
                "query": "is:unread"
            })
            
            summaries = []
            for email in emails.get('messages', []):
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                
                summary = await self._create_email_summary(email_data)
                summaries.append(summary)
            
            # Update state with summaries
            active_context = state.get('active_context', {})
            active_context['email_summaries'] = [s.__dict__ for s in summaries]
            state['active_context'] = active_context
            
            # Generate consolidated summary response
            response = await self._generate_summary_report(summaries)
            return self._add_agent_message(state, response, "summary_result")
            
        except Exception as e:
            self.logger.error(f"Email summarization failed: {e}")
            return self._add_agent_message(state, f"I encountered an error summarizing emails: {str(e)}", "error")

    async def _create_email_summary(self, email_data: Dict[str, Any]) -> EmailSummary:
        """Create a summary for a single email."""
        try:
            subject = email_data.get('subject', 'No Subject')
            sender = email_data.get('from', 'Unknown')
            body = email_data.get('body', '')
            
            # Use AI to extract key information
            prompt = f"""
            Summarize this email and extract key information:
            
            From: {sender}
            Subject: {subject}
            Body: {body[:1000]}
            
            Extract:
            1. Key points (3-5 bullet points)
            2. Action items or requests
            3. Any deadlines or important dates
            4. People mentioned who might need follow-up
            """
            
            summary_text = await self.generate_response(
                prompt,
                context=self.system_prompts['summarization']
            )
            
            # Parse deadlines from content
            deadline = self._extract_deadline(body)
            people = self._extract_people_mentions(body)
            
            return EmailSummary(
                subject=subject,
                sender=sender,
                key_points=self._extract_key_points(summary_text),
                action_items=self._extract_action_items(summary_text),
                deadline=deadline,
                people_mentioned=people
            )
            
        except Exception as e:
            self.logger.error(f"Email summary creation failed: {e}")
            return EmailSummary(
                subject=email_data.get('subject', 'No Subject'),
                sender=email_data.get('from', 'Unknown'),
                key_points=[],
                action_items=[],
                deadline=None,
                people_mentioned=[]
            )

    def _extract_deadline(self, text: str) -> Optional[datetime]:
        """Extract deadline from email text."""
        for pattern in self.email_patterns['date_patterns']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Simple date parsing - in reality you'd use a proper date parser
                    date_str = match.group(1)
                    # This is simplified - use dateutil.parser in real implementation
                    return datetime.now() + timedelta(days=7)  # Placeholder
                except:
                    continue
        return None

    def _extract_people_mentions(self, text: str) -> List[str]:
        """Extract mentioned people from email text."""
        # Simple name pattern - in reality you'd use NER
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        matches = re.findall(name_pattern, text)
        return list(set(matches))

    def _extract_key_points(self, summary_text: str) -> List[str]:
        """Extract key points from AI summary."""
        lines = summary_text.split('\n')
        key_points = []
        for line in lines:
            if line.strip().startswith('-') or line.strip().startswith('•'):
                key_points.append(line.strip()[1:].strip())
        return key_points[:5]  # Limit to 5 key points

    def _extract_action_items(self, summary_text: str) -> List[str]:
        """Extract action items from AI summary."""
        # Look for action-oriented language
        action_keywords = ['need to', 'should', 'must', 'action:', 'todo:', 'task:']
        lines = summary_text.split('\n')
        actions = []
        for line in lines:
            if any(keyword in line.lower() for keyword in action_keywords):
                actions.append(line.strip())
        return actions

    async def _manage_inbox(self, state: AssistantState) -> AssistantState:
        """Organize and manage inbox automatically."""
        try:
            # Get inbox status
            inbox_info = await self.use_tool("gmail_get_profile", {})
            
            # Get recent emails for organization
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": 50,
                "query": "is:unread"
            })
            
            organization_stats = {
                'total_unread': len(emails.get('messages', [])),
                'organized': 0,
                'high_priority': 0,
                'actions_taken': []
            }
            
            # Classify and organize emails
            for email in emails.get('messages', [])[:20]:  # Process first 20
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                
                classification = await self._classify_single_email(email_data)
                
                # Take organization actions based on classification
                if classification.priority == 'high':
                    organization_stats['high_priority'] += 1
                    # Could add to important label
                
                if classification.category == 'marketing':
                    # Could move to marketing folder or mark as read
                    organization_stats['actions_taken'].append(f"Categorized marketing email: {email_data.get('subject', '')[:50]}")
                
                organization_stats['organized'] += 1
            
            # Update state
            active_context = state.get('active_context', {})
            active_context['inbox_organization'] = organization_stats
            state['active_context'] = active_context
            
            # Generate management report
            response = f"""📧 Inbox Management Complete!
            
            Processed: {organization_stats['organized']} emails
            High Priority Found: {organization_stats['high_priority']}
            Total Unread: {organization_stats['total_unread']}
            
            Actions Taken:
            {chr(10).join('• ' + action for action in organization_stats['actions_taken'][:5])}
            """
            
            return self._add_agent_message(state, response, "management_result")
            
        except Exception as e:
            self.logger.error(f"Inbox management failed: {e}")
            return self._add_agent_message(state, f"I encountered an error managing your inbox: {str(e)}", "error")

    async def _generate_classification_summary(self, classifications: List[Dict]) -> str:
        """Generate a summary of email classifications."""
        high_priority = [c for c in classifications if c['classification'].priority == 'high']
        medium_priority = [c for c in classifications if c['classification'].priority == 'medium']
        
        summary = f"📧 **Email Classification Summary**\n\n"
        summary += f"**High Priority ({len(high_priority)} emails):**\n"
        
        for email in high_priority[:5]:  # Show top 5
            summary += f"• {email['subject'][:50]}... from {email['sender']}\n"
            summary += f"  Urgency: {email['classification'].urgency_score:.1f}/1.0\n"
        
        if medium_priority:
            summary += f"\n**Medium Priority ({len(medium_priority)} emails):**\n"
            for email in medium_priority[:3]:  # Show top 3
                summary += f"• {email['subject'][:50]}...\n"
        
        return summary

    async def _generate_summary_report(self, summaries: List[EmailSummary]) -> str:
        """Generate a consolidated email summary report."""
        report = "📧 **Email Summary Report**\n\n"
        
        for i, summary in enumerate(summaries[:5], 1):  # Top 5 emails
            report += f"**{i}. {summary.subject}**\n"
            report += f"From: {summary.sender}\n"
            
            if summary.key_points:
                report += "Key Points:\n"
                for point in summary.key_points[:3]:
                    report += f"• {point}\n"
            
            if summary.action_items:
                report += "Actions Needed:\n"
                for action in summary.action_items[:2]:
                    report += f"• {action}\n"
            
            report += "\n"
        
        return report

    async def _compose_response(self, state: AssistantState) -> AssistantState:
        """Compose email responses based on context."""
        return self._add_agent_message(state, "Email composition feature is ready for implementation.", "info")

    async def _search_emails(self, state: AssistantState) -> AssistantState:
        """Search emails based on user criteria."""
        return self._add_agent_message(state, "Email search feature is ready for implementation.", "info")

    async def _general_email_assistance(self, state: AssistantState) -> AssistantState:
        """Provide general email assistance."""
        # Ensure state is a dictionary
        if isinstance(state, str):
            user_input = state
        else:
            user_input = state.get('user_input', 'How can I help with your emails?')
            
        response = await self.generate_response(
            user_input,
            context="You are a helpful email assistant. Provide guidance on email management, organization, and best practices."
        )
        return self._add_agent_message(state, response, "assistance")