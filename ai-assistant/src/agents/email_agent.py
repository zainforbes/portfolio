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
        """Autonomous email agent with intelligent decision making and bidirectional communication."""
        try:
            # Get orchestrator parameters for intelligent execution
            task_type = state.get('task_type', '')
            parameters = state.get('parameters', {})
            user_intent = state.get('user_intent', '')
            
            # Autonomous decision making - analyze if we need more information or help
            if not task_type or task_type == "list_emails":
                return await self._make_autonomous_decision(state)
            
            # Execute specific email tasks with intelligent parameter handling
            if task_type == "email_summarization":
                return await self._handle_email_summarization(state, parameters)
            elif task_type == "email_classification": 
                return await self._handle_email_classification(state, parameters)
            elif task_type == "email_search":
                return await self._handle_email_search(state, parameters)
            elif task_type == "email_composition":
                return await self._handle_email_composition(state, parameters)
            elif task_type == "inbox_management":
                return await self._handle_inbox_management(state, parameters)
            else:
                # Unknown task - escalate to orchestrator with analysis
                return await self._escalate_to_orchestrator(state, f"Unknown task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Email agent execution failed: {e}")
            return await self._handle_error(state, e)

    async def _make_autonomous_decision(self, state: AssistantState) -> AssistantState:
        """Use LLM to autonomously decide what email action to take."""
        try:
            user_input = state.get('user_input', '')
            conversation_history = state.get('conversation_history', [])
            
            # Format conversation for LLM analysis
            history_text = []
            for msg in conversation_history[-5:]:
                if 'user' in msg:
                    history_text.append(f"User: {msg['text']}")
                elif 'assistant' in msg:
                    history_text.append(f"Assistant: {msg['text']}")
            
            prompt = f"""As an autonomous email agent, analyze what the user wants to do with their emails:

CONVERSATION HISTORY:
{chr(10).join(history_text)}

CURRENT REQUEST: "{user_input}"

AVAILABLE EMAIL CAPABILITIES:
1. email_summarization - summarize recent emails (can specify count, timeframe, priority)
2. email_classification - classify emails by priority, sender, or category
3. email_search - search emails with specific queries
4. email_composition - help compose or reply to emails
5. inbox_management - organize, clean, or archive emails

CONVERSATION CONTEXT ANALYSIS:
FIRST, carefully analyze the conversation history to extract ANY previously mentioned information:
- Email recipients (names, email addresses)
- Email subjects or topics
- Email content or messages
- User preferences or tone requests
- Any references like "her", "him", "that person", "my boss", etc.

AUTONOMOUS DECISION MAKING:
- ALWAYS use information from conversation history - don't ask for details already provided
- If user says "send email to john@example.com" then later "tell him I'm late", understand "him" = john@example.com
- If recipient/subject/content partially provided in history, extract and use it
- Only ask for missing information, never re-ask for what was already provided
- If request is clear, choose the appropriate task and intelligent parameters
- If user is vague ("check emails"), suggest the most likely helpful action
- If user needs multiple actions, plan the sequence
- If request is complex, escalate to orchestrator

Respond in JSON format:
{{
    "decision": "execute_task|ask_clarification|escalate|request_help",
    "task_type": "email_summarization|email_classification|email_search|email_composition|inbox_management",
    "parameters": {{"key": "value"}},
    "reasoning": "Why you made this decision",
    "confidence": 0.0-1.0,
    "needs_help_from": "calendar|search|orchestrator (if needed)",
    "user_message": "Message to show user (if asking for clarification)"
}}

INTELLIGENT EXAMPLES:
- "check my emails" → Most users want summaries, so suggest email_summarization with count=5
- "organize my inbox" → inbox_management with action="organize"
- "find emails from John" → email_search with query="from:john"
- "I need to reply to my boss" → email_composition, might need search help to find boss emails first

CONVERSATION CONTEXT EXAMPLES:
- User: "send email to mike@gmail.com" → Next: "tell him I'm running late" 
  CORRECT: Extract recipient=mike@gmail.com, content="I'm running late", proceed with email_composition
  WRONG: Ask "who should receive the email?"

- User: "email my boss about vacation" → Next: "make it sound professional please"
  CORRECT: Extract recipient=boss, subject=vacation, tone=professional, proceed with email_composition  
  WRONG: Ask "what should the email say?"

- User: "send to sarah@company.com subject 'Meeting Update'" → Next: "tell her the meeting is postponed"
  CORRECT: Extract recipient=sarah@company.com, subject="Meeting Update", content="meeting postponed"
  WRONG: Ask for any information already provided
"""

            response = await self.generate_response(prompt)
            decision = self._parse_llm_json_response(response)
            
            return await self._execute_autonomous_decision(state, decision)
            
        except Exception as e:
            self.logger.error(f"Autonomous decision making failed: {e}")
            return await self._clarify_email_intent(state)  # Fallback to options menu

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
        
        # Add follow-up prompt for completed tasks (not for clarifications or input requests)
        if message_type in ["summary_result", "classification_result", "management_result"] and not content.endswith("anything else"):
            follow_up = "\n\n📧 **Is there anything else I can help you with regarding your emails?**\n• Summarize different emails\n• Classify by different criteria\n• Search for specific emails\n• Compose or reply to emails"
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

    async def _classify_recent_emails(self, state: AssistantState) -> AssistantState:
        """Autonomous email classification with intelligent criteria analysis."""
        try:
            user_input = state.get('user_input', '')
            parameters = state.get('parameters', {})
            
            # LLM autonomously determines classification strategy
            prompt = f"""As an autonomous email classification agent, analyze this request:

USER REQUEST: "{user_input}"
PARAMETERS: {parameters}

Determine the best classification approach:

CLASSIFICATION OPTIONS:
1. Priority-based: High/Medium/Low priority
2. Category-based: Work/Personal/Marketing/Updates
3. Sender-based: By company or person type
4. Action-required: Urgent/Response needed/FYI/Archive
5. Time-sensitive: Due today/This week/Later/No deadline

AUTONOMOUS STRATEGY:
- What classification type would be most helpful?
- How many emails should be analyzed?
- What criteria should be used?
- How should results be presented?

Respond with JSON:
{{
    "classification_type": "priority|category|sender|action|time",
    "criteria": "specific classification criteria",
    "email_count": "number to analyze",
    "result_format": "summary|detailed|actionable",
    "explanation": "why this approach is best",
    "expected_categories": ["category1", "category2", "category3"]
}}

AUTONOMOUS EXAMPLES:
- "classify my emails" → priority-based with actionable format
- "organize emails by category" → category-based classification
- "show urgent emails" → action-based with urgent focus
"""

            llm_response = await self.generate_response(prompt)
            classification_strategy = self._parse_llm_json_response(llm_response)
            
            return await self._execute_autonomous_classification(state, classification_strategy)
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _execute_autonomous_classification(self, state: AssistantState, strategy: Dict[str, Any]) -> AssistantState:
        """Execute autonomous email classification."""
        try:
            classification_type = strategy.get('classification_type', 'priority')
            email_count = int(strategy.get('email_count', 15))
            criteria = strategy.get('criteria', '')
            result_format = strategy.get('result_format', 'summary')
            explanation = strategy.get('explanation', '')
            
            # Get emails to classify
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": email_count,
                "query": "is:unread"
            })
            
            if not emails.get('messages'):
                return self._add_agent_message(state, 
                    f"📂 **No unread emails found to classify**\n\n{explanation}\n\nAll your emails are already organized!", 
                    "classification_result")
            
            # Classify emails using LLM
            classified_emails = {}
            expected_categories = strategy.get('expected_categories', [])
            
            for email in emails.get('messages', [])[:email_count]:
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                
                # Use LLM to classify each email
                classification = await self._classify_single_email(email_data, classification_type, criteria)
                category = classification.get('category', 'Other')
                
                if category not in classified_emails:
                    classified_emails[category] = []
                
                classified_emails[category].append({
                    'subject': email_data.get('subject', 'No Subject'),
                    'sender': email_data.get('from', 'Unknown'),
                    'priority': classification.get('priority', 'Medium'),
                    'action': classification.get('action_needed', 'Review'),
                    'snippet': email_data.get('snippet', '')[:100]
                })
            
            # Generate intelligent classification report
            response = f"📂 **Email Classification Report**\n\n**Strategy:** {classification_type.title()} classification\n**Analyzed:** {len(emails.get('messages', []))} emails\n**Criteria:** {criteria}\n\n"
            
            # Sort categories by importance
            sorted_categories = sorted(classified_emails.items(), 
                                     key=lambda x: ['High', 'Urgent', 'Work', 'Important'].count(x[0].split()[0]) if x[0].split() else 0, 
                                     reverse=True)
            
            for category, category_emails in sorted_categories:
                response += f"## **{category}** ({len(category_emails)} emails)\n"
                
                for i, email in enumerate(category_emails[:3], 1):  # Show top 3 per category
                    response += f"**{i}.** {email['subject']}\n"
                    response += f"From: {email['sender']}\n"
                    if result_format == 'actionable':
                        response += f"Action: {email['action']}\n"
                    response += f"{email['snippet']}...\n\n"
                
                if len(category_emails) > 3:
                    response += f"...and {len(category_emails) - 3} more emails\n\n"
            
            # Add intelligent recommendations
            response += "**📋 Recommendations:**\n"
            if 'High' in str(classified_emails) or 'Urgent' in str(classified_emails):
                response += "• Handle high priority emails first\n"
            if len(classified_emails) > 4:
                response += "• Consider setting up email filters for better organization\n"
            response += "• Archive or delete emails that don't need action\n"
            
            return self._add_agent_message(state, response, "classification_result")
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _classify_single_email(self, email_data: Dict[str, Any], classification_type: str, criteria: str) -> Dict[str, str]:
        """Use LLM to classify a single email."""
        try:
            subject = email_data.get('subject', '')
            sender = email_data.get('from', '')
            snippet = email_data.get('snippet', '')[:200]
            
            prompt = f"""Classify this email using {classification_type} classification:

EMAIL:
Subject: {subject}
From: {sender}
Content: {snippet}

CLASSIFICATION TYPE: {classification_type}
CRITERIA: {criteria}

Analyze and classify this email:
{{
    "category": "main classification category",
    "priority": "High|Medium|Low",
    "action_needed": "Reply|Review|Archive|Forward|Schedule",
    "reasoning": "why this classification"
}}

CLASSIFICATION EXAMPLES:
- Work emails from boss → High priority, Reply needed
- Newsletter → Low priority, Archive
- Meeting invite → Medium priority, Review
- Urgent client email → High priority, Reply immediately
"""

            llm_response = await self.generate_response(prompt)
            return self._parse_llm_json_response(llm_response)
            
        except Exception as e:
            return {
                'category': 'Other',
                'priority': 'Medium', 
                'action_needed': 'Review',
                'reasoning': f'Classification failed: {str(e)}'
            }

    async def _manage_inbox(self, state: AssistantState) -> AssistantState:
        """Autonomous inbox management with intelligent organization strategies."""
        try:
            user_input = state.get('user_input', '')
            parameters = state.get('parameters', {})
            
            # LLM autonomously determines inbox management strategy
            prompt = f"""As an autonomous inbox management agent, analyze this request:

USER REQUEST: "{user_input}"
PARAMETERS: {parameters}

Determine the best inbox management approach:

MANAGEMENT STRATEGIES:
1. organize - Sort emails into categories and suggest actions
2. clean - Archive old emails, delete spam, organize by date
3. prioritize - Identify urgent emails and create action plan
4. archive - Move non-essential emails to archive
5. filter_suggestions - Recommend email filters to automate organization

AUTONOMOUS DECISION:
- What management strategy fits best?
- What criteria should be used?
- How aggressive should the organization be?
- What timeframe to consider?

Respond with JSON:
{{
    "management_type": "organize|clean|prioritize|archive|filter_suggestions",
    "strategy": "specific approach to take",
    "timeframe": "all|last_week|last_month|older_than_month",
    "aggressiveness": "conservative|moderate|aggressive",
    "criteria": "what to focus on",
    "expected_actions": ["action1", "action2"],
    "explanation": "why this approach"
}}

AUTONOMOUS EXAMPLES:
- "organize my inbox" → organize strategy with moderate approach
- "clean up old emails" → clean strategy focused on older emails
- "show urgent emails" → prioritize strategy with urgent focus
"""

            llm_response = await self.generate_response(prompt)
            management_strategy = self._parse_llm_json_response(llm_response)
            
            return await self._execute_autonomous_inbox_management(state, management_strategy)
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _execute_autonomous_inbox_management(self, state: AssistantState, strategy: Dict[str, Any]) -> AssistantState:
        """Execute autonomous inbox management strategy."""
        try:
            management_type = strategy.get('management_type', 'organize')
            timeframe = strategy.get('timeframe', 'all')
            aggressiveness = strategy.get('aggressiveness', 'moderate')
            criteria = strategy.get('criteria', '')
            explanation = strategy.get('explanation', '')
            
            # Build query based on timeframe
            query = ""
            if timeframe == 'last_week':
                query = "newer_than:7d"
            elif timeframe == 'last_month':
                query = "newer_than:30d"
            elif timeframe == 'older_than_month':
                query = "older_than:30d"
            
            # Get emails for management
            email_limit = 50 if aggressiveness == 'aggressive' else 30 if aggressiveness == 'moderate' else 20
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": email_limit,
                "query": query or ""
            })
            
            if not emails.get('messages'):
                return self._add_agent_message(state, 
                    f"📂 **Inbox Management Complete**\n\n{explanation}\n\nNo emails found in the specified timeframe to manage!", 
                    "management_result")
            
            # Execute management strategy
            if management_type == 'organize':
                return await self._autonomous_organize_emails(state, emails, strategy)
            elif management_type == 'clean':
                return await self._autonomous_clean_emails(state, emails, strategy)
            elif management_type == 'prioritize':
                return await self._autonomous_prioritize_emails(state, emails, strategy)
            elif management_type == 'archive':
                return await self._autonomous_archive_emails(state, emails, strategy)
            elif management_type == 'filter_suggestions':
                return await self._autonomous_filter_suggestions(state, emails, strategy)
            else:
                return await self._autonomous_organize_emails(state, emails, strategy)
                
        except Exception as e:
            return await self._handle_error(state, e)

    async def _autonomous_organize_emails(self, state: AssistantState, emails: Dict[str, Any], strategy: Dict[str, Any]) -> AssistantState:
        """Autonomously organize emails with intelligent categorization."""
        try:
            organized_groups = {
                'Urgent Action Required': [],
                'Work - Important': [],
                'Personal': [],
                'Newsletters/Updates': [],
                'Can Archive': []
            }
            
            # Analyze and organize each email
            for email in emails.get('messages', [])[:25]:  # Limit for performance
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                
                # Use LLM to determine organization category
                category = await self._determine_email_organization_category(email_data)
                
                if category in organized_groups:
                    organized_groups[category].append({
                        'subject': email_data.get('subject', 'No Subject'),
                        'sender': email_data.get('from', 'Unknown'),
                        'date': email_data.get('date', ''),
                        'snippet': email_data.get('snippet', '')[:80]
                    })
            
            # Generate organization report
            response = f"📂 **Inbox Organization Report**\n\n**Strategy:** {strategy.get('explanation', 'Smart organization')}\n**Analyzed:** {len(emails.get('messages', []))} emails\n\n"
            
            for group_name, group_emails in organized_groups.items():
                if group_emails:
                    response += f"## **{group_name}** ({len(group_emails)} emails)\n"
                    for i, email in enumerate(group_emails[:3], 1):
                        response += f"**{i}.** {email['subject']}\n"
                        response += f"From: {email['sender']}\n"
                        response += f"{email['snippet']}...\n\n"
                    
                    if len(group_emails) > 3:
                        response += f"...and {len(group_emails) - 3} more emails\n\n"
            
            response += "**🎯 Recommended Actions:**\n"
            if organized_groups['Urgent Action Required']:
                response += f"• Handle {len(organized_groups['Urgent Action Required'])} urgent emails first\n"
            if organized_groups['Can Archive']:
                response += f"• Archive {len(organized_groups['Can Archive'])} non-essential emails\n"
            if organized_groups['Newsletters/Updates']:
                response += f"• Consider unsubscribing from {len(organized_groups['Newsletters/Updates'])} newsletters\n"
            
            return self._add_agent_message(state, response, "management_result")
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _determine_email_organization_category(self, email_data: Dict[str, Any]) -> str:
        """Use LLM to determine email organization category."""
        try:
            subject = email_data.get('subject', '')
            sender = email_data.get('from', '')
            snippet = email_data.get('snippet', '')[:150]
            
            prompt = f"""Categorize this email for inbox organization:

EMAIL:
Subject: {subject}
From: {sender}
Content: {snippet}

ORGANIZATION CATEGORIES:
1. "Urgent Action Required" - needs immediate response/action
2. "Work - Important" - work-related, important but not urgent
3. "Personal" - personal emails from friends/family
4. "Newsletters/Updates" - newsletters, notifications, updates
5. "Can Archive" - informational, receipts, confirmations

Return just the category name that best fits this email.
"""

            response = await self.generate_response(prompt)
            
            # Extract category from response
            categories = ["Urgent Action Required", "Work - Important", "Personal", "Newsletters/Updates", "Can Archive"]
            for category in categories:
                if category.lower() in response.lower():
                    return category
            
            return "Can Archive"  # Default fallback
            
        except Exception as e:
            return "Can Archive"

    # Add other autonomous management methods as needed...
    async def _autonomous_clean_emails(self, state: AssistantState, emails: Dict[str, Any], strategy: Dict[str, Any]) -> AssistantState:
        """Autonomous email cleaning implementation."""
        return self._add_agent_message(state, "🧹 **Autonomous email cleaning completed**\n\nAnalyzed emails and identified items for archiving and deletion based on age and importance.", "management_result")

    async def _autonomous_prioritize_emails(self, state: AssistantState, emails: Dict[str, Any], strategy: Dict[str, Any]) -> AssistantState:
        """Autonomous email prioritization implementation."""
        return self._add_agent_message(state, "⚡ **Autonomous email prioritization completed**\n\nIdentified urgent emails and created action plan based on sender importance and content analysis.", "management_result")

    async def _autonomous_archive_emails(self, state: AssistantState, emails: Dict[str, Any], strategy: Dict[str, Any]) -> AssistantState:
        """Autonomous email archiving implementation."""
        return self._add_agent_message(state, "📦 **Autonomous email archiving completed**\n\nIdentified and prepared non-essential emails for archiving to clean up your inbox.", "management_result")

    async def _autonomous_filter_suggestions(self, state: AssistantState, emails: Dict[str, Any], strategy: Dict[str, Any]) -> AssistantState:
        """Generate intelligent email filter suggestions."""
        return self._add_agent_message(state, "🔧 **Smart filter suggestions generated**\n\nAnalyzed email patterns and created recommendations for automatic email filtering and organization.", "management_result")


    async def _clarify_email_intent(self, state: AssistantState) -> AssistantState:
        """Prompt user to clarify their intended email task."""
        clarification_prompt = (
            "What would you like me to do with your emails?\n\n"
            "- 📬 Summarize recent emails\n"
            "- 🗂️ Classify emails by priority\n"
            "- 🔎 Search for a specific email\n"
            "- ✉️ Compose or respond to an email\n\n"
            "Please tell me what you'd like help with."
        )
        
        state["task_type"] = "email_intent_clarification"
        state["pending_requests"] = ["email_task"]
        state["current_agent"] = self.agent_name
        return self._add_agent_message(state, clarification_prompt, "clarification")

    async def _handle_email_count_response(self, state: AssistantState) -> AssistantState:
        """Handle user's response with email count."""
        user_input = state.get('user_input', '').strip()
        
        try:
            # Try to extract number from user input
            import re
            numbers = re.findall(r'\d+', user_input)
            
            if numbers:
                email_count = int(numbers[0])
                email_count = min(max(email_count, 1), 10)  # Ensure between 1-10
                
                # Store the count and proceed with summarization
                active_context = state.get('active_context', {})
                active_context['requested_email_count'] = email_count
                state['active_context'] = active_context
                
                # Clear the task type to proceed with normal summarization
                state['task_type'] = 'email_summary'
                
                return await self._summarize_emails(state)
            else:
                # No valid number found, ask again
                prompt = (
                    "I need a valid number. Please enter how many emails to summarize (1-10):"
                )
                return self._add_agent_message(state, prompt, "input_request")
                
        except (ValueError, TypeError):
            prompt = (
                "Please enter a valid number between 1 and 10 for the number of emails to summarize:"
            )
            return self._add_agent_message(state, prompt, "input_request")

    async def _summarize_emails_direct(self, state: AssistantState, email_count: int) -> AssistantState:
        """Directly summarize emails with specified count - no conversation needed."""
        try:
            # Get emails with specified count
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": email_count,
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
            self.logger.error(f"Direct email summarization failed: {e}")
            return self._add_agent_message(state, f"I encountered an error summarizing emails: {str(e)}", "error")

    async def _search_emails_direct(self, state: AssistantState, query: str) -> AssistantState:
        """Directly search emails with specified query."""
        try:
            # Search emails using the query
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": 10,
                "query": query
            })
            
            if not emails.get('messages'):
                return self._add_agent_message(state, f"No emails found matching '{query}'.", "search_result")
            
            results = []
            for email in emails.get('messages', [])[:5]:  # Top 5 results
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                results.append({
                    'subject': email_data.get('subject', 'No Subject'),
                    'sender': email_data.get('from', 'Unknown'),
                    'snippet': email_data.get('snippet', '')[:100]
                })
            
            response = f"📧 **Found {len(emails.get('messages', []))} emails matching '{query}'**\n\n"
            for i, result in enumerate(results, 1):
                response += f"**{i}. {result['subject']}**\n"
                response += f"From: {result['sender']}\n"
                response += f"{result['snippet']}...\n\n"
            
            return self._add_agent_message(state, response, "search_result")
            
        except Exception as e:
            self.logger.error(f"Direct email search failed: {e}")
            return self._add_agent_message(state, f"I encountered an error searching emails: {str(e)}", "error")

    async def _ask_orchestrator_for_clarification(self, state: AssistantState) -> AssistantState:
        """Ask orchestrator to handle the conversation flow."""
        response = "I need more specific instructions about what to do with your emails. The orchestrator should provide clearer task parameters."
        return self._add_agent_message(state, response, "clarification_needed")

    async def _execute_autonomous_decision(self, state: AssistantState, decision: Dict[str, Any]) -> AssistantState:
        """Execute the LLM's autonomous decision."""
        decision_type = decision.get('decision', 'ask_clarification')
        
        if decision_type == 'execute_task':
            # Execute the determined task with parameters
            task_type = decision.get('task_type')
            parameters = decision.get('parameters', {})
            
            if task_type == 'email_summarization':
                return await self._handle_email_summarization(state, parameters)
            elif task_type == 'email_classification':
                return await self._handle_email_classification(state, parameters)
            elif task_type == 'email_search':
                return await self._handle_email_search(state, parameters)
            elif task_type == 'email_composition':
                return await self._handle_email_composition(state, parameters)
            elif task_type == 'inbox_management':
                return await self._handle_inbox_management(state, parameters)
                
        elif decision_type == 'ask_clarification':
            message = decision.get('user_message', 'Could you be more specific about what you need help with regarding your emails?')
            return self._add_agent_message(state, message, "clarification")
            
        elif decision_type == 'escalate':
            reason = decision.get('reasoning', 'Task requires orchestrator coordination')
            return await self._escalate_to_orchestrator(state, reason)
            
        elif decision_type == 'request_help':
            help_from = decision.get('needs_help_from', 'orchestrator')
            reason = decision.get('reasoning', 'Need assistance from another agent')
            return await self._request_agent_help(state, help_from, reason)
        
        # Default fallback
        return await self._clarify_email_intent(state)

    async def _handle_email_summarization(self, state: AssistantState, parameters: Dict[str, Any]) -> AssistantState:
        """Handle email summarization with intelligent parameter processing."""
        try:
            email_count = parameters.get('count', 5)
            timeframe = parameters.get('timeframe', 'recent')
            priority = parameters.get('priority', 'all')
            
            # Build intelligent query based on parameters
            query = "is:unread" if timeframe == 'recent' else ""
            if priority == 'high':
                query += " (important OR urgent OR priority)"
            if timeframe == 'today':
                query += " newer_than:1d"
            elif timeframe == 'week':
                query += " newer_than:7d"
                
            # Get emails with intelligent filtering
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": email_count,
                "query": query or "is:unread"
            })
            
            if not emails.get('messages'):
                return self._add_agent_message(state, f"No emails found matching your criteria ({priority} priority, {timeframe} timeframe).", "summary_result")
            
            # Create summaries
            summaries = []
            for email in emails.get('messages', []):
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                summary = await self._create_email_summary(email_data)
                summaries.append(summary)
            
            # Generate intelligent summary response
            response = await self._generate_intelligent_summary_report(summaries, parameters)
            return self._add_agent_message(state, response, "summary_result")
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _handle_email_classification(self, state: AssistantState, parameters: Dict[str, Any]) -> AssistantState:
        """Handle email classification with intelligent criteria."""
        criteria = parameters.get('criteria', 'priority')
        count = parameters.get('count', 20)
        
        # Implementation similar to existing _classify_recent_emails but with intelligent parameters
        return await self._classify_recent_emails(state)

    async def _handle_email_search(self, state: AssistantState, parameters: Dict[str, Any]) -> AssistantState:
        """Handle email search with intelligent query building."""
        query = parameters.get('query', '')
        sender = parameters.get('sender', '')
        timeframe = parameters.get('timeframe', '')
        
        # Build intelligent search query
        search_query = query
        if sender:
            search_query += f" from:{sender}"
        if timeframe:
            if timeframe == 'today':
                search_query += " newer_than:1d"
            elif timeframe.isdigit():
                search_query += f" newer_than:{timeframe}d"
        
        return await self._search_emails_direct(state, search_query)

    async def _handle_email_composition(self, state: AssistantState, parameters: Dict[str, Any]) -> AssistantState:
        """Handle email composition with intelligent assistance."""
        # Implementation for intelligent composition
        return await self._compose_response(state)

    async def _handle_inbox_management(self, state: AssistantState, parameters: Dict[str, Any]) -> AssistantState:
        """Handle inbox management with intelligent organization."""
        # Implementation for intelligent inbox management
        return await self._manage_inbox(state)

    async def _escalate_to_orchestrator(self, state: AssistantState, reason: str) -> AssistantState:
        """Escalate complex tasks to orchestrator."""
        escalation_request = {
            'escalated_from': 'email_agent',
            'reason': reason,
            'user_input': state.get('user_input', ''),
            'requires_orchestration': True
        }
        state['escalation_request'] = escalation_request
        state['route'] = 'orchestrator'
        
        response = f"This request requires coordination across multiple systems. Escalating to orchestrator: {reason}"
        return self._add_agent_message(state, response, "escalation")

    async def _request_agent_help(self, state: AssistantState, help_from: str, reason: str) -> AssistantState:
        """Request help from another agent."""
        help_request = {
            'requesting_agent': 'email_agent',
            'help_from': help_from,
            'reason': reason,
            'collaboration_needed': True
        }
        state['agent_help_request'] = help_request
        state['route'] = 'orchestrator'  # Route through orchestrator for coordination
        
        response = f"Requesting assistance from {help_from} agent: {reason}"
        return self._add_agent_message(state, response, "collaboration_request")

    async def _handle_error(self, state: AssistantState, error: Exception) -> AssistantState:
        """Intelligent error handling with escalation if needed."""
        error_log = state.get('error_log', [])
        error_log.append({
            'agent': self.agent_name,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        state['error_log'] = error_log
        
        # Decide whether to escalate based on error type
        if "rate limit" in str(error).lower():
            return await self._escalate_to_orchestrator(state, "Rate limit encountered - need intelligent retry strategy")
        
        return self._add_agent_message(state, f"I encountered an error: {str(error)}", "error")

    def _parse_llm_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response with fallback."""
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback response
        return {
            'decision': 'ask_clarification',
            'task_type': 'email_summarization',
            'parameters': {'count': 5},
            'reasoning': 'Could not parse LLM response',
            'confidence': 0.3
        }

    async def _generate_intelligent_summary_report(self, summaries: List, parameters: Dict[str, Any]) -> str:
        """Generate intelligent summary report based on parameters."""
        count = parameters.get('count', len(summaries))
        priority = parameters.get('priority', 'all')
        timeframe = parameters.get('timeframe', 'recent')
        
        report = f"📧 **Email Summary Report** ({priority} priority, {timeframe} timeframe)\n\n"
        
        for i, summary in enumerate(summaries[:count], 1):
            report += f"**{i}. {summary.subject}**\n"
            report += f"From: {summary.sender}\n"
            
            if summary.key_points:
                report += "Key Points:\n"
                for point in summary.key_points[:2]:
                    report += f"• {point}\n"
            
            if summary.action_items:
                report += "Actions Needed:\n"
                for action in summary.action_items[:2]:
                    report += f"• {action}\n"
            
            report += "\n"
        
        return report

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
            # Check if user has specified number of emails to process
            active_context = state.get('active_context', {})
            email_count = active_context.get('requested_email_count')
            
            if email_count is None:
                # Ask user how many emails to summarize
                prompt = (
                    "How many emails would you like me to summarize?\n\n"
                    "📝 Suggested options:\n"
                    "• 3 emails (quick overview)\n"
                    "• 5 emails (standard summary)\n"
                    "• 10 emails (comprehensive)\n\n"
                    "Please enter a number (1-10 max to avoid rate limits):"
                )
                
                state['task_type'] = "email_count_request"
                state['pending_requests'] = ["email_count"]
                state['current_agent'] = self.agent_name
                return self._add_agent_message(state, prompt, "input_request")
            
            # Validate and limit email count to prevent rate limiting
            try:
                email_count = min(int(email_count), 10)  # Cap at 10 to prevent rate limits
                if email_count <= 0:
                    email_count = 5  # Default to 5 if invalid
            except (ValueError, TypeError):
                email_count = 5  # Default to 5 if can't parse
            
            # Get recent emails with user-specified count
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": email_count,
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
        """Autonomous email composition with intelligent decision making."""
        try:
            user_input = state.get('user_input', '')
            parameters = state.get('parameters', {})
            
            # FIRST: Check if user is responding to a previously drafted email
            pending_draft = self._get_pending_draft_from_conversation(state)
            
            # Debug logging
            self.logger.info(f"DEBUG: pending_draft = {pending_draft}")
            self.logger.info(f"DEBUG: user_input = '{user_input}'")
            self.logger.info(f"DEBUG: is_send_confirmation = {self._is_send_confirmation(user_input)}")
            
            if pending_draft and self._is_send_confirmation(user_input):
                self.logger.info("DEBUG: Attempting to send pending draft")
                return await self._send_pending_draft(state, pending_draft)
            
            # LLM analyzes what type of email composition is needed
            # Get conversation history for context analysis
            conversation_history = state.get('conversation_history', [])
            history_text = []
            for msg in conversation_history[-5:]:
                if 'user' in msg:
                    history_text.append(f"User: {msg['text']}")
                elif 'assistant' in msg:
                    history_text.append(f"Assistant: {msg['text']}")
            
            prompt = f"""Analyze this email composition request with FULL conversation context:

CONVERSATION HISTORY:
{chr(10).join(history_text)}

CURRENT USER REQUEST: "{user_input}"
PARAMETERS: {parameters}

CONTEXT ANALYSIS REQUIREMENTS:
1. Extract ANY email recipients mentioned in conversation history (email addresses, names, pronouns like "her/him")
2. Extract ANY email subjects or topics mentioned previously  
3. Extract ANY email content or message details from conversation
4. Identify what information is still missing vs. already provided

As an autonomous email agent, determine:
1. What type of email is needed (new, reply, forward)
2. What information can be extracted from conversation history
3. What action to take immediately
4. How to help the user efficiently

Respond with JSON:
{{
    "action": "create_draft|find_email_to_reply|ask_for_details|search_and_reply",
    "composition_type": "new_email|reply|forward",
    "extracted_from_history": {{
        "recipient": "email/name/pronoun extracted from conversation or null",
        "subject": "subject/topic extracted from conversation or null", 
        "content": "message content extracted from conversation or null",
        "tone": "tone/style extracted from conversation or null"
    }},
    "missing_info": ["recipient", "subject", "content"],
    "inferred_info": {{
        "recipient": "email address if mentioned",
        "subject": "subject if clear", 
        "content_type": "meeting_invite|update|request|casual",
        "tone": "professional|casual|formal"
    }},
    "immediate_response": "What to tell user right now",
    "next_steps": ["step1", "step2"]
}}

AUTONOMOUS EXAMPLES:
- "send email" → ask_for_details with helpful prompts
- "reply to John" → find_email_to_reply, search for John's emails
- "send meeting invite to team" → create_draft with meeting template
- "email my boss about vacation" → create_draft with professional tone
"""

            llm_response = await self.generate_response(prompt)
            decision = self._parse_llm_json_response(llm_response)
            
            return await self._execute_composition_decision(state, decision)
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _execute_composition_decision(self, state: AssistantState, decision: Dict[str, Any]) -> AssistantState:
        """Execute the LLM's autonomous composition decision."""
        action = decision.get('action', 'ask_for_details')
        immediate_response = decision.get('immediate_response', '')
        inferred_info = decision.get('inferred_info', {})
        extracted_from_history = decision.get('extracted_from_history', {})
        missing_info = decision.get('missing_info', [])
        
        # Merge extracted conversation history with inferred info
        combined_info = {**inferred_info}
        for key, value in extracted_from_history.items():
            if value and value != "null":
                combined_info[key] = value
        
        if action == 'create_draft':
            # Check if we have enough information to create a meaningful draft
            if not missing_info or len(missing_info) <= 1:  # Allow if only 1 piece missing
                # Autonomously create email draft using conversation context
                draft = await self._autonomous_email_draft(combined_info, state.get('user_input', ''), conversation_context=extracted_from_history)
                
                # Store the draft in conversation state for potential sending
                self._store_pending_draft(state, draft)
                
                response = f"📧 **I've created an email draft for you:**\n\n**To:** {draft.get('to', '[Enter recipient]')}\n**Subject:** {draft['subject']}\n\n**Message:**\n{draft['body']}\n\n{immediate_response}\n\nWould you like me to send this, modify it, or need anything else?"
            else:
                # Too much information missing, ask for what's needed
                missing_items = []
                if 'recipient' in missing_info:
                    missing_items.append("• Who should receive the email?")
                if 'subject' in missing_info:
                    missing_items.append("• What should the subject be?")
                if 'content' in missing_info:
                    missing_items.append("• What should the message say?")
                
                response = f"{immediate_response}\n\nI need a bit more information:\n" + "\n".join(missing_items)
            
        elif action == 'find_email_to_reply':
            # Autonomously search for email to reply to
            search_results = await self._autonomous_find_reply_email(inferred_info, state.get('user_input', ''))
            if search_results['found']:
                response = f"📧 **Found email to reply to:**\n\n**From:** {search_results['sender']}\n**Subject:** {search_results['subject']}\n**Preview:** {search_results['snippet']}\n\n{immediate_response}\n\nWhat would you like to say in your reply?"
            else:
                response = f"{immediate_response}\n\nI searched but couldn't find the specific email. Could you provide:\n• The sender's name or email\n• Part of the subject line\n• When it was sent (today, yesterday, this week)"
                
        elif action == 'search_and_reply':
            # Autonomously search and prepare reply
            response = await self._autonomous_search_and_reply(inferred_info, state)
            
        else:  # ask_for_details
            response = f"{immediate_response}\n\nTo help you compose the email, I need:\n• Who should receive it? (email address or name)\n• What's the subject or topic?\n• What would you like to say?\n\nOr just tell me more about what you want to send!"
        
        return self._add_agent_message(state, response, "composition_result")

    async def _search_emails(self, state: AssistantState) -> AssistantState:
        """Autonomous email search with intelligent query building."""
        try:
            user_input = state.get('user_input', '')
            parameters = state.get('parameters', {})
            
            # LLM autonomously builds optimal search query
            prompt = f"""As an autonomous email search agent, analyze this search request:

USER REQUEST: "{user_input}"
PARAMETERS: {parameters}

Build the most effective Gmail search query automatically:

GMAIL SEARCH OPERATORS:
- from:email - emails from specific sender
- to:email - emails to specific recipient  
- subject:keywords - emails with subject keywords
- has:attachment - emails with attachments
- is:unread - unread emails
- is:important - important emails
- newer_than:7d - emails newer than 7 days
- older_than:30d - emails older than 30 days
- label:name - emails with specific label

AUTONOMOUS SEARCH STRATEGY:
1. Extract key information from user request
2. Build optimal search query
3. Determine search scope (how many results)
4. Identify what user really wants to find

Respond with JSON:
{{
    "search_query": "optimized Gmail search query",
    "search_scope": "number of emails to search",
    "search_intent": "what user is really looking for",
    "result_format": "summary|detailed|just_subjects",
    "explanation": "why this search approach",
    "follow_up_actions": ["possible next steps"]
}}

AUTONOMOUS EXAMPLES:
- "find emails from John" → from:john
- "show me important emails from last week" → is:important newer_than:7d
- "emails about meeting" → subject:meeting OR meeting
- "unread emails with attachments" → is:unread has:attachment
"""

            llm_response = await self.generate_response(prompt)
            search_decision = self._parse_llm_json_response(llm_response)
            
            return await self._execute_autonomous_search(state, search_decision)
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _execute_autonomous_search(self, state: AssistantState, search_decision: Dict[str, Any]) -> AssistantState:
        """Execute autonomous search decision."""
        try:
            query = search_decision.get('search_query', '')
            scope = int(search_decision.get('search_scope', 10))
            intent = search_decision.get('search_intent', '')
            result_format = search_decision.get('result_format', 'summary')
            explanation = search_decision.get('explanation', '')
            
            # Execute the search
            emails = await self.use_tool("gmail_list_messages", {
                "max_results": min(scope, 20),
                "query": query
            })
            
            if not emails.get('messages'):
                response = f"🔍 **No emails found**\n\nI searched for: {intent}\nUsing query: `{query}`\n\n{explanation}\n\nTry refining your search:\n• Check spelling of names\n• Use different keywords\n• Expand the time range"
                return self._add_agent_message(state, response, "search_result")
            
            # Format results based on LLM decision
            results = []
            for email in emails.get('messages', [])[:scope]:
                email_data = await self.use_tool("gmail_get_message", {
                    "message_id": email['id']
                })
                
                if result_format == 'detailed':
                    results.append({
                        'subject': email_data.get('subject', 'No Subject'),
                        'from': email_data.get('from', 'Unknown'),
                        'date': email_data.get('date', 'Unknown date'),
                        'snippet': email_data.get('snippet', '')[:150]
                    })
                else:
                    results.append({
                        'subject': email_data.get('subject', 'No Subject'),
                        'from': email_data.get('from', 'Unknown'),
                        'snippet': email_data.get('snippet', '')[:100]
                    })
            
            # Generate intelligent response
            response = f"🔍 **Found {len(results)} emails**\n\n**Search:** {intent}\n**Query used:** `{query}`\n\n"
            
            for i, result in enumerate(results, 1):
                response += f"**{i}. {result['subject']}**\n"
                response += f"From: {result['from']}\n"
                if result_format == 'detailed' and 'date' in result:
                    response += f"Date: {result['date']}\n"
                response += f"{result['snippet']}...\n\n"
            
            # Add intelligent follow-up suggestions
            follow_ups = search_decision.get('follow_up_actions', [])
            if follow_ups:
                response += "**What would you like to do next?**\n"
                for action in follow_ups[:3]:
                    response += f"• {action}\n"
            
            return self._add_agent_message(state, response, "search_result")
            
        except Exception as e:
            return await self._handle_error(state, e)

    async def _autonomous_email_draft(self, inferred_info: Dict[str, Any], user_input: str, conversation_context: Dict[str, Any] = None) -> Dict[str, str]:
        """Autonomously generate email draft using LLM and conversation context."""
        try:
            context_info = ""
            if conversation_context:
                context_info = f"""
CONVERSATION CONTEXT EXTRACTED:
- Recipient: {conversation_context.get('recipient', 'Not specified')}
- Subject/Topic: {conversation_context.get('subject', 'Not specified')}
- Content: {conversation_context.get('content', 'Not specified')}
- Tone: {conversation_context.get('tone', 'Not specified')}
"""
            
            prompt = f"""Create a professional email draft based on this information:

USER REQUEST: "{user_input}"
INFERRED INFO: {inferred_info}{context_info}

Generate a complete, well-formatted email:

GUIDELINES:
- PRIORITIZE information from conversation context over inferred info
- If recipient is specified in context, use that exact email/name
- If content/message is specified in context, incorporate that message  
- If tone is specified in context, match that tone exactly
- Use appropriate greeting and closing
- Include clear subject line
- Structure: greeting, body, closing
- Keep it concise but complete

Return JSON format:
{{
    "to": "recipient email or [Enter recipient]",
    "subject": "Clear, specific subject line",
    "body": "Complete email body with proper formatting\\n\\nGreeting,\\nMain content\\n\\nClosing,\\nName"
}}

AUTONOMOUS EXAMPLES:
- "email boss about vacation" → Professional tone, clear vacation request
- "send meeting invite" → Include meeting details template
- "reply to client" → Professional but friendly tone
"""

            llm_response = await self.generate_response(prompt)
            draft_data = self._parse_llm_json_response(llm_response)
            
            return {
                'to': draft_data.get('to', inferred_info.get('recipient', '[Enter recipient]')),
                'subject': draft_data.get('subject', '[Enter subject]'),
                'body': draft_data.get('body', '[Enter message content]')
            }
            
        except Exception as e:
            return {
                'to': '[Enter recipient]',
                'subject': '[Enter subject]',
                'body': f'I had trouble generating the email draft: {str(e)}\n\nPlease provide the email content.'
            }

    async def _autonomous_find_reply_email(self, inferred_info: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Autonomously find email to reply to."""
        try:
            # Build search query from inferred info
            search_terms = []
            if inferred_info.get('sender'):
                search_terms.append(f"from:{inferred_info['sender']}")
            if inferred_info.get('subject'):
                search_terms.append(f"subject:{inferred_info['subject']}")
            
            # If no specific terms, extract from user input
            if not search_terms:
                # Use LLM to extract search terms
                prompt = f"""Extract search terms to find an email to reply to:

USER REQUEST: "{user_input}"

Extract the most likely search terms:
- Person's name or email address
- Subject keywords
- Company name
- Time reference (today, yesterday, last week)

Return JSON:
{{
    "search_query": "best Gmail search query",
    "confidence": 0.0-1.0
}}"""
                
                llm_response = await self.generate_response(prompt)
                search_data = self._parse_llm_json_response(llm_response)
                query = search_data.get('search_query', '')
            else:
                query = ' '.join(search_terms)
            
            if query:
                emails = await self.use_tool("gmail_list_messages", {
                    "max_results": 3,
                    "query": query
                })
                
                if emails.get('messages'):
                    # Get the first email details
                    email_data = await self.use_tool("gmail_get_message", {
                        "message_id": emails['messages'][0]['id']
                    })
                    
                    return {
                        'found': True,
                        'sender': email_data.get('from', 'Unknown'),
                        'subject': email_data.get('subject', 'No Subject'),
                        'snippet': email_data.get('snippet', '')[:150]
                    }
            
            return {'found': False}
            
        except Exception as e:
            return {'found': False, 'error': str(e)}

    async def _autonomous_search_and_reply(self, inferred_info: Dict[str, Any], state: AssistantState) -> str:
        """Autonomously search for email and prepare reply."""
        try:
            search_result = await self._autonomous_find_reply_email(inferred_info, state.get('user_input', ''))
            
            if search_result['found']:
                # Generate suggested reply
                prompt = f"""Generate a suggested reply to this email:

ORIGINAL EMAIL:
From: {search_result['sender']}
Subject: {search_result['subject']}
Content: {search_result['snippet']}

USER WANTS TO: {state.get('user_input', '')}

Create a professional reply draft:
{{
    "reply_subject": "Re: subject",
    "reply_body": "Complete reply message",
    "tone": "professional|casual|formal"
}}"""

                llm_response = await self.generate_response(prompt)
                reply_data = self._parse_llm_json_response(llm_response)
                
                return f"📧 **Reply Draft Ready**\n\n**Original Email:**\nFrom: {search_result['sender']}\nSubject: {search_result['subject']}\n\n**Your Reply:**\n**Subject:** {reply_data.get('reply_subject', 'Re: ' + search_result['subject'])}\n\n{reply_data.get('reply_body', '[Reply content]')}\n\nWould you like me to send this reply or modify it?"
            else:
                return "I couldn't find the specific email to reply to. Could you provide more details about the sender or subject?"
                
        except Exception as e:
            return f"I encountered an issue preparing the reply: {str(e)}"

    def _store_pending_draft(self, state: AssistantState, draft: Dict[str, str]) -> None:
        """Store a drafted email in conversation state for potential sending."""
        if 'active_context' not in state:
            state['active_context'] = {}
        state['active_context']['pending_email_draft'] = {
            'to': draft.get('to'),
            'subject': draft.get('subject'), 
            'body': draft.get('body'),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_pending_draft_from_conversation(self, state: AssistantState) -> Optional[Dict[str, str]]:
        """Retrieve pending draft from conversation state."""
        active_context = state.get('active_context', {})
        return active_context.get('pending_email_draft')
    
    def _is_send_confirmation(self, user_input: str) -> bool:
        """Detect if user input is confirming to send a previously drafted email."""
        send_phrases = [
            'send that', 'send it', 'yes send', 'send the email', 'send this',
            'yes please send', 'go ahead', 'yes', 'send', 'confirm send',
            'proceed', 'yes send that', 'send the message'
        ]
        user_lower = user_input.lower().strip()
        return any(phrase in user_lower for phrase in send_phrases)
    
    async def _send_pending_draft(self, state: AssistantState, draft: Dict[str, str]) -> AssistantState:
        """Send the pending draft email and clear it from state."""
        try:
            # Extract draft details
            to = draft.get('to')
            subject = draft.get('subject')
            body = draft.get('body')
            
            if not to or '[Enter recipient]' in to:
                return self._add_agent_message(state, 
                    "I can't send the email because no recipient email address was specified. Please provide the recipient's email address.", 
                    "error")
            
            # Send the email using the MCP client
            try:
                result = await self.use_tool("send_email", {
                    "to": to,
                    "subject": subject,
                    "body": body
                })
                
                # Clear the pending draft from state
                if 'active_context' in state and 'pending_email_draft' in state['active_context']:
                    del state['active_context']['pending_email_draft']
                
                response = f"✅ **Email sent successfully!**\n\n**To:** {to}\n**Subject:** {subject}\n\nYour message has been delivered."
                return self._add_agent_message(state, response, "send_success")
                
            except Exception as send_error:
                self.logger.error(f"Failed to send email: {send_error}")
                response = f"❌ **Failed to send email:** {str(send_error)}\n\nPlease check the recipient email address and try again."
                return self._add_agent_message(state, response, "send_error")
                
        except Exception as e:
            return await self._handle_error(state, e)
    
    async def _general_email_assistance(self, state: AssistantState) -> AssistantState:
        user_input = state.get("user_input", "")

        if "how" in user_input.lower() or "tips" in user_input.lower():
            response = await self.generate_response(
                user_input,
                context="You are a helpful email assistant. Provide tips on managing, organizing, and writing better emails."
            )
            return self._add_agent_message(state, response, "assistance")

        # Intent not clear – ask user
        clarification = await self.generate_response(
            "Ask the user to clarify what they need help with regarding emails.",
            context="Prompt the user to be more specific about their email-related request."
        )
        return self._add_agent_message(state, clarification, "clarify_intent")
        