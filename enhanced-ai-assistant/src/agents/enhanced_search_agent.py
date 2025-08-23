import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict
from urllib.parse import urlparse

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

class EnhancedSearchAgent(EnhancedBaseAgent):
    """
    Enhanced Search Agent with full AI capabilities:
    - Intelligent query optimization and refinement
    - Advanced result analysis and synthesis
    - Multi-source research coordination
    - Fact verification and source validation
    - Context-aware search strategies
    - Resource optimization and caching
    - Error recovery and fallback strategies
    """
    
    def __init__(self, mcp_client, agent_name: str = "EnhancedSearchAgent"):
        capabilities = [
            'web_search', 'research_analysis', 'fact_verification',
            'information_synthesis', 'source_validation', 'query_optimization',
            'search_result_ranking', 'content_summarization', 'trend_analysis'
        ]
        
        super().__init__(mcp_client, agent_name, capabilities)
        
        # Search-specific configuration
        self.search_cache: Dict[str, Any] = {}
        self.query_patterns = self._initialize_query_patterns()
        self.source_reliability = self._initialize_source_reliability()
        self.search_strategies = self._initialize_search_strategies()
        
    def get_task_types(self) -> List[TaskType]:
        """Return search task types this agent can handle"""
        return [
            TaskType.WEB_SEARCH,
            TaskType.RESEARCH_ANALYSIS,
            TaskType.FACT_VERIFICATION,
            TaskType.INFORMATION_SYNTHESIS,
            TaskType.SOURCE_VALIDATION
        ]
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize search-specific prompt templates"""
        return {
            'query_optimization': """ADVANCED SEARCH QUERY OPTIMIZATION

Optimize this search query for maximum effectiveness:

ORIGINAL QUERY: "{original_query}"
SEARCH CONTEXT: {search_context}
USER INTENT: {user_intent}

OPTIMIZATION REQUIREMENTS:
1. Improve search precision and recall
2. Consider multiple search strategies
3. Include relevant keywords and synonyms
4. Account for different information sources
5. Optimize for current trends and terminology
6. Consider search engine algorithms

RESPOND WITH STRUCTURED JSON:
{{
    "optimized_queries": [
        {{
            "query": "optimized_search_string",
            "strategy": "broad|specific|academic|news|recent",
            "expected_results": "description",
            "confidence": 0.0-1.0
        }}
    ],
    "query_analysis": {{
        "intent_category": "factual|research|comparison|how_to|current_events",
        "complexity_level": "simple|moderate|complex",
        "information_type": "facts|opinions|procedures|data|news"
    }},
    "search_strategy": {{
        "recommended_approach": "single_query|multi_query|iterative|comparative",
        "source_preferences": ["academic", "news", "official", "community"],
        "temporal_focus": "recent|historical|comprehensive"
    }},
    "confidence": 0.0-1.0,
    "reasoning": "detailed optimization explanation"
}}

HALLUCINATION PREVENTION:
- Base optimizations on proven search techniques
- Only suggest realistic query improvements
- Maintain the user's original intent""",

            'result_analysis': """COMPREHENSIVE SEARCH RESULT ANALYSIS

Analyze these search results for quality and relevance:

ORIGINAL QUERY: "{original_query}"
SEARCH RESULTS: {search_results}
USER CONTEXT: {user_context}

ANALYSIS REQUIREMENTS:
1. Evaluate result relevance and quality
2. Assess source credibility and authority
3. Identify key themes and patterns
4. Extract actionable insights
5. Detect potential misinformation
6. Synthesize comprehensive answer

RESPOND WITH STRUCTURED JSON:
{{
    "result_evaluation": {{
        "total_results": 0,
        "high_quality_results": 0,
        "relevance_score": 0.0-1.0,
        "source_diversity": 0.0-1.0
    }},
    "source_analysis": {{
        "authoritative_sources": ["list"],
        "questionable_sources": ["list"],
        "source_types": {{"news": 0, "academic": 0, "commercial": 0, "government": 0}}
    }},
    "content_synthesis": {{
        "key_findings": ["list", "of", "main", "points"],
        "consensus_information": ["verified", "facts"],
        "conflicting_information": ["disputed", "points"],
        "information_gaps": ["missing", "information"]
    }},
    "answer_synthesis": "comprehensive_answer_based_on_results",
    "fact_verification": {{
        "verified_facts": ["list"],
        "uncertain_claims": ["list"],
        "contradictory_information": ["list"]
    }},
    "confidence": 0.0-1.0,
    "reasoning": "detailed analysis explanation"
}}""",

            'fact_verification': """ADVANCED FACT VERIFICATION

Verify the accuracy of these claims using search results:

CLAIMS TO VERIFY: {claims}
SEARCH RESULTS: {search_results}
VERIFICATION CONTEXT: {context}

VERIFICATION REQUIREMENTS:
1. Cross-reference multiple reliable sources
2. Assess claim specificity and verifiability
3. Identify supporting and contradicting evidence
4. Evaluate source credibility and bias
5. Consider recency and relevance of sources
6. Provide confidence scores for each claim

RESPOND WITH STRUCTURED JSON:
{{
    "claim_verification": [
        {{
            "claim": "specific_claim_text",
            "verification_status": "verified|partially_verified|unverified|contradicted",
            "confidence": 0.0-1.0,
            "supporting_sources": ["list"],
            "contradicting_sources": ["list"],
            "evidence_strength": "strong|moderate|weak|insufficient",
            "reasoning": "verification_explanation"
        }}
    ],
    "overall_assessment": {{
        "verified_claims": 0,
        "uncertain_claims": 0,
        "contradicted_claims": 0,
        "reliability_score": 0.0-1.0
    }},
    "source_reliability": {{
        "highly_reliable": ["sources"],
        "moderately_reliable": ["sources"],
        "questionable": ["sources"]
    }},
    "verification_summary": "overall_verification_summary",
    "confidence": 0.0-1.0
}}""",

            'research_synthesis': """INTELLIGENT RESEARCH SYNTHESIS

Synthesize information from multiple sources into comprehensive research:

RESEARCH TOPIC: "{research_topic}"
SOURCE MATERIALS: {source_materials}
SYNTHESIS CONTEXT: {synthesis_context}

SYNTHESIS REQUIREMENTS:
1. Identify key themes and patterns across sources
2. Resolve contradictions and inconsistencies
3. Create coherent narrative from fragmented information
4. Highlight certainties vs uncertainties
5. Provide balanced perspective
6. Include source attribution

RESPOND WITH STRUCTURED JSON:
{{
    "research_synthesis": {{
        "executive_summary": "concise_overview",
        "key_themes": ["main", "themes", "identified"],
        "main_findings": [
            {{
                "finding": "specific_finding",
                "support_level": "strong|moderate|limited",
                "sources": ["supporting_sources"],
                "confidence": 0.0-1.0
            }}
        ],
        "contradictions_resolved": ["explanation", "of", "contradictions"],
        "uncertainties": ["areas", "of", "uncertainty"],
        "research_gaps": ["identified", "gaps"]
    }},
    "comprehensive_answer": "detailed_synthesized_response",
    "source_attribution": {{
        "primary_sources": ["most_important_sources"],
        "supporting_sources": ["additional_sources"],
        "source_quality_assessment": "overall_source_quality"
    }},
    "confidence": 0.0-1.0,
    "reasoning": "synthesis_methodology_explanation"
}}"""
        }
    
    def _initialize_query_patterns(self) -> Dict[str, Any]:
        """Initialize query pattern recognition"""
        return {
            'intent_patterns': {
                'factual': [r'what is', r'who is', r'when did', r'where is', r'how much', r'how many'],
                'procedural': [r'how to', r'how do', r'steps to', r'guide to', r'tutorial'],
                'comparison': [r'vs', r'versus', r'compare', r'difference between', r'better than'],
                'current_events': [r'latest', r'recent', r'news', r'update', r'current', r'today'],
                'research': [r'research', r'study', r'analysis', r'investigate', r'explore']
            },
            'complexity_indicators': {
                'simple': [r'define', r'what is', r'basic'],
                'moderate': [r'explain', r'describe', r'overview'],
                'complex': [r'analyze', r'compare', r'evaluate', r'research', r'comprehensive']
            },
            'temporal_patterns': {
                'recent': [r'latest', r'recent', r'current', r'today', r'now', r'2024', r'2025'],
                'historical': [r'history', r'origin', r'when did', r'first', r'originally'],
                'trending': [r'trending', r'popular', r'viral', r'hot topic']
            }
        }
    
    def _initialize_source_reliability(self) -> Dict[str, float]:
        """Initialize source reliability scoring"""
        return {
            # High reliability domains
            '.edu': 0.9, '.gov': 0.95, '.org': 0.7,
            'wikipedia.org': 0.8, 'ncbi.nlm.nih.gov': 0.95,
            'nature.com': 0.9, 'science.org': 0.9,
            # News sources
            'reuters.com': 0.85, 'ap.org': 0.85, 'bbc.com': 0.8,
            'npr.org': 0.8, 'pbs.org': 0.8,
            # Tech sources  
            'github.com': 0.7, 'stackoverflow.com': 0.75,
            # Commercial but reliable
            'webmd.com': 0.7, 'mayoclinic.org': 0.85,
            # Lower reliability
            '.com': 0.5, 'blog': 0.4, 'forum': 0.3
        }
    
    def _initialize_search_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize search strategy configurations"""
        return {
            'comprehensive': {
                'query_count': 3,
                'result_count': 15,
                'source_diversity': True,
                'fact_checking': True
            },
            'quick': {
                'query_count': 1,
                'result_count': 5,
                'source_diversity': False,
                'fact_checking': False
            },
            'academic': {
                'query_count': 2,
                'result_count': 10,
                'source_diversity': True,
                'preferred_sources': ['.edu', '.org', 'scholar.google.com']
            },
            'news': {
                'query_count': 2,
                'result_count': 8,
                'temporal_focus': 'recent',
                'preferred_sources': ['news', 'reuters', 'ap', 'bbc']
            }
        }
    
    async def execute(self, state: EnhancedAssistantState) -> EnhancedAssistantState:
        """Main execution method using the full AI pipeline"""
        return await self.execute_with_full_pipeline(state)
    
    async def _execute_task(self, decision: AgentDecision, 
                          state: EnhancedAssistantState, 
                          context: Dict[str, Any]) -> TaskResult:
        """Execute search-specific tasks with proper state management"""
        task_type = decision.parameters.get('task_type', TaskType.WEB_SEARCH.value)
        
        # Record the decision in state
        record_agent_decision(state, decision)
        
        try:
            if task_type == TaskType.WEB_SEARCH.value:
                return await self._handle_web_search(decision, state, context)
            elif task_type == TaskType.RESEARCH_ANALYSIS.value:
                return await self._handle_research_analysis(decision, state, context)
            elif task_type == TaskType.FACT_VERIFICATION.value:
                return await self._handle_fact_verification(decision, state, context)
            elif task_type == TaskType.INFORMATION_SYNTHESIS.value:
                return await self._handle_information_synthesis(decision, state, context)
            elif task_type == TaskType.SOURCE_VALIDATION.value:
                return await self._handle_source_validation(decision, state, context)
            else:
                error_msg = f"Unknown search task type: {task_type}"
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
    
    async def _handle_web_search(self, decision: AgentDecision,
                               state: EnhancedAssistantState,
                               context: Dict[str, Any]) -> TaskResult:
        """Handle intelligent web search with optimization"""
        try:
            # Check rate limits before making API calls
            if is_rate_limited(state, 'gemini') or is_rate_limited(state, 'brave_api'):
                return await self._handle_rate_limited_search(decision, state, context)
                
            user_input = state.get('user_input', '')
            
            # 1. Optimize search query
            query_optimization = await self._optimize_search_query(user_input, context, state)
            
            # Update resource metrics for query optimization
            update_resource_metrics(state, api_calls=1, processing_time=0.3)
            
            # 2. Determine search strategy
            search_strategy = self._select_search_strategy(query_optimization, context)
            
            # 3. Execute optimized searches
            search_results = await self._execute_optimized_searches(
                query_optimization, search_strategy, state
            )
            
            if not search_results or search_results.get('search_unavailable'):
                # Handle search unavailable scenario
                fallback_response = search_results.get('fallback_response', '') if search_results else ''
                response = self._format_search_unavailable_response(user_input, fallback_response)
                
                return TaskResult(
                    success=True,
                    data=response,
                    confidence=0.6,
                    task_type=TaskType.WEB_SEARCH.value,
                    agent=self.agent_name
                )
            
            # 4. Analyze and synthesize results
            result_analysis = await self._analyze_search_results(
                search_results, query_optimization, context, state
            )
            
            # 5. Generate comprehensive response
            response = self._format_search_response(
                result_analysis, query_optimization, search_results
            )
            
            # 6. Store results in state for follow-up queries
            state['search_results'] = search_results.get('results', [])
            state['search_analysis'] = result_analysis
            state['original_query'] = user_input
            
            return TaskResult(
                success=True,
                data=response,
                confidence=result_analysis.get('confidence', 0.8),
                task_type=TaskType.WEB_SEARCH.value,
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.WEB_SEARCH.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    async def _handle_research_analysis(self, decision: AgentDecision,
                                      state: EnhancedAssistantState,
                                      context: Dict[str, Any]) -> TaskResult:
        """Handle comprehensive research analysis"""
        try:
            # Get previous search results or perform new search
            search_results = state.get('search_results', [])
            if not search_results:
                # No previous results, perform search first
                search_task = await self._handle_web_search(decision, state, context)
                search_results = state.get('search_results', [])
            
            # Perform comprehensive research analysis
            research_synthesis = await self._synthesize_research_findings(
                search_results, context, state
            )
            
            # Generate research report
            response = self._format_research_analysis_response(research_synthesis)
            
            return TaskResult(
                success=True,
                data=response,
                confidence=research_synthesis.get('confidence', 0.8),
                task_type=TaskType.RESEARCH_ANALYSIS.value,
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.RESEARCH_ANALYSIS.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    async def _handle_fact_verification(self, decision: AgentDecision,
                                      state: EnhancedAssistantState,
                                      context: Dict[str, Any]) -> TaskResult:
        """Handle fact verification using multiple sources"""
        try:
            user_input = state.get('user_input', '')
            
            # Extract claims to verify
            claims = await self._extract_verifiable_claims(user_input)
            
            # Get search results for verification
            search_results = state.get('search_results', [])
            if not search_results:
                # Perform targeted search for fact verification
                verification_query = f"verify facts: {user_input}"
                search_task = await self._perform_verification_search(verification_query, state)
                search_results = search_task.get('results', [])
            
            # Verify claims against sources
            verification_result = await self._verify_claims_against_sources(
                claims, search_results, context, state
            )
            
            # Generate verification response
            response = self._format_fact_verification_response(verification_result, claims)
            
            return TaskResult(
                success=True,
                data=response,
                confidence=verification_result.get('confidence', 0.8),
                task_type=TaskType.FACT_VERIFICATION.value,
                agent=self.agent_name
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                task_type=TaskType.FACT_VERIFICATION.value,
                agent=self.agent_name,
                error=str(e)
            )
    
    # === CORE SEARCH OPERATIONS ===
    
    async def _optimize_search_query(self, user_input: str, context: Dict[str, Any], 
                                   state: EnhancedAssistantState) -> Dict[str, Any]:
        """Optimize search query using advanced LLM analysis"""
        user_intent = self._classify_search_intent(user_input)
        search_context = context.get('summary', '')
        
        prompt = self.prompt_templates['query_optimization'].format(
            original_query=user_input,
            search_context=search_context,
            user_intent=user_intent
        )
        
        try:
            response = await self.generate_response(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            return self._get_fallback_query_optimization(user_input)
    
    async def _execute_optimized_searches(self, query_optimization: Dict[str, Any],
                                        search_strategy: Dict[str, Any],
                                        state: EnhancedAssistantState) -> Dict[str, Any]:
        """Execute multiple optimized searches"""
        optimized_queries = query_optimization.get('optimized_queries', [])
        all_results = []
        
        for query_info in optimized_queries[:search_strategy.get('query_count', 2)]:
            query = query_info.get('query', '')
            
            try:
                # Check cache first
                cache_key = f"search_{hash(query)}"
                if cache_key in self.search_cache:
                    cache_entry = self.search_cache[cache_key]
                    if (datetime.now() - cache_entry['timestamp']).seconds < 1800:  # 30 min cache
                        all_results.extend(cache_entry['results'])
                        continue
                
                # Perform search using MCP client
                search_result = await self._perform_single_search(query, state)
                
                if search_result.get('results'):
                    all_results.extend(search_result['results'])
                    
                    # Cache results
                    self.search_cache[cache_key] = {
                        'results': search_result['results'],
                        'timestamp': datetime.now()
                    }
                elif search_result.get('search_unavailable'):
                    return search_result  # Return unavailable status
                    
            except Exception as e:
                self.logger.warning(f"Search query failed: {query} - {e}")
                continue
        
        return {
            'results': self._deduplicate_results(all_results),
            'total_queries': len(optimized_queries),
            'successful_queries': len([r for r in all_results if r])
        }
    
    async def _perform_single_search(self, query: str, 
                                   state: EnhancedAssistantState) -> Dict[str, Any]:
        """Perform a single search with fallback handling"""
        try:
            # Try direct search through Gemini MCP client search tools
            if hasattr(self.mcp_client, 'search_tools') and self.mcp_client.search_tools:
                try:
                    return await self.mcp_client.search_tools.web_search(query, 10)
                except Exception as e:
                    self.logger.warning(f"Search tools failed: {e}")
            
            # Fallback: try MCP tool interface
            try:
                return await self.use_tool('search_web', {'query': query, 'count': 10})
            except Exception as e:
                self.logger.warning(f"MCP tool search failed: {e}")
            
            # Final fallback: Gemini MCP client _search_web method
            return await self.mcp_client._search_web(query, 10)
            
        except Exception as e:
            self.logger.error(f"All search methods failed for query '{query}': {e}")
            return {'results': [], 'error': str(e)}
    
    async def _analyze_search_results(self, search_results: Dict[str, Any],
                                    query_optimization: Dict[str, Any],
                                    context: Dict[str, Any],
                                    state: EnhancedAssistantState) -> Dict[str, Any]:
        """Analyze search results comprehensively"""
        original_query = query_optimization.get('original_query', state.get('user_input', ''))
        results = search_results.get('results', [])
        user_context = context.get('summary', '')
        
        prompt = self.prompt_templates['result_analysis'].format(
            original_query=original_query,
            search_results=json.dumps(results[:10], indent=2),  # Top 10 results
            user_context=user_context
        )
        
        try:
            response = await self.generate_response(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Result analysis failed: {e}")
            return self._get_fallback_result_analysis(results)
    
    async def _synthesize_research_findings(self, search_results: List[Dict[str, Any]],
                                          context: Dict[str, Any],
                                          state: EnhancedAssistantState) -> Dict[str, Any]:
        """Synthesize comprehensive research from multiple sources"""
        research_topic = state.get('user_input', '')
        synthesis_context = context.get('summary', '')
        
        # Prepare source materials
        source_materials = []
        for result in search_results[:15]:  # Top 15 results
            source_materials.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'description': result.get('description', ''),
                'reliability_score': self._assess_source_reliability(result.get('url', ''))
            })
        
        prompt = self.prompt_templates['research_synthesis'].format(
            research_topic=research_topic,
            source_materials=json.dumps(source_materials, indent=2),
            synthesis_context=synthesis_context
        )
        
        try:
            response = await self.generate_response(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Research synthesis failed: {e}")
            return self._get_fallback_research_synthesis(search_results)
    
    # === UTILITY METHODS ===
    
    def _classify_search_intent(self, query: str) -> str:
        """Classify the intent of a search query"""
        query_lower = query.lower()
        
        for intent, patterns in self.query_patterns['intent_patterns'].items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return intent
        
        return 'general'
    
    def _select_search_strategy(self, query_optimization: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal search strategy based on query analysis"""
        query_analysis = query_optimization.get('query_analysis', {})
        complexity = query_analysis.get('complexity_level', 'moderate')
        intent = query_analysis.get('intent_category', 'general')
        
        # Select strategy based on complexity and intent
        if complexity == 'complex' or intent == 'research':
            return self.search_strategies['comprehensive']
        elif intent == 'current_events':
            return self.search_strategies['news']
        elif intent == 'factual' and any(term in query_analysis.get('information_type', '') 
                                       for term in ['facts', 'data']):
            return self.search_strategies['academic']
        else:
            return self.search_strategies['quick']
    
    def _assess_source_reliability(self, url: str) -> float:
        """Assess the reliability of a source URL"""
        if not url:
            return 0.5
        
        domain = urlparse(url).netloc.lower()
        
        # Check exact domain matches
        for known_domain, score in self.source_reliability.items():
            if known_domain in domain:
                return score
        
        # Check domain extensions
        if domain.endswith('.edu'):
            return 0.9
        elif domain.endswith('.gov'):
            return 0.95
        elif domain.endswith('.org'):
            return 0.7
        elif domain.endswith('.com'):
            return 0.5
        
        return 0.4  # Unknown domain
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate search results"""
        seen_urls = set()
        deduplicated = []
        
        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(result)
        
        return deduplicated
    
    # === RESPONSE FORMATTERS ===
    
    def _format_search_response(self, result_analysis: Dict[str, Any],
                              query_optimization: Dict[str, Any],
                              search_results: Dict[str, Any]) -> str:
        """Format comprehensive search response"""
        answer = result_analysis.get('answer_synthesis', '')
        if not answer:
            answer = "I found relevant information but couldn't synthesize a comprehensive answer."
        
        response = f"{answer}\n\n"
        
        # Add key findings if available
        content_synthesis = result_analysis.get('content_synthesis', {})
        key_findings = content_synthesis.get('key_findings', [])
        if key_findings:
            response += "🔍 **Key Findings:**\n"
            for finding in key_findings[:3]:
                response += f"• {finding}\n"
            response += "\n"
        
        # Add source quality information
        source_analysis = result_analysis.get('source_analysis', {})
        authoritative_sources = source_analysis.get('authoritative_sources', [])
        if authoritative_sources:
            response += f"📊 **Sources:** {len(authoritative_sources)} authoritative sources analyzed\n"
        
        # Add confidence and follow-up
        confidence = result_analysis.get('confidence', 0.8)
        response += f"📈 **Confidence:** {confidence:.0%}\n\n"
        
        response += "🔍 **What else would you like to search for?**\n"
        response += "• Research specific topics\n• Verify facts and claims\n• Analyze trends\n• Compare information sources"
        
        return response
    
    def _format_search_unavailable_response(self, user_input: str, 
                                          fallback_response: str) -> str:
        """Format response when search is unavailable"""
        response = f"🔍 **Search Results for '{user_input}'**\n\n"
        
        if fallback_response:
            response += fallback_response + "\n\n"
        else:
            response += "Search services are temporarily unavailable. Here's what I can tell you based on my knowledge:\n\n"
            response += f"I'd be happy to help with information about {user_input}, but I currently don't have access to real-time search capabilities. "
            response += "Please try again later, or ask me about something I might know from my training data.\n\n"
        
        response += "🔄 **You can try:**\n"
        response += "• Rephrasing your search query\n"
        response += "• Asking about related topics\n" 
        response += "• Trying again in a few minutes\n"
        response += "• Asking me questions I can answer from my knowledge base"
        
        return response
    
    def _format_research_analysis_response(self, research_synthesis: Dict[str, Any]) -> str:
        """Format comprehensive research analysis response"""
        synthesis = research_synthesis.get('research_synthesis', {})
        executive_summary = synthesis.get('executive_summary', '')
        key_themes = synthesis.get('key_themes', [])
        main_findings = synthesis.get('main_findings', [])
        
        response = f"📊 **Research Analysis Report**\n\n"
        
        if executive_summary:
            response += f"**Executive Summary:**\n{executive_summary}\n\n"
        
        if key_themes:
            response += "🎯 **Key Themes:**\n"
            for theme in key_themes:
                response += f"• {theme}\n"
            response += "\n"
        
        if main_findings:
            response += "🔍 **Main Findings:**\n"
            for i, finding in enumerate(main_findings[:5], 1):
                finding_text = finding.get('finding', '') if isinstance(finding, dict) else finding
                support_level = finding.get('support_level', 'moderate') if isinstance(finding, dict) else 'moderate'
                response += f"{i}. {finding_text} ({support_level} support)\n"
            response += "\n"
        
        # Add uncertainties and gaps
        uncertainties = synthesis.get('uncertainties', [])
        if uncertainties:
            response += "❓ **Areas of Uncertainty:**\n"
            for uncertainty in uncertainties[:3]:
                response += f"• {uncertainty}\n"
            response += "\n"
        
        confidence = research_synthesis.get('confidence', 0.8)
        response += f"📈 **Analysis Confidence:** {confidence:.0%}\n\n"
        
        response += "🔍 **Need more research?** I can help with:\n"
        response += "• Deeper analysis of specific aspects\n"
        response += "• Fact verification\n"
        response += "• Source validation\n"
        response += "• Related topic exploration"
        
        return response
    
    def _format_fact_verification_response(self, verification_result: Dict[str, Any],
                                         claims: List[str]) -> str:
        """Format fact verification response"""
        response = f"✅ **Fact Verification Report**\n\n"
        
        claim_verifications = verification_result.get('claim_verification', [])
        
        if claim_verifications:
            response += "**Claim Analysis:**\n"
            for i, verification in enumerate(claim_verifications, 1):
                claim = verification.get('claim', f'Claim {i}')
                status = verification.get('verification_status', 'unverified')
                confidence = verification.get('confidence', 0.0)
                
                status_emoji = {
                    'verified': '✅',
                    'partially_verified': '⚠️',
                    'unverified': '❓',
                    'contradicted': '❌'
                }.get(status, '❓')
                
                response += f"\n{i}. **{claim}**\n"
                response += f"   {status_emoji} Status: {status.replace('_', ' ').title()}\n"
                response += f"   📊 Confidence: {confidence:.0%}\n"
                
                reasoning = verification.get('reasoning', '')
                if reasoning:
                    response += f"   💡 {reasoning}\n"
        
        # Overall assessment
        overall = verification_result.get('overall_assessment', {})
        if overall:
            response += f"\n**Overall Assessment:**\n"
            response += f"• Verified Claims: {overall.get('verified_claims', 0)}\n"
            response += f"• Uncertain Claims: {overall.get('uncertain_claims', 0)}\n"
            response += f"• Contradicted Claims: {overall.get('contradicted_claims', 0)}\n"
            response += f"• Reliability Score: {overall.get('reliability_score', 0.0):.0%}\n"
        
        return response
    
    # === FALLBACK METHODS ===
    
    def _get_fallback_query_optimization(self, user_input: str) -> Dict[str, Any]:
        """Fallback query optimization using pattern matching"""
        return {
            'optimized_queries': [
                {
                    'query': user_input,
                    'strategy': 'broad',
                    'expected_results': 'general information',
                    'confidence': 0.5
                }
            ],
            'query_analysis': {
                'intent_category': self._classify_search_intent(user_input),
                'complexity_level': 'moderate',
                'information_type': 'general'
            },
            'confidence': 0.5,
            'reasoning': 'Fallback query optimization used'
        }
    
    def _get_fallback_result_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback result analysis using simple heuristics"""
        total_results = len(results)
        high_quality_count = sum(1 for r in results if self._assess_source_reliability(r.get('url', '')) > 0.7)
        
        return {
            'result_evaluation': {
                'total_results': total_results,
                'high_quality_results': high_quality_count,
                'relevance_score': 0.6,
                'source_diversity': 0.7
            },
            'answer_synthesis': f"Based on {total_results} search results, I found relevant information but couldn't provide a detailed analysis.",
            'confidence': 0.5,
            'reasoning': 'Fallback analysis due to processing limitations'
        }
    
    def _get_fallback_research_synthesis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback research synthesis"""
        return {
            'research_synthesis': {
                'executive_summary': f'Analysis based on {len(results)} sources',
                'key_themes': ['Information available from multiple sources'],
                'main_findings': [{'finding': 'Multiple perspectives found', 'support_level': 'moderate'}]
            },
            'comprehensive_answer': f'Based on the available sources, there appears to be relevant information, but detailed synthesis is not available.',
            'confidence': 0.5
        }
    
    def _get_task_patterns(self) -> Dict[str, List[str]]:
        """Get search-specific task patterns"""
        return {
            'web_search': ['search', 'find', 'look up', 'research', 'what is', 'who is', 'how to'],
            'research_analysis': ['analyze', 'research', 'investigate', 'study', 'comprehensive'],
            'fact_verification': ['verify', 'fact check', 'confirm', 'validate', 'true or false'],
            'information_synthesis': ['synthesize', 'combine', 'merge information', 'overview'],
            'source_validation': ['reliable', 'trustworthy', 'credible', 'source quality']
        }
    
    async def _handle_rate_limited_search(self, decision: AgentDecision, 
                                        state: EnhancedAssistantState, 
                                        context: Dict[str, Any]) -> TaskResult:
        """Handle rate-limited search with fallback"""
        user_input = state.get('user_input', '')
        fallback_response = f"I'm temporarily rate-limited for search queries. Based on my knowledge, I can provide some general information about: {user_input}. Please try again in a moment for real-time search results."
        
        return TaskResult(
            success=True,
            data=fallback_response,
            confidence=0.4,
            task_type=TaskType.WEB_SEARCH.value,
            agent=self.agent_name
        )