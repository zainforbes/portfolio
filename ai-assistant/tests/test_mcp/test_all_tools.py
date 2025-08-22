#!/usr/bin/env python3
"""
Comprehensive test suite for AI Assistant tools
Tests: GeminiClient, MCP Integration, SearchTools, and overall workflow
"""

import asyncio
import os
import sys
import traceback
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ToolsTester:
    def __init__(self):
        self.results = {
            "environment": {"passed": 0, "failed": 0, "details": []},
            "gemini": {"passed": 0, "failed": 0, "details": []},
            "mcp": {"passed": 0, "failed": 0, "details": []},
            "search": {"passed": 0, "failed": 0, "details": []},
            "integration": {"passed": 0, "failed": 0, "details": []}
        }
        
    def log_result(self, category: str, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        if passed:
            self.results[category]["passed"] += 1
            status = "✅"
        else:
            self.results[category]["failed"] += 1
            status = "❌"
        
        self.results[category]["details"].append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        
        print(f"   {status} {test_name}: {details}")
    
    def test_environment(self):
        """Test environment setup and variables"""
        print("\n🔧 Testing Environment Setup")
        print("=" * 50)
        
        # Test API keys
        brave_key = os.getenv("BRAVE_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        self.log_result("environment", "BRAVE_API_KEY", 
                       bool(brave_key), 
                       f"Found (length: {len(brave_key)})" if brave_key else "Not found")
        
        self.log_result("environment", "GOOGLE_API_KEY", 
                       bool(google_key), 
                       f"Found (length: {len(google_key)})" if google_key else "Not found")
        
        # Test Python version
        version_ok = sys.version_info >= (3, 8)
        self.log_result("environment", "Python Version", 
                       version_ok, 
                       f"Python {sys.version.split()[0]} {'✓' if version_ok else '(needs 3.8+)'}")
        
        # Test required imports
        required_packages = [
            ("google.generativeai", "google-generativeai"),
            ("httpx", "httpx"),
            ("dotenv", "python-dotenv")
        ]
        
        for package, pip_name in required_packages:
            try:
                __import__(package)
                self.log_result("environment", f"Package {pip_name}", True, "Available")
            except ImportError:
                self.log_result("environment", f"Package {pip_name}", False, f"Missing - run: pip install {pip_name}")
    
    def test_gemini_client(self):
        """Test Gemini client functionality"""
        print("\n🤖 Testing Gemini Client")
        print("=" * 50)
        
        try:
            from src.utils.gemini_client import GeminiClient
            self.log_result("gemini", "Import GeminiClient", True, "Successfully imported")
            
            # Test initialization
            try:
                client = GeminiClient()
                self.log_result("gemini", "Initialize Client", True, f"Model: {client.model_name}")
                
                # Test classification
                try:
                    result = client.classify("Send an email to John about the meeting")
                    expected_agent = result.get("agent", "").lower()
                    confidence = result.get("confidence", 0)
                    
                    classification_ok = expected_agent in ["email", "coordinator"] and confidence > 0
                    self.log_result("gemini", "Classification", classification_ok, 
                                  f"Agent: {expected_agent}, Confidence: {confidence:.2f}")
                    
                    # Test chat functionality
                    try:
                        chat_response = client.chat("Hello, can you respond with just 'OK' to test?")
                        chat_ok = "OK" in chat_response and not chat_response.startswith("❌")
                        self.log_result("gemini", "Chat Response", chat_ok, 
                                      f"Response: {chat_response[:50]}{'...' if len(chat_response) > 50 else ''}")
                    except Exception as e:
                        self.log_result("gemini", "Chat Response", False, f"Error: {str(e)}")
                
                except Exception as e:
                    self.log_result("gemini", "Classification", False, f"Error: {str(e)}")
            
            except Exception as e:
                self.log_result("gemini", "Initialize Client", False, f"Error: {str(e)}")
        
        except ImportError as e:
            self.log_result("gemini", "Import GeminiClient", False, f"Import error: {str(e)}")
    
    async def test_mcp_integration(self):
        """Test MCP client and integration"""
        print("\n🔌 Testing MCP Integration")
        print("=" * 50)
        
        try:
            from src.mcp_integration.mcp_client import MCPClient
            self.log_result("mcp", "Import MCPClient", True, "Successfully imported")
            
            # Test MCP client initialization
            try:
                client = MCPClient()
                self.log_result("mcp", "Initialize MCP Client", True, "Client created")
                
                # Test basic MCP functionality (if your client has any basic methods)
                # This is a placeholder - adapt based on your actual MCPClient methods
                try:
                    # Test cleanup
                    await client.shutdown()
                    self.log_result("mcp", "MCP Shutdown", True, "Clean shutdown completed")
                except Exception as e:
                    self.log_result("mcp", "MCP Shutdown", False, f"Error: {str(e)}")
                
            except Exception as e:
                self.log_result("mcp", "Initialize MCP Client", False, f"Error: {str(e)}")
        
        except ImportError as e:
            self.log_result("mcp", "Import MCPClient", False, f"Import error: {str(e)}")
    
    async def test_search_tools(self):
        """Test search tools functionality"""
        print("\n🔍 Testing Search Tools")
        print("=" * 50)
        
        try:
            from src.mcp_integration.mcp_client import MCPClient
            from src.mcp_integration.search_tools import SearchTools
            self.log_result("search", "Import SearchTools", True, "Successfully imported")
            
            client = None
            search = None
            
            try:
                client = MCPClient()
                search = SearchTools(client)
                self.log_result("search", "Initialize SearchTools", True, "Search tools created")
                
                # Test search tools startup
                try:
                    await search.start()
                    self.log_result("search", "Search Tools Start", True, "Started successfully")
                    
                    # Test web search
                    try:
                        results = await search.web_search("Python tutorial", count=3)
                        search_ok = isinstance(results, list) and len(results) > 0
                        details = f"Got {len(results)} results" if search_ok else "No results"
                        if search_ok and results:
                            details += f" - First: {results[0].get('title', 'No title')[:30]}..."
                        
                        self.log_result("search", "Web Search", search_ok, details)
                        
                    except Exception as e:
                        self.log_result("search", "Web Search", False, f"Error: {str(e)}")
                
                except Exception as e:
                    self.log_result("search", "Search Tools Start", False, f"Error: {str(e)}")
            
            except Exception as e:
                self.log_result("search", "Initialize SearchTools", False, f"Error: {str(e)}")
            
            finally:
                # Cleanup
                if client:
                    try:
                        await client.shutdown()
                    except:
                        pass
        
        except ImportError as e:
            self.log_result("search", "Import SearchTools", False, f"Import error: {str(e)}")
    
    async def test_integration_workflow(self):
        """Test complete integration workflow"""
        print("\n🔄 Testing Integration Workflow")
        print("=" * 50)
        
        try:
            from src.utils.gemini_client import GeminiClient
            from src.mcp_integration.mcp_client import MCPClient
            from src.mcp_integration.search_tools import SearchTools
            
            # Test complete workflow
            gemini = None
            mcp_client = None
            search = None
            
            try:
                # Initialize all components
                gemini = GeminiClient()
                mcp_client = MCPClient()
                search = SearchTools(mcp_client)
                await search.start()
                
                self.log_result("integration", "Full System Init", True, "All components initialized")
                
                # Test workflow: Classify -> Search -> Respond
                test_query = "Find information about Python web frameworks"
                
                # Step 1: Classify the request
                classification = gemini.classify(test_query)
                classify_ok = "agent" in classification and "confidence" in classification
                agent = classification.get("agent", "unknown")
                
                self.log_result("integration", "Query Classification", classify_ok, 
                              f"Agent: {agent}, Confidence: {classification.get('confidence', 0):.2f}")
                
                # Step 2: Perform search based on classification
                if agent in ["coordinator", "task"]:  # Assuming these might need search
                    search_results = await search.web_search("Python web frameworks", count=2)
                    search_ok = len(search_results) > 0
                    self.log_result("integration", "Contextual Search", search_ok, 
                                  f"Found {len(search_results)} results")
                    
                    # Step 3: Generate response with context
                    if search_ok:
                        context = f"Based on search results: {search_results[0].get('title', '')}"
                        response = gemini.chat(f"Briefly summarize: {context}")
                        response_ok = len(response) > 10 and not response.startswith("❌")
                        self.log_result("integration", "Contextual Response", response_ok, 
                                      f"Generated {len(response)} char response")
                
                self.log_result("integration", "Complete Workflow", True, "End-to-end test completed")
                
            except Exception as e:
                self.log_result("integration", "Integration Workflow", False, f"Error: {str(e)}")
                traceback.print_exc()
            
            finally:
                # Cleanup all components
                if mcp_client:
                    await mcp_client.shutdown()
        
        except Exception as e:
            self.log_result("integration", "Integration Setup", False, f"Setup error: {str(e)}")
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("🎯 TEST RESULTS SUMMARY")
        print("="*60)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in self.results.items():
            passed = results["passed"]
            failed = results["failed"]
            total = passed + failed
            
            if total > 0:
                success_rate = (passed / total) * 100
                status = "✅" if failed == 0 else "⚠️" if passed > failed else "❌"
                
                print(f"{status} {category.upper()}: {passed}/{total} passed ({success_rate:.1f}%)")
                
                total_passed += passed
                total_failed += failed
        
        print("-" * 60)
        grand_total = total_passed + total_failed
        overall_rate = (total_passed / grand_total) * 100 if grand_total > 0 else 0
        
        overall_status = "🎉" if total_failed == 0 else "⚠️" if total_passed > total_failed else "🚨"
        print(f"{overall_status} OVERALL: {total_passed}/{grand_total} tests passed ({overall_rate:.1f}%)")
        
        if total_failed == 0:
            print("\n🎉 All tests passed! Your AI assistant is ready to go!")
        elif total_passed > total_failed:
            print(f"\n⚠️  Most tests passed, but {total_failed} issues need attention.")
        else:
            print(f"\n🚨 Multiple issues detected. Please fix the failed tests.")
        
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    """Run comprehensive test suite"""
    print("🚀 AI Assistant Comprehensive Test Suite")
    print("Testing all components and integrations...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = ToolsTester()
    
    # Run all tests
    tester.test_environment()
    tester.test_gemini_client()
    await tester.test_mcp_integration()
    await tester.test_search_tools()
    await tester.test_integration_workflow()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())