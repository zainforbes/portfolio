# src/mcp_integration/mcp_client.py (Improved version with better debugging)
import asyncio
import json
import subprocess
from typing import Dict, List, Optional, Any
import logging
import os

class MCPClient:
    def __init__(self):
        self.servers = {}
        self.tools = {}
        self.logger = logging.getLogger(__name__)
        # Enable debug logging
        logging.basicConfig(level=logging.INFO)
    
    async def start_server(self, server_name: str, command: List[str], args: Dict[str, Any] = None):
        """Start an MCP server process"""
        try:
            self.logger.info(f"Starting server {server_name} with command: {' '.join(command)}")
            
            # Set up environment
            env = os.environ.copy()
            if args and 'env' in args:
                env.update(args['env'])
            
            # Start the server process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            self.servers[server_name] = {
                'process': process,
                'args': args or {}
            }
            
            # Initialize connection
            init_success = await self._initialize_server(server_name)
            if not init_success:
                self.logger.error(f"Failed to initialize server {server_name}")
                return False
                
            self.logger.info(f"Successfully started and initialized server {server_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start server {server_name}: {e}")
            return False
    
    async def _initialize_server(self, server_name: str):
        """Initialize connection with MCP server"""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "mcp-client",
                    "version": "1.0.0"
                }
            }
        }
        
        try:
            response = await self._send_request(server_name, init_request)
            self.logger.info(f"Initialize response: {json.dumps(response, indent=2)}")
            
            if 'result' in response:
                return True
            elif 'error' in response:
                self.logger.error(f"Server initialization error: {response['error']}")
                return False
            else:
                self.logger.error(f"Unexpected initialize response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during server initialization: {e}")
            return False
    
    async def _send_request(self, server_name: str, request: Dict):
        """Send JSON-RPC request to server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not started")
        
        process = self.servers[server_name]['process']
        
        # Send request
        request_json = json.dumps(request) + '\n'
        self.logger.debug(f"Sending request to {server_name}: {request_json.strip()}")
        
        process.stdin.write(request_json.encode())
        await process.stdin.drain()
        
        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            if not response_line:
                raise Exception("Empty response from server")
                
            response_text = response_line.decode().strip()
            self.logger.debug(f"Received response from {server_name}: {response_text}")
            
            response = json.loads(response_text)
            return response
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout waiting for response from {server_name}")
            raise Exception("Server response timeout")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from {server_name}: {response_text}")
            raise Exception(f"Invalid JSON response: {e}")
    
    async def list_tools(self, server_name: str) -> List[Dict]:
        """Get available tools from server"""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        try:
            response = await self._send_request(server_name, request)
            
            if 'result' in response:
                tools = response['result'].get('tools', [])
                self.tools[server_name] = tools
                self.logger.info(f"Found {len(tools)} tools for {server_name}: {[t.get('name', 'unnamed') for t in tools]}")
                return tools
            elif 'error' in response:
                self.logger.error(f"Error listing tools: {response['error']}")
                return []
            else:
                self.logger.error(f"Unexpected tools/list response: {response}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception listing tools: {e}")
            return []
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict = None) -> Dict:
        """Call a specific tool"""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }
        
        try:
            response = await self._send_request(server_name, request)
            
            if 'error' in response:
                self.logger.error(f"Tool call error: {response['error']}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Exception calling tool {tool_name}: {e}")
            return {
                "error": {
                    "code": -32603,
                    "message": f"Client error: {str(e)}"
                }
            }
    
    async def shutdown(self):
        """Shutdown all server connections"""
        for server_name, server_info in self.servers.items():
            try:
                process = server_info['process']
                if process.returncode is None:  # Process is still running
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Force killing server {server_name}")
                        process.kill()
                        await process.wait()
            except Exception as e:
                self.logger.error(f"Error shutting down {server_name}: {e}")
        
        self.servers.clear()
        self.tools.clear()