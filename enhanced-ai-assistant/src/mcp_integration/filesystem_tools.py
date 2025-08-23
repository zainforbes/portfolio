# src/mcp_integration/filesystem_tools.py (FIXED version)
import os
import sys
import platform
from pathlib import Path
from .gemini_mcp_client import GeminiMCPClient
from typing import List, Dict, Any

class FilesystemTools:
    def __init__(self, mcp_client: GeminiMCPClient):
        self.client = mcp_client
        self.server_name = "filesystem"
    
    def _get_filesystem_server_command(self, allowed_directories):
        """Get the correct command for your Windows system"""
        
        # Your npm prefix is: C:\Users\User\AppData\Roaming\npm
        npm_prefix = r"C:\Users\User\AppData\Roaming\npm"
        
        # Base commands to try
        base_commands = [
            # Method 1: Try regular npx (this failed in your case)
            ["npx", "@modelcontextprotocol/server-filesystem"],
            
            # Method 2: Try npx.cmd (this worked in your case)
            ["npx.cmd", "@modelcontextprotocol/server-filesystem"],
            
            # Method 3: Direct path to the server (backup)
            ["node", os.path.join(npm_prefix, "node_modules", "@modelcontextprotocol", "server-filesystem", "dist", "index.js")],
        ]
        
        # The MCP filesystem server takes allowed directories as positional arguments
        final_commands = []
        for base_command in base_commands:
            cmd_with_args = base_command.copy()
            if allowed_directories:
                # Add directories as positional arguments
                cmd_with_args.extend(allowed_directories)
            final_commands.append(cmd_with_args)
        
        return final_commands
    
    def _extract_content_from_mcp_response(self, response: Dict) -> str:
        """Extract text content from MCP response format"""
        if 'result' not in response:
            return ""
        
        result = response['result']
        
        # Handle different response formats
        if isinstance(result, dict):
            # New MCP format: {"content": [{"type": "text", "text": "content"}]}
            if 'content' in result and isinstance(result['content'], list):
                text_parts = []
                for item in result['content']:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                return ''.join(text_parts)
            
            # Old format: {"content": "content"}  
            elif 'content' in result and isinstance(result['content'], str):
                return result['content']
        
        # Fallback: return empty string
        return ""
    
    def _extract_entries_from_list_response(self, response: Dict) -> List[Dict]:
        """Extract directory entries from MCP list response"""
        if 'result' not in response:
            return []
        
        result = response['result']
        
        # Handle new MCP format
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            if isinstance(content, list) and len(content) > 0:
                # The content is text like "[FILE] test.txt"
                # We need to parse this into proper entries
                entries = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        # Parse lines like "[FILE] test.txt" or "[DIR] folder"
                        for line in text.split('\n'):
                            line = line.strip()
                            if line.startswith('[FILE]'):
                                name = line[6:].strip()
                                entries.append({'name': name, 'type': 'file'})
                            elif line.startswith('[DIR]'):
                                name = line[5:].strip()
                                entries.append({'name': name, 'type': 'directory'})
                return entries
        
        # Handle old format (if any)
        if isinstance(result, dict) and 'entries' in result:
            return result['entries']
        
        return []
    
    async def start(self, allowed_directories = None):
        """Initialize filesystem tools via GeminiMCPClient"""
        # Filesystem functionality is built into GeminiMCPClient, so just verify it's initialized
        try:
            await self.client.initialize()
            print("✓ Filesystem tools initialized via GeminiMCPClient!")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize filesystem tools: {e}")
            return False
    
    async def read_file(self, filepath: str) -> str:
        """Read contents of a file"""
        result = await self.client._read_file(filepath)
        return result
    
    async def write_file(self, filepath: str, content: str) -> bool:
        """Write content to a file"""
        result = await self.client._write_file(filepath, content)
        # Check if the operation was successful (no error message means success)
        return not result.startswith("File writing failed")
    
    async def list_directory(self, dirpath: str) -> List[Dict]:
        """List contents of a directory"""
        result = await self.client._list_directory(dirpath)
        # Parse the directory listing text into dictionary format
        entries = []
        if result and not result.startswith("Directory listing failed"):
            lines = result.split('\n')
            for line in lines:
                if line.strip().startswith('DIR:') or line.strip().startswith('FILE:'):
                    parts = line.strip().split(': ', 1)
                    if len(parts) == 2:
                        type_str = parts[0].strip()
                        name = parts[1].split(' (')[0] if ' (' in parts[1] else parts[1]
                        entries.append({
                            'name': name,
                            'type': 'directory' if type_str == 'DIR' else 'file'
                        })
        return entries