# tests/test_mcp/test_filesystem_fixed.py
import asyncio
import tempfile
import os
from pathlib import Path

# Add the src directory to the Python path
import sys
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.mcp_integration.mcp_client import MCPClient
from src.mcp_integration.filesystem_tools import FilesystemTools

async def test_filesystem():
    client = MCPClient()
    fs_tools = FilesystemTools(client)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in directory: {temp_dir}")
        
        # Start filesystem server
        success = await fs_tools.start([temp_dir])
        assert success, "Failed to start filesystem server"
        
        # Test file operations
        test_file = os.path.join(temp_dir, "test.txt")
        test_content = "Hello, MCP World!"
        
        print(f"\n1. Writing file: {test_file}")
        # Write file
        write_success = await fs_tools.write_file(test_file, test_content)
        assert write_success, "Failed to write file"
        print("✓ File write successful")
        
        print(f"\n2. Reading file: {test_file}")
        # Read file
        read_content = await fs_tools.read_file(test_file)
        print(f"   Read content: '{read_content}'")
        print(f"   Expected:     '{test_content}'")
        assert read_content == test_content, f"File content mismatch: got '{read_content}', expected '{test_content}'"
        print("✓ File read successful")
        
        print(f"\n3. Listing directory: {temp_dir}")
        # List directory
        entries = await fs_tools.list_directory(temp_dir)
        print(f"   Directory entries: {entries}")
        entry_names = [entry.get('name', '') for entry in entries]
        assert "test.txt" in entry_names, f"File not in directory listing: {entry_names}"
        print("✓ Directory listing successful")
        
        print(f"\n4. Testing multiple files")
        # Test multiple files
        test_file2 = os.path.join(temp_dir, "test2.txt")
        test_content2 = "Second file content"
        
        await fs_tools.write_file(test_file2, test_content2)
        read_content2 = await fs_tools.read_file(test_file2)
        assert read_content2 == test_content2, "Second file content mismatch"
        print("✓ Multiple file operations successful")
        
        # Final directory listing
        entries = await fs_tools.list_directory(temp_dir)
        entry_names = [entry.get('name', '') for entry in entries]
        assert len(entry_names) == 2, f"Expected 2 files, found {len(entry_names)}"
        assert "test.txt" in entry_names and "test2.txt" in entry_names
        print("✓ Final directory listing correct")
        
        print("\n🎉 All filesystem tests passed!")
        
        # Cleanup
        await client.shutdown()

if __name__ == "__main__":
    print("Running fixed filesystem test...")
    asyncio.run(test_filesystem())