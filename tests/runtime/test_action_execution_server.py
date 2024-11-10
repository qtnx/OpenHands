import os
import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from fastapi import FastAPI

from openhands.runtime.action_execution_server import ActionExecutor
from openhands.events.action import (
    CmdRunAction,
    FileReadAction, 
    FileWriteAction,
    IPythonRunCellAction,
    CodeIndexerAction
)
from openhands.events.observation import (
    CmdOutputObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
    ErrorObservation
)
from openhands.events.serialization import event_to_dict

@pytest.fixture
def app():
    """Create a fresh FastAPI app for testing."""
    app = FastAPI()
    return app

@pytest.fixture
def test_client(app):
    """Create a test client with the test app."""
    return TestClient(app)

@pytest.fixture
def temp_workspace():
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)

@pytest_asyncio.fixture
async def action_executor(temp_workspace):
    """Create and initialize an ActionExecutor for testing."""
    executor = ActionExecutor(
        plugins_to_load=[],  # Empty for basic tests
        work_dir=temp_workspace,
        username="test_user",
        user_id=1000,
        browsergym_eval_env=None
    )
    await executor.ainit()
    yield executor
    executor.close()

class TestActionExecutionServer:
    
    @pytest.mark.asyncio
    async def test_cmd_run_action(self, action_executor):
        """Test basic command execution"""
        action = CmdRunAction(command="echo 'hello world'")
        observation = await action_executor.run_action(action)
        
        assert isinstance(observation, CmdOutputObservation)
        assert observation.exit_code == 0
        assert "hello world" in observation.content

    @pytest.mark.asyncio
    async def test_file_operations(self, action_executor, temp_workspace):
        """Test file read/write operations"""
        # Test file write
        test_content = "test content\nline 2"
        write_action = FileWriteAction(
            path="test.txt",
            content=test_content
        )
        write_obs = await action_executor.run_action(write_action)
        assert isinstance(write_obs, FileWriteObservation)
        
        # Test file read
        read_action = FileReadAction(path="test.txt")
        read_obs = await action_executor.run_action(read_action)
        assert isinstance(read_obs, FileReadObservation)
        assert read_obs.content == test_content + "\n"

    @pytest.mark.asyncio
    async def test_invalid_file_operations(self, action_executor):
        """Test error handling for invalid file operations"""
        # Test reading non-existent file
        read_action = FileReadAction(path="nonexistent.txt")
        obs = await action_executor.run_action(read_action)
        assert isinstance(obs, ErrorObservation)
        assert "File not found" in obs.error

    @pytest.mark.asyncio
    async def test_jupyter_plugin(self, action_executor):
        """Test Jupyter plugin functionality"""
        # Skip if Jupyter plugin not available
        if 'jupyter' not in action_executor.plugins:
            pytest.skip("Jupyter plugin not available")
            
        action = IPythonRunCellAction(code="print('Hello from Jupyter')")
        obs = await action_executor.run_action(action)
        assert isinstance(obs, IPythonRunCellObservation)
        assert "Hello from Jupyter" in obs.content

    def test_server_info(self, test_client):
        """Test server info endpoint"""
        response = test_client.get("/server_info")
        assert response.status_code == 200
        data = response.json()
        assert "uptime" in data
        assert "idle_time" in data

    def test_file_upload_download(self, test_client, temp_workspace):
        """Test file upload and download functionality"""
        # Test file upload
        test_content = b"test file content"
        files = {"file": ("test.txt", test_content)}
        response = test_client.post(
            "/upload_file",
            files=files,
            params={"destination": temp_workspace}
        )
        assert response.status_code == 200
        
        # Verify file was uploaded
        uploaded_file = Path(temp_workspace) / "test.txt"
        assert uploaded_file.exists()
        assert uploaded_file.read_bytes() == test_content

        # Test file download
        response = test_client.get(
            "/download_files",
            params={"path": temp_workspace}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_list_files(self, test_client, temp_workspace):
        """Test file listing functionality"""
        # Create some test files and directories
        os.makedirs(os.path.join(temp_workspace, "testdir"))
        Path(temp_workspace, "test1.txt").write_text("test1")
        Path(temp_workspace, "test2.txt").write_text("test2")
        
        response = test_client.post(
            "/list_files",
            json={"path": temp_workspace}
        )
        assert response.status_code == 200
        files = response.json()
        
        assert "testdir/" in files  # Directory should have trailing slash
        assert "test1.txt" in files
        assert "test2.txt" in files

    @pytest.mark.asyncio
    async def test_code_indexer(self, action_executor):
        """Test code indexer plugin functionality"""
        # Skip if code indexer plugin not available
        if 'code_indexer' not in action_executor.plugins:
            pytest.skip("Code indexer plugin not available")
            
        action = CodeIndexerAction(
            query="test query",
            codebase_path=temp_workspace,
            reindex_request=True
        )
        obs = await action_executor.run_action(action)
        assert not isinstance(obs, ErrorObservation)

    def test_authentication(self, test_client):
        """Test API key authentication"""
        # Test without API key
        response = test_client.post("/execute_action", json={"action": {}})
        assert response.status_code == 403
        
        # Test with invalid API key
        headers = {"X-Session-API-Key": "invalid_key"}
        response = test_client.post(
            "/execute_action",
            json={"action": {}},
            headers=headers
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_error_handling(self, action_executor):
        """Test error handling for invalid actions"""
        # Test invalid command
        action = CmdRunAction(command="invalid_command")
        obs = await action_executor.run_action(action)
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code != 0

        # Test timeout
        action = CmdRunAction(command="sleep 10")
        action.timeout = 1
        obs = await action_executor.run_action(action)
        assert isinstance(obs, CmdOutputObservation)
        assert "timeout" in obs.content.lower()

if __name__ == "__main__":
    pytest.main([__file__])