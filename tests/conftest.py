import os
import shutil
import tempfile
from unittest.mock import patch

import pytest


@pytest.fixture(scope='session')
def test_project_dir():
    """Create a temporary test project directory."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture(scope='session')
def test_config():
    """Provide test configuration."""
    return {
        'name': 'code_indexer',
        'description': 'Test agent',
        'model': 'gpt-4o',
        'temperature': 0,
        'llm': {'api_key': os.environ['OPENAI_API_KEY'], 'model': 'gpt-4o'},
    }


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')


@pytest.fixture
def mock_embeddings():
    """Provide mock embeddings."""
    with patch('langchain_openai.OpenAIEmbeddings') as mock:
        yield mock


@pytest.fixture
def mock_llm():
    """Provide mock LLM."""
    with patch('langchain_openai.ChatOpenAI') as mock:
        yield mock


@pytest.fixture
def mock_vector_store():
    """Provide mock vector store."""
    with patch('langchain_community.vectorstores.FAISS') as mock:
        yield mock
