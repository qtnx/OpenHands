import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from openhands.agenthub.code_indexer_agent.code_indexer_agent import CodeIndexerAgent
from openhands.events.action import (
    CodeResponseAction,
    ErrorAction,
    NoOpAction,
)
from openhands.events.observation import (
    CodeQueryObservation,
)


@pytest.fixture
def test_dir(request):
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    request.addfinalizer(lambda: shutil.rmtree(temp_dir))
    return temp_dir


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.invoke.return_value.content = 'Test response'
    return mock


@pytest.fixture
def mock_embeddings():
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.as_retriever.return_value = MagicMock()
    return mock


@pytest.fixture
def test_config():
    return {'model': 'gpt-3.5-turbo', 'temperature': 0, 'max_tokens': 1000}


@pytest.fixture
def agent(test_dir, test_config, mock_llm, mock_embeddings, mock_vector_store):
    """Create a test agent with mocked dependencies."""
    with patch(
        'langchain_community.document_loaders.DirectoryLoader'
    ) as mock_loader, patch('langchain_community.vectorstores.Chroma') as mock_chroma:
        # Setup mock loader
        mock_loader.return_value.load.return_value = [
            MagicMock(page_content='test content', metadata={'source': 'test.py'})
        ]

        # Setup mock Chroma
        mock_chroma.from_documents.return_value = mock_vector_store

        # Create agent
        agent = CodeIndexerAgent(
            config=test_config,
            codebase_path=test_dir,
            llm=mock_llm,
            embeddings=mock_embeddings,
        )

        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(test_dir)

        yield agent

        # Cleanup
        os.chdir(original_dir)


def create_test_files(test_dir):
    """Create mock files for testing."""
    # Create .gitignore
    with open(os.path.join(test_dir, '.gitignore'), 'w') as f:
        f.write('*.pyc\nvenv/\n.env\n')

    # Create Python file
    with open(os.path.join(test_dir, 'test.py'), 'w') as f:
        f.write('def hello():\n    print("Hello, World!")\n')

    # Create markdown file
    with open(os.path.join(test_dir, 'README.md'), 'w') as f:
        f.write('# Test Project\nThis is a test project.\n')

    # Create excluded files
    os.makedirs(os.path.join(test_dir, 'venv'))
    with open(os.path.join(test_dir, 'venv', 'test.txt'), 'w') as f:
        f.write('This should be excluded\n')


def test_init(agent):
    """Test agent initialization."""
    assert agent.vector_store is not None
    assert agent.llm is not None
    assert agent.embeddings is not None


def test_get_gitignore_patterns(agent, test_dir):
    """Test gitignore pattern parsing."""
    create_test_files(test_dir)
    patterns = agent._get_gitignore_patterns()
    assert '**/*.pyc' in patterns
    assert '**/venv/**' in patterns
    assert '**/.env' in patterns


def test_file_patterns(agent):
    """Test file pattern definitions."""
    assert '*.py' in agent.FILE_PATTERNS
    assert '*.md' in agent.FILE_PATTERNS
    assert '*.json' in agent.FILE_PATTERNS


def test_exclude_patterns(agent):
    """Test exclude pattern definitions."""
    assert '**/venv/**' in agent.EXCLUDE_PATTERNS
    assert '**/__pycache__/**' in agent.EXCLUDE_PATTERNS
    assert '**/.git/**' in agent.EXCLUDE_PATTERNS


def test_search(agent):
    """Test search functionality."""
    # Mock vector store search
    mock_docs = [MagicMock(page_content='test content', metadata={'source': 'test.py'})]
    agent.vector_store.similarity_search.return_value = mock_docs

    # Perform search
    results = agent.search('test query')

    # Verify search was called
    agent.vector_store.similarity_search.assert_called_once_with('test query', k=4)
    assert len(results) == 1


def test_query(agent):
    """Test query functionality."""
    # Mock retrieval chain
    agent.retrieval_chain.invoke.return_value = 'Test response'

    # Perform query
    response = agent.query('test question')

    # Verify query was called
    agent.retrieval_chain.invoke.assert_called_once_with('test question')
    assert response == 'Test response'


def test_step_with_code_query(agent):
    """Test step method with code query observation."""
    # Mock state
    mock_state = Mock()
    mock_state.get_last_observation.return_value = CodeQueryObservation(
        query='test query'
    )

    # Mock query response
    formatted_response = """
    Location: test.py
    Type: class
    Name: TestClass
    Details: This is a test response
    """
    agent.query = Mock(return_value=formatted_response)

    # Execute step
    action = agent.step(mock_state)

    # Verify response
    assert isinstance(action, CodeResponseAction)
    assert action.response == formatted_response


def test_step_with_error(agent):
    """Test step method with error handling."""
    # Mock state
    mock_state = Mock()
    mock_state.get_last_observation.return_value = CodeQueryObservation(
        query='test query'
    )

    # Mock query to raise exception
    agent.query = Mock(side_effect=Exception('Test error'))

    # Execute step
    action = agent.step(mock_state)

    # Verify error response
    assert isinstance(action, ErrorAction)
    assert action.error == 'Test error'


def test_step_with_invalid_observation(agent):
    """Test step method with invalid observation."""
    # Mock state with invalid observation
    mock_state = Mock()
    mock_state.get_last_observation.return_value = Mock()

    # Execute step
    action = agent.step(mock_state)

    # Verify no-op response
    assert isinstance(action, NoOpAction)
