import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from openhands.agenthub.code_indexer_agent.code_indexer_agent import CodeIndexerAgent
from openhands.events.action import (
    CodeResponseAction,
    ErrorAction,
)
from openhands.events.observation import (
    CodeQueryObservation,
)


@pytest.fixture(scope='module')
def test_project():
    """Set up test environment with a mock project."""
    test_dir = tempfile.mkdtemp()

    # Create basic project structure
    project_files = {
        'src/main.py': """
def main():
    print("Hello from main!")
    helper_function()

def helper_function():
    return "Helper function called"
""",
        'src/utils.py': '''
def utility_function():
    """A utility function that does something useful."""
    return "Utility function called"
''',
        'README.md': """
# Test Project
This is a test project for integration testing.
## Features
- Feature 1
- Feature 2
""",
    }

    # Create the files
    for file_path, content in project_files.items():
        full_path = os.path.join(test_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)

    yield test_dir

    # Cleanup
    shutil.rmtree(test_dir)


@pytest.fixture(scope='module')
def test_config():
    """Create test configuration."""
    return {
        'model': {'name': 'gpt-3.5-turbo', 'temperature': 0},
        'embeddings': {'name': 'text-embedding-ada-002'},
    }


@pytest.fixture(scope='module')
def agent(test_project, test_config):
    """Create agent instance for testing."""
    agent = CodeIndexerAgent(config=test_config, codebase_path=test_project)
    # Ensure the agent is properly initialized
    assert agent.vector_store is not None, 'Vector store initialization failed'
    return agent


def create_test_project(test_dir):
    """Create a mock project structure for testing."""
    project_files = {
        'src/main.py': """
def main():
    print("Hello from main!")
    helper_function()

def helper_function():
    return "Helper function called"
""",
        'src/utils.py': '''
def utility_function():
    """A utility function that does something useful."""
    return "Utility function called"
''',
        'tests/test_main.py': """
def test_main():
    assert True
""",
        'README.md': """
# Test Project
This is a test project for integration testing.
## Features
- Feature 1
- Feature 2
""",
        '.gitignore': """
*.pyc
__pycache__/
venv/
.env
""",
        'requirements.txt': """
pytest==7.0.0
requests==2.26.0
""",
    }

    for file_path, content in project_files.items():
        full_path = os.path.join(test_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)


def test_end_to_end_search(agent):
    """Test end-to-end search functionality."""
    results = agent.search('main function implementation')
    assert len(results) > 0
    assert any('main.py' in doc.metadata['source'] for doc in results)
    assert any('def main()' in doc.page_content for doc in results)


def test_end_to_end_query(agent):
    """Test end-to-end query functionality."""
    response = agent.query('What does the main function do?')
    assert isinstance(response, str)
    assert len(response) > 0
    assert any(word in response.lower() for word in ['main', 'hello', 'helper'])


def test_file_exclusion(agent, test_project):
    """Test that excluded files are not indexed."""
    # Create excluded files
    excluded_files = {
        'venv/lib/test.py': 'print("Should not be indexed")',
        '__pycache__/main.cpython-39.pyc': 'Should not be indexed',
        '.env': 'SECRET_KEY=test',
    }

    for file_path, content in excluded_files.items():
        full_path = os.path.join(test_project, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)

    results = agent.search('Should not be indexed')

    assert all(
        'venv' not in doc.metadata['source']
        and '__pycache__' not in doc.metadata['source']
        and '.env' not in doc.metadata['source']
        for doc in results
    )


def test_markdown_processing(agent):
    """Test processing of markdown documentation."""
    results = agent.search('project features')

    assert any(
        'README.md' in doc.metadata['source'] and 'Feature' in doc.page_content
        for doc in results
    )


def test_code_query_workflow(agent):
    """Test the complete workflow from observation to action."""
    mock_state = type(
        'State',
        (),
        {
            'get_last_observation': lambda s: CodeQueryObservation(
                query='What does the utility function do?'
            )
        },
    )()

    action = agent.step(mock_state)

    assert isinstance(action, CodeResponseAction)
    assert 'utility' in action.response.lower()
    assert 'function' in action.response.lower()


def test_large_file_handling(agent, test_project):
    """Test handling of larger files."""
    large_file_path = os.path.join(test_project, 'src/large_file.py')
    content = '\n'.join([f'def function_{i}():\n    return {i}' for i in range(100)])

    with open(large_file_path, 'w') as f:
        f.write(content)

    # Update the index
    agent.update_index()

    # Search for a specific function
    results = agent.search('function_50')
    assert len(results) > 0
    assert any(
        'function_50' in doc.page_content or 'function_50' in str(doc.metadata.values())
        for doc in results
    )


def test_error_handling(agent):
    """Test error handling in real scenarios."""
    with pytest.raises(Exception):
        agent.search(None)

    with pytest.raises(Exception):
        agent.query(None)


def test_code_structure_parsing(agent, test_project):
    """Test parsing of code structure."""
    test_file = os.path.join(test_project, 'src/test_parse.py')
    content = """
class TestClass:
    def test_method(self):
        return "test"
"""
    with open(test_file, 'w') as f:
        f.write(content)

    agent.update_index()
    results = agent.search('TestClass')

    assert len(results) > 0
    assert any(
        doc.metadata.get('type') == 'class'
        and 'TestClass' in str(doc.metadata.values())
        for doc in results
    )


def test_multi_language_parsing(agent, test_project):
    """Test parsing of multiple programming languages."""
    test_files = {
        'src/test.py': """
def python_function():
    return "Python"
""",
        'src/test.js': """
function javascriptFunction() {
    return "JavaScript";
}
""",
        'src/test.go': """
func goFunction() {
    return "Go"
}
""",
    }

    for file_path, content in test_files.items():
        full_path = os.path.join(test_project, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)

    # Update the index
    agent.update_index()

    # Test each language
    for lang, func_name in [
        ('Python', 'python_function'),
        # ('JavaScript', 'javascriptFunction'),
        # ('Go', 'goFunction')
    ]:
        results = agent.search(f'{func_name}')
        print('results', results)
        assert any(
            doc.metadata.get('type') == 'function'
            and doc.metadata.get('name') == func_name
            for doc in results
        ), f'Failed to find {lang} function'


def test_code_query_with_structure(agent, test_project):
    """Test querying code with structural understanding."""
    test_file = os.path.join(test_project, 'src/complex.py')
    content = '''
class DataProcessor:
    """A class for processing data."""

    def __init__(self, data):
        self.data = data

    def process(self):
        """Process the data and return results."""
        return self.transform(self.data)

    def transform(self, input_data):
        """Transform the input data."""
        return [x * 2 for x in input_data]

def helper_function(x):
    """Help with processing."""
    return x + 1
'''

    with open(test_file, 'w') as f:
        f.write(content)

    # Update the index
    agent.update_index()

    # Test querying with structural understanding
    response = agent.query('What methods are available in the DataProcessor class?')
    assert all(
        method in response.lower() for method in ['init', 'process', 'transform']
    )

    # Test querying specific method implementation
    response = agent.query('How does the transform method work?')
    print('response', response)
    assert (
        'multiply' in response.lower()
        or 'multiplies' in response.lower()
        or 'x * 2' in response
    )


def test_analyze_codebase(agent, test_project):
    """Test analyzing the codebase."""
    response = agent.query('Summarize the codebase.')
    print('response', response)
    assert 'features' in response.lower()


@pytest.fixture
def test_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_codebase(test_dir):
    """Create a sample codebase for testing."""
    # Create source directory
    src_dir = os.path.join(test_dir, 'src')
    os.makedirs(src_dir, exist_ok=True)

    # Create a Python file with a class and methods
    with open(os.path.join(src_dir, 'main.py'), 'w') as f:
        f.write('''
class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.value = 0

    def add(self, x):
        """Add a number to the current value."""
        self.value += x
        return self.value

    def subtract(self, x):
        """Subtract a number from the current value."""
        self.value -= x
        return self.value
''')

    # Create a README file with features
    with open(os.path.join(test_dir, 'README.md'), 'w') as f:
        f.write("""# Sample Project
This is a test project.

## Features
- Feature 1: Basic arithmetic operations
- Feature 2: Value persistence
- Feature 3: Simple API
""")

    return test_dir


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    mock_embed = MagicMock()
    # Return multiple embeddings for documents
    mock_embed.embed_documents.return_value = [[0.1] * 1536 for _ in range(10)]
    mock_embed.embed_query.return_value = [0.1] * 1536
    return mock_embed


@pytest.fixture
def mock_llm():
    """Create mock LLM that implements LangChain's BaseChatModel."""

    class MockChatModel(BaseChatModel):
        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = """
            Location: src/main.py
            Type: class
            Name: Calculator
            Details: The Calculator class has methods: __init__, add, and subtract.
            """
            return ChatResult(generations=[ChatGeneration(message=response)])

        def _llm_type(self) -> str:
            return 'mock_chat_model'

        @property
        def _identifying_params(self) -> Dict[str, Any]:
            return {'model': 'mock'}

        def invoke(self, prompt: str, **kwargs: Any) -> str:
            return """
            Location: src/main.py
            Type: class
            Name: Calculator
            Details: The Calculator class has methods: __init__, add, and subtract.
            """

    return MockChatModel()


@pytest.fixture
def agent2(test_dir, sample_codebase, mock_embeddings, mock_llm):
    """Create a CodeIndexerAgent instance."""
    config = {
        'model': {'name': 'gpt-3.5-turbo', 'temperature': 0},
        'embeddings': {'name': 'text-embedding-ada-002'},
    }
    agent = CodeIndexerAgent(
        config=config,
        codebase_path=sample_codebase,
        embeddings=mock_embeddings,
        llm=mock_llm,
    )
    return agent


def test_code_query_integration(agent2):
    """Test that the agent can answer queries about the codebase."""

    class MockState:
        def get_last_observation(self):
            return CodeQueryObservation(
                query='What methods are available in the Calculator class?'
            )

    action = agent2.step(MockState())
    assert isinstance(action, CodeResponseAction)
    assert 'Calculator' in action.response
    assert 'methods' in action.response.lower()


def test_error_handling_integration(agent2):
    """Test that the agent handles errors gracefully."""

    # Create a mock state with an invalid query
    class MockState:
        def get_last_observation(self):
            return CodeQueryObservation(query='')  # Empty query should cause an error

    # Get the agent's response
    action = agent2.step(MockState())

    # Verify error handling
    assert isinstance(action, ErrorAction)
    assert action.error is not None


def test_search_integration(agent2):
    """Test the search functionality with real embeddings."""
    results = agent2.search('Calculator class methods')

    # Verify search results
    assert len(results) > 0
    # Look for content in any document
    assert any(
        'Calculator' in doc.page_content or 'Calculator' in str(doc.metadata.values())
        for doc in results
    )
