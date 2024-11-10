from typing import Any, Dict, Optional, ClassVar, List
from dataclasses import dataclass

from langchain.schema import Document
# from openhands.agenthub.code_indexer_agent.code_indexer_agent import CodeIndexerAgent

from openhands.events.action import Action, CodeResponseAction, ErrorAction
from openhands.events.observation import CodeQueryObservation, ErrorObservation, Observation
from openhands.runtime.plugins.code_indexer.code_indexer_agent import CodeIndexerAgent
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement


@dataclass
class CodeIndexerRequirement(PluginRequirement):
    """Requirement for code indexing capabilities."""
    name: str = 'code_indexer'
    documentation: str = """
    Code Indexer Plugin provides capabilities for:
    - Indexing and searching codebases
    - Answering queries about code structure and functionality
    - Understanding relationships between code components
    
    Usage:
    - Query codebase: explore_codebase("What does class X do?")
    - Search code: search_code("function name", k=5)
    - Update index: update_code_index()
    """


class CodeIndexerPlugin(Plugin):
    """Plugin for code indexing and querying capabilities."""
    
    _instance: ClassVar[Optional['CodeIndexerPlugin']] = None
    name: str = 'code_indexer'

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the code indexer plugin."""
        super().__init__()
        self.config = config or self.config
        self.agent: Optional[CodeIndexerAgent] = None
        print(f"CodeIndexerPlugin config: {self.config}")
        self.codebase_path = "/workspace" #self.config.get('codebase_path', '.')

    @classmethod
    def get_instance(cls) -> 'CodeIndexerPlugin':
        """Get singleton instance of the plugin."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self, username: str) -> None:
        """Initialize the code indexer agent."""
        try:
            self.agent = CodeIndexerAgent(
                config=self.config,
                codebase_path=self.codebase_path,
            )
            self.agent.initialize_rag()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CodeIndexerAgent: {str(e)}")

    async def run(self, action: Action) -> Observation:
        """Execute code indexing related actions."""
        if not self.agent:
            return ErrorObservation(error_id="CodeIndexerAgent not initialized")

        if isinstance(action, CodeQueryObservation):
            try:
                response = self.agent.query(action.query)
                return CodeResponseAction(response=response)
            except Exception as e:
                return ErrorObservation(error_id=f"Query execution failed: {str(e)}")

        return ErrorObservation(error_id=f"Unsupported action type: {type(action)}")

    def query(self, query: str) -> str:
        """Query the codebase."""
        if not self.agent:
            raise RuntimeError("CodeIndexerAgent not initialized")
        return self.agent.query(query)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the codebase."""
        if not self.agent:
            raise RuntimeError("CodeIndexerAgent not initialized")
        return self.agent.search(query)

    async def initialize_rag(self) -> None:
        """Initialize the RAG system."""
        if self.agent:
            self.agent.initialize_rag()

    async def update_index(self) -> None:
        """Update the index."""
        if self.agent:
            self.agent.initialize_rag()

    def cleanup(self) -> None:
        """Clean up resources."""
        self.agent = None
        type(self)._instance = None


__all__ = ['CodeIndexerPlugin', 'CodeIndexerRequirement']
