from dataclasses import dataclass
from openhands.runtime.plugins.requirement import PluginRequirement

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