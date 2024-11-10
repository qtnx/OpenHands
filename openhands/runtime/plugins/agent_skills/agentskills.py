from inspect import signature
from typing import List, Dict, Any

from openhands.runtime.plugins.agent_skills import file_ops, file_reader
from openhands.runtime.plugins.agent_skills.utils.dependency import import_functions

import_functions(
    module=file_ops, function_names=file_ops.__all__, target_globals=globals()
)
import_functions(
    module=file_reader, function_names=file_reader.__all__, target_globals=globals()
)
__all__ = file_ops.__all__ + file_reader.__all__

DOCUMENTATION = ''
for func_name in __all__:
    func = globals()[func_name]

    cur_doc = func.__doc__
    # remove indentation from docstring and extra empty lines
    cur_doc = '\n'.join(filter(None, map(lambda x: x.strip(), cur_doc.split('\n'))))
    # now add a consistent 4 indentation
    cur_doc = '\n'.join(map(lambda x: ' ' * 4 + x, cur_doc.split('\n')))

    fn_signature = f'{func.__name__}' + str(signature(func))
    DOCUMENTATION += f'{fn_signature}:\n{cur_doc}\n\n'


# Add file_editor (a function)
from openhands.runtime.plugins.agent_skills.file_editor import file_editor  # noqa: E402

__all__ += ['file_editor']


def explore_codebase(query: str, codebase_path: str = ".") -> str:
    """
    Query the codebase for information using natural language.
    
    Args:
        query: The natural language query about the codebase
        codebase_path: Optional path to the codebase root
        
    Returns:
        str: Formatted response with relevant code information
    """
    try:
        from openhands.runtime.plugins.code_indexer import CodeIndexerPlugin
        indexer = CodeIndexerPlugin.get_instance()
        return indexer.query(query)
    except Exception as e:
        return f"Error querying codebase: {str(e)}"

def search_code(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the codebase for relevant code snippets.
    
    Args:
        query: Search query
        k: Number of results to return
        
    Returns:
        List[Dict]: List of relevant code snippets with metadata
    """
    try:
        from openhands.runtime.plugins.code_indexer import CodeIndexerPlugin
        indexer = CodeIndexerPlugin.get_instance()
        results = indexer.search(query, k=k)
        return [
            {
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
                'relevance': doc.metadata.get('relevance', 0.0)
            }
            for doc in results
        ]
    except Exception as e:
        return [{'error': str(e)}]
