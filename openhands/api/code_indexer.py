from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class CodeQuery(BaseModel):
    query: str
    k: int = 4


class SearchResult(BaseModel):
    content: str
    source: str
    relevance: float


@router.post('/search')
async def search_code(request: CodeQuery):
    """Search the codebase for relevant code snippets."""
    try:
        results = agent.search(request.query, k=request.k)
        return {
            'results': [
                {
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'relevance': doc.metadata.get('score', 0.0),
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/query')
async def query_code(request: CodeQuery):
    """Query the codebase using natural language."""
    try:
        response = agent.query(request.query)
        return {'response': response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
