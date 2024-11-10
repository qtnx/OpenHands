from dataclasses import dataclass
from typing import ClassVar
from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class CodeIndexerAction(Action):
    """Action for code indexing operations."""
    query: str
    codebase_path: str = "/workspace"
    reindex_request: bool = False
    action: str = ActionType.CODE_INDEXER
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'Exploring codebase with query: {self.query}'

    def __str__(self) -> str:
        return f'**CodeIndexerAction**\nQUERY: {self.query}\nCODEBASE_PATH: {self.codebase_path}'

    def __repr__(self) -> str:
        return self.__str__()