from typing import List

from openhands.events.action.action import Action, ActionConfirmationStatus
from openhands.events.action.agent import (
    AgentDelegateAction,
    AgentFinishAction,
    AgentRejectAction,
    AgentSummarizeAction,
    ChangeAgentStateAction,
)
from openhands.events.action.browse import BrowseInteractiveAction, BrowseURLAction
from openhands.events.action.commands import CmdRunAction, IPythonRunCellAction
from openhands.events.action.empty import NullAction
from openhands.events.action.files import (
    FileEditAction,
    FileReadAction,
    FileWriteAction,
)
from openhands.events.action.message import (
    CodeResponseAction,
    ErrorAction,
    MessageAction,
    NoOpAction,
)
from openhands.events.action.tasks import AddTaskAction, ModifyTaskAction
from openhands.events.action.indexer import CodeIndexerAction


# TODO: edit me
class BashCommandAction(Action):
    """Action for running a bash command."""

    def __init__(self, command: str):
        self.command = command


class MultiAction(Action):
    """Action for running multiple actions."""

    def __init__(self, actions: List[Action]):
        self.actions = actions


__all__ = [
    'Action',
    'NullAction',
    'CmdRunAction',
    'BrowseURLAction',
    'BrowseInteractiveAction',
    'FileReadAction',
    'FileWriteAction',
    'FileEditAction',
    'AgentFinishAction',
    'AgentRejectAction',
    'AgentDelegateAction',
    'AgentSummarizeAction',
    'AddTaskAction',
    'ModifyTaskAction',
    'ChangeAgentStateAction',
    'IPythonRunCellAction',
    'MessageAction',
    'ActionConfirmationStatus',
    'CodeResponseAction',
    'ErrorAction',
    'NoOpAction',
    'BashCommandAction',
    'MultiAction',
    'CodeIndexerAction'
]
