# Add these classes to the existing observation.py file

from dataclasses import dataclass
from openhands.core.schema.observation import ObservationType
from openhands.events.observation.observation import Observation


class CodeQueryObservation(Observation):
    """Observation containing a query about the codebase."""

    def __init__(self, query: str):
        self.query = query



@dataclass
class CodeQueryObservationRes(Observation):
    """Observation containing a query about the codebase."""

    observation: str = ObservationType.CODE_QUERY
    error: bool = False
    query: str = ''
    content: str = ''

    @property
    def message(self) -> str:
        return f'Code query: {self.query}\nResponse Content: {self.content}'

    def __str__(self) -> str:
        return f'**CodeQueryObservationRes (source={self.source}, error={self.error})**\n{self.content}'

class CommandOutputObservation(Observation):
    """Observation containing output from a command execution."""

    def __init__(self, output: str):
        self.output = output


class FileContentObservation(Observation):
    """Observation containing the content of a file."""

    def __init__(self, content: str, filepath: str):
        self.content = content
        self.filepath = filepath
