# Add these classes to the existing observation.py file

from openhands.events.observation.observation import Observation


class CodeQueryObservation(Observation):
    """Observation containing a query about the codebase."""

    def __init__(self, query: str):
        self.query = query


class CommandOutputObservation(Observation):
    """Observation containing output from a command execution."""

    def __init__(self, output: str):
        self.output = output


class FileContentObservation(Observation):
    """Observation containing the content of a file."""

    def __init__(self, content: str, filepath: str):
        self.content = content
        self.filepath = filepath
