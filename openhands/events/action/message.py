from dataclasses import dataclass

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class MessageAction(Action):
    content: str
    images_urls: list[str] | None = None
    wait_for_response: bool = False
    action: str = ActionType.MESSAGE
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return self.content

    def __str__(self) -> str:
        ret = f'**MessageAction** (source={self.source}) (cause={
            self.cause})\n'
        ret += f'CONTENT: {self.content}'
        if self.images_urls:
            for url in self.images_urls:
                ret += f'\nIMAGE_URL: {url}'
        return ret


class CodeResponseAction(Action):
    """Action containing response to a code query."""

    def __init__(self, response: str):
        self.response = response


class ErrorAction(Action):
    """Action indicating an error occurred."""

    def __init__(self, error: str):
        self.error = error


class NoOpAction(Action):
    """Action indicating no operation needed."""

    pass
