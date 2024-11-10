from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

from openhands.events.action import Action
from openhands.events.observation import Observation


class Plugin:
    """Base class for plugins."""
    name: str

    """Plugin-specific configuration."""
    config: Dict[str, Any] = field(default_factory=dict)

    async def initialize(self, username: str) -> None:
        """Initialize the plugin."""
        pass

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass

    @abstractmethod
    async def run(self, action: Action) -> Observation:
        """Run the plugin for a given action."""
        pass


@dataclass
class PluginRequirement:
    """Requirement for a plugin."""

    name: str
