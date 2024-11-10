# Requirements
from openhands.runtime.plugins.agent_skills import (
    AgentSkillsPlugin,
    AgentSkillsRequirement,
)
from openhands.runtime.plugins.jupyter import JupyterPlugin, JupyterRequirement
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement
from openhands.runtime.plugins.code_indexer import CodeIndexerPlugin, CodeIndexerRequirement

__all__ = [
    'Plugin',
    'PluginRequirement',
    'AgentSkillsRequirement',
    'AgentSkillsPlugin',
    'JupyterRequirement',
    'JupyterPlugin',
    'CodeIndexerPlugin',
    'CodeIndexerRequirement',
]

ALL_PLUGINS = {
    'jupyter': JupyterPlugin,
    'agent_skills': AgentSkillsPlugin,
    'code_indexer': CodeIndexerPlugin,
}
