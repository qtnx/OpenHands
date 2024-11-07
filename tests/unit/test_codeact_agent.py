from unittest.mock import Mock

import pytest

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.message import TextContent
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.observation.commands import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.observation.error import ErrorObservation
from openhands.llm.llm import LLM


@pytest.fixture
def mock_state():
    state = Mock(spec=State)
    state.max_iterations = 10
    state.iteration = 0
    return state


@pytest.fixture
def agent() -> CodeActAgent:
    agent = CodeActAgent(llm=LLM(LLMConfig()), config=AgentConfig())
    agent.llm = Mock()
    agent.llm.config = Mock()
    agent.llm.config.max_message_chars = 100
    return agent


def test_cmd_output_observation_message(agent: CodeActAgent):
    agent.config.function_calling = False
    obs = CmdOutputObservation(
        command='echo hello', content='Command output', command_id=1, exit_code=0
    )

    results = agent.get_observation_message(obs, tool_call_id_to_message={})
    assert len(results) == 1

    result = results[0]
    assert result is not None
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'OBSERVATION:' in result.content[0].text
    assert 'Command output' in result.content[0].text
    assert 'Command finished with exit code 0' in result.content[0].text


def test_ipython_run_cell_observation_message(agent: CodeActAgent):
    agent.config.function_calling = False
    obs = IPythonRunCellObservation(
        code='plt.plot()',
        content='IPython output\n![image](data:image/png;base64,ABC123)',
    )

    results = agent.get_observation_message(obs, tool_call_id_to_message={})
    assert len(results) == 1

    result = results[0]
    assert result is not None
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'OBSERVATION:' in result.content[0].text
    assert 'IPython output' in result.content[0].text
    assert (
        '![image](data:image/png;base64, ...) already displayed to user'
        in result.content[0].text
    )
    assert 'ABC123' not in result.content[0].text


def test_agent_delegate_observation_message(agent: CodeActAgent):
    agent.config.function_calling = False
    obs = AgentDelegateObservation(
        content='Content', outputs={'content': 'Delegated agent output'}
    )

    results = agent.get_observation_message(obs, tool_call_id_to_message={})
    assert len(results) == 1

    result = results[0]
    assert result is not None
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'OBSERVATION:' in result.content[0].text
    assert 'Delegated agent output' in result.content[0].text


def test_error_observation_message(agent: CodeActAgent):
    agent.config.function_calling = False
    obs = ErrorObservation('Error message')

    results = agent.get_observation_message(obs, tool_call_id_to_message={})
    assert len(results) == 1

    result = results[0]
    assert result is not None
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'OBSERVATION:' in result.content[0].text
    assert 'Error message' in result.content[0].text
    assert 'Error occurred in processing last action' in result.content[0].text


def test_unknown_observation_message(agent: CodeActAgent):
    obs = Mock()

    with pytest.raises(ValueError, match='Unknown observation type'):
        agent.get_observation_message(obs, tool_call_id_to_message={})


def test_initial_plan_creation(agent, mock_state):
    # Mock LLM response
    mock_plan = 'Test execution plan'
    agent.llm.completion.return_value.choices = [Mock(message=Mock(content=mock_plan))]

    # Test initial plan creation
    action = agent.step(mock_state)

    assert isinstance(action, MessageAction)
    assert mock_plan in action.content
    assert 'approve' in action.content.lower()
    assert 'modify' in action.content.lower()
    assert 'reject' in action.content.lower()
    assert action.wait_for_response is True
    assert agent.waiting_for_feedback is True


def test_plan_approval(agent, mock_state):
    # Setup initial state
    agent.plan = 'Test plan'
    agent.waiting_for_feedback = True
    mock_state.get_last_user_message.return_value = 'approve'

    # Test plan approval
    action = agent.step(mock_state)

    assert isinstance(action, MessageAction)
    assert 'approved' in action.content.lower()
    assert agent.plan_approved is True
    assert agent.waiting_for_feedback is False


def test_plan_rejection(agent, mock_state):
    # Setup initial state
    agent.plan = 'Test plan'
    agent.waiting_for_feedback = True
    mock_state.get_last_user_message.return_value = 'reject'

    # Test plan rejection
    action = agent.step(mock_state)

    assert isinstance(action, AgentFinishAction)
    assert agent.plan is None
    assert agent.waiting_for_feedback is False


def test_plan_modification(agent, mock_state):
    # Setup initial state
    agent.plan = 'Test plan'
    agent.waiting_for_feedback = True
    mock_state.get_last_user_message.return_value = 'modify make it more detailed'

    # Mock LLM response for new plan
    mock_new_plan = 'More detailed test plan'
    agent.llm.completion.return_value.choices = [
        Mock(message=Mock(content=mock_new_plan))
    ]

    # Test plan modification
    action = agent.step(mock_state)

    assert isinstance(action, MessageAction)
    assert mock_new_plan in action.content
    assert 'approve' in action.content.lower()
    assert 'modify' in action.content.lower()
    assert agent.waiting_for_feedback is True
    assert agent.plan == mock_new_plan


def test_waiting_for_feedback(agent, mock_state):
    # Setup initial state
    agent.waiting_for_feedback = True
    mock_state.get_last_user_message.return_value = None

    # Test waiting state
    action = agent.step(mock_state)

    assert isinstance(action, MessageAction)
    assert 'waiting' in action.content.lower()
    assert action.wait_for_response is True
    assert agent.waiting_for_feedback is True


def test_reset_agent(agent):
    # Setup initial state
    agent.plan = 'Test plan'
    agent.plan_approved = True
    agent.waiting_for_feedback = True

    # Test reset
    agent.reset()

    assert agent.plan is None
    assert agent.plan_approved is False
    assert agent.waiting_for_feedback is False
