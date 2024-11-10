from evaluation.integration_tests.tests.base import BaseIntegrationTest, TestResult
from evaluation.utils.shared import assert_and_raise
from openhands.events.action import (
    CmdRunAction,
    MessageAction,
    AgentFinishAction,
    CodeResponseAction,
)
from openhands.events.event import Event
from openhands.events.observation import AgentDelegateObservation
from openhands.runtime.base import Runtime
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.indexer import CodeIndexerAction
import time


class Test(BaseIntegrationTest):
    INSTRUCTION = "Analyze the code in /workspace/sample and tell me what the Calculator class does using `explore_codebase` function."

    @classmethod
    def initialize_runtime(cls, runtime: Runtime) -> None:
        # Create workspace and sample directory
        action = CmdRunAction(
            command='mkdir -p /workspace/sample', keep_prompt=False)
        obs = runtime.run_action(action)
        assert_and_raise(obs.exit_code == 0,
                         f'Failed to create directory: {obs.content}')

        # Create a sample Python file with a Calculator class
        calculator_code = '''
class Calculator:
    """A simple calculator class for basic arithmetic operations."""
    
    def __init__(self):
        self.value = 0
        self.history = []
    
    def add(self, x):
        """Add a number to the current value."""
        self.value += x
        self.history.append(f"Added {x}")
        return self.value
    
    def subtract(self, x):
        """Subtract a number from the current value."""
        self.value -= x
        self.history.append(f"Subtracted {x}")
        return self.value
        
    def get_history(self):
        """Return the history of operations."""
        return self.history
'''
        action = CmdRunAction(
            command=f'cat > /workspace/sample/calculator.py << EOL\n{
                calculator_code}\nEOL',
            keep_prompt=False,
        )
        obs = runtime.run_action(action)

        assert_and_raise(obs.exit_code == 0,
                         f'Failed to create calculator.py: {obs.content}')

        # Reindex the codebase

        # Sleep briefly before reindexing to ensure file is written
        # time.sleep(2)

        action = CodeIndexerAction(
            query='calculator.py',
            reindex_request=True,
        )
        obs = runtime.run_action(action)
        assert_and_raise(not obs.error,
                         f'Failed to reindex codebase: {obs.content}')

    @classmethod
    def verify_result(cls, runtime: Runtime, histories: list[Event]) -> TestResult:
        logger.debug(f'Histories: {[type(h).__name__ for h in histories]}')
        # Check for relevant responses in the message history
        message_actions = []
        for event in histories:
            # Convert dict to appropriate Event subclass if needed
            if isinstance(event, dict):
                if 'action' in event and event['action'] == 'finish':
                    message_actions.append(event)
                continue
            if isinstance(event, (MessageAction, AgentFinishAction, AgentDelegateObservation)):
                message_actions.append(event)

        # Define key concepts that should be mentioned in the analysis
        key_concepts = [
            'calculator',
            'arithmetic',
            'add',
            'subtract',
            'history',
            'operations',
        ]

        # Check each message for the key concepts
        for event in message_actions:
            logger.debug(f'EVENT: {event}')
            if isinstance(event, AgentDelegateObservation):
                content = event.content
            elif isinstance(event, AgentFinishAction):
                content = event.outputs.get('content', '')
            elif isinstance(event, MessageAction):
                content = event.content
            elif isinstance(event, CodeResponseAction):
                content = event.response
            else:
                if isinstance(event, dict):
                    if 'message' in event:
                        content = event['message']
                    else:
                        content = event.get('content', '')
                else:
                    continue

            content = content.lower()
            logger.debug(f'CONTENT response: {content}')
            found_concepts = [
                concept for concept in key_concepts if concept in content]

            # If we find at least 4 key concepts, consider it a success
            if len(found_concepts) >= 4:
                return TestResult(success=True)

        return TestResult(
            success=False,
            reason=(
                f'The analysis does not sufficiently describe the Calculator class. '
                f'Total messages: {len(message_actions)}. '
                f'Messages: {message_actions}'
            ),
        )
