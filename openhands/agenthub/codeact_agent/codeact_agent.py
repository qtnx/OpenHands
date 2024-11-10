import json
import os
import traceback
from collections import deque
from itertools import islice
from typing import Optional, Union, List, cast

from litellm import ModelResponse

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.action_parser import CodeActResponseParser
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.observation import (
    AgentDelegateObservation,
    BrowserOutputObservation,
    CmdOutputObservation,
    FileEditObservation,
    IPythonRunCellObservation,
    UserRejectObservation,
)
from openhands.events.observation.code import CodeQueryObservationRes
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.microagent import MicroAgent
from openhands.utils.prompt import PromptManager
from openhands.events.action.indexer import CodeIndexerAction

PLAN_CAUSE_CODE = 100


class CodeActAgent(Agent):
    VERSION = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents’ **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]
    obs_prefix = 'OBSERVATION:\n'

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm, config)
        self.reset()

        self.micro_agent = (
            MicroAgent(
                os.path.join(
                    os.path.dirname(__file__), 'micro', f'{config.micro_agent_name}.md'
                )
            )
            if config.micro_agent_name
            else None
        )

        self.function_calling_active = self.config.function_calling
        if self.function_calling_active and not self.llm.is_function_calling_active():
            logger.warning(
                f'Function calling not supported for model {self.llm.config.model}. '
                'Disabling function calling.'
            )
            self.function_calling_active = False

        if self.function_calling_active:
            # Function calling mode
            self.tools = codeact_function_calling.get_tools(
                codeact_enable_browsing=self.config.codeact_enable_browsing,
                codeact_enable_jupyter=self.config.codeact_enable_jupyter,
                codeact_enable_llm_editor=self.config.codeact_enable_llm_editor,
            )
            logger.debug(
                f'TOOLS loaded for CodeActAgent: {json.dumps(self.tools, indent=2)}'
            )
            self.system_prompt = codeact_function_calling.SYSTEM_PROMPT
            self.initial_user_message = None
        else:
            # Non-function-calling mode
            self.action_parser = CodeActResponseParser()
            self.prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__)),
                agent_skills_docs=AgentSkillsRequirement.documentation,
                micro_agent=self.micro_agent,
            )
            self.system_prompt = self.prompt_manager.system_message
            self.initial_user_message = self.prompt_manager.initial_user_message

        self.pending_actions: deque[Action] = deque()
        self.plan_approved = False  # Track if plan has been approved
        self.plan = None  # Store the execution plan
        self.waiting_for_feedback = False
        self.current_step = None

    def get_action_message(
        self,
        action: Action,
        pending_tool_call_action_messages: dict[str, Message],
    ) -> list[Message]:
        """Converts an action into a message format that can be sent to the LLM.

        This method handles different types of actions and formats them appropriately:
        1. For tool-based actions (AgentDelegate, CmdRun, IPythonRunCell, FileEdit) and agent-sourced AgentFinish:
            - In function calling mode: Stores the LLM's response in pending_tool_call_action_messages
            - In non-function calling mode: Creates a message with the action string
        2. For MessageActions: Creates a message with the text content and optional image content

        Args:
            action (Action): The action to convert. Can be one of:
                - CmdRunAction: For executing bash commands
                - IPythonRunCellAction: For running IPython code
                - FileEditAction: For editing files
                - BrowseInteractiveAction: For browsing the web
                - AgentFinishAction: For ending the interaction
                - MessageAction: For sending messages
            pending_tool_call_action_messages (dict[str, Message]): Dictionary mapping response IDs
                to their corresponding messages. Used in function calling mode to track tool calls
                that are waiting for their results.

        Returns:
            list[Message]: A list containing the formatted message(s) for the action.
                May be empty if the action is handled as a tool call in function calling mode.

        Note:
            In function calling mode, tool-based actions are stored in pending_tool_call_action_messages
            rather than being returned immediately. They will be processed later when all corresponding
            tool call results are available.
        """

        if isinstance(action, CodeIndexerAction):
            logger.debug(f'CodeIndexerAction in get_action_message: {action}')

        # create a regular message from an event
        if isinstance(
            action,
            (
                AgentDelegateAction,
                CmdRunAction,
                IPythonRunCellAction,
                FileEditAction,
                BrowseInteractiveAction,
                CodeIndexerAction
            ),
        ) or (isinstance(action, AgentFinishAction) and action.source == 'agent'):
            if self.function_calling_active:
                tool_metadata = action.tool_call_metadata
                assert tool_metadata is not None, (
                    'Tool call metadata should NOT be None when function calling is enabled. Action: '
                    + str(action)
                )

                llm_response: ModelResponse = tool_metadata.model_response
                assistant_msg = llm_response.choices[0].message
                logger.debug(
                        f'Action message in function calling mode: {assistant_msg} {tool_metadata}'
                )
                # Add the LLM message (assistant) that initiated the tool calls
                # (overwrites any previous message with the same response_id)
                pending_tool_call_action_messages[llm_response.id] = Message(
                    role=assistant_msg.role,
                    # tool call content SHOULD BE a string
                    content=[TextContent(text=assistant_msg.content or '')]
                    if assistant_msg.content is not None
                    else [],
                    tool_calls=assistant_msg.tool_calls,
                )
                return []
            else:
                assert not isinstance(action, BrowseInteractiveAction), (
                    'BrowseInteractiveAction is not supported in non-function calling mode. Action: '
                    + str(action)
                )
                content = [TextContent(text=self.action_parser.action_to_str(action))]
                return [
                    Message(
                        role='user' if action.source == 'user' else 'assistant',
                        content=content,
                    )
                ]
        elif isinstance(action, MessageAction):
            logger.debug(f'MessageAction in get_action_message: {action}')
            # skip if source is planning
            if action.cause == PLAN_CAUSE_CODE and not action.wait_for_response and not self.waiting_for_feedback:
                return []
            role = 'user' if action.source == 'user' else 'assistant'
            content = [TextContent(text=action.content or '')]
            if self.llm.vision_is_active() and action.images_urls:
                content.append(ImageContent(image_urls=action.images_urls))
            return [
                Message(
                    role=role,
                    content=content,
                )
            ]
        return []

    def get_observation_message(
        self,
        obs: Observation,
        tool_call_id_to_message: dict[str, Message],
    ) -> list[Message]:
        """Converts an observation into a message format that can be sent to the LLM.

        This method handles different types of observations and formats them appropriately:
        - CmdOutputObservation: Formats command execution results with exit codes
        - IPythonRunCellObservation: Formats IPython cell execution results, replacing base64 images
        - FileEditObservation: Formats file editing results
        - AgentDelegateObservation: Formats results from delegated agent tasks
        - ErrorObservation: Formats error messages from failed actions
        - UserRejectObservation: Formats user rejection messages

        In function calling mode, observations with tool_call_metadata are stored in
        tool_call_id_to_message for later processing instead of being returned immediately.

        Args:
            obs (Observation): The observation to convert
            tool_call_id_to_message (dict[str, Message]): Dictionary mapping tool call IDs
                to their corresponding messages (used in function calling mode)

        Returns:
            list[Message]: A list containing the formatted message(s) for the observation.
                May be empty if the observation is handled as a tool response in function calling mode.

        Raises:
            ValueError: If the observation type is unknown
        """
        message: Message
        max_message_chars = self.llm.config.max_message_chars
        obs_prefix = 'OBSERVATION:\n'
        if isinstance(obs, CmdOutputObservation):
            text = obs_prefix + truncate_content(
                obs.content + obs.interpreter_details, max_message_chars
            )
            text += f'\n[Command finished with exit code {obs.exit_code}]'
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, IPythonRunCellObservation):
            text = obs_prefix + obs.content
            # replace base64 images with a placeholder
            splitted = text.split('\n')
            for i, line in enumerate(splitted):
                if '![image](data:image/png;base64,' in line:
                    splitted[i] = (
                        '![image](data:image/png;base64, ...) already displayed to user'
                    )
            text = '\n'.join(splitted)
            text = truncate_content(text, max_message_chars)
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, FileEditObservation):
            text = obs_prefix + truncate_content(str(obs), max_message_chars)
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, BrowserOutputObservation):
            text = obs.get_agent_obs_text()
            message = Message(
                role='user',
                content=[TextContent(text=obs_prefix + text)],
            )
        elif isinstance(obs, AgentDelegateObservation):
            text = obs_prefix + truncate_content(
                obs.outputs['content'] if 'content' in obs.outputs else '',
                max_message_chars,
            )
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, ErrorObservation):
            text = obs_prefix + truncate_content(obs.content, max_message_chars)
            text += '\n[Error occurred in processing last action]'
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, UserRejectObservation):
            text = 'OBSERVATION:\n' + truncate_content(obs.content, max_message_chars)
            text += '\n[Last action has been rejected by the user]'
            message = Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, CodeQueryObservationRes):
            text = obs_prefix + obs.message
            message = Message(role='user', content=[TextContent(text=text)])
        else:
            # If an observation message is not returned, it will cause an error
            # when the LLM tries to return the next message
            raise ValueError(f'Unknown observation type: {type(obs)}')

        if self.function_calling_active:
            # Update the message as tool response properly
            if (tool_call_metadata := obs.tool_call_metadata) is not None:
                # Store the tool call response message for later processing in get_action_message
                tool_call_id_to_message[tool_call_metadata.tool_call_id] = Message(
                    role='tool',
                    content=message.content,
                    tool_call_id=tool_call_metadata.tool_call_id,
                    name=tool_call_metadata.function_name,
                )
                logger.debug(
                    f'Tool call message updated: {tool_call_id_to_message[tool_call_metadata.tool_call_id]} will be removed'
                )
                # No need to return the observation message
                # because it will be added by get_action_message when all the corresponding
                # tool calls in the SAME request are processed
                return []

        return [message]

    def reset(self) -> None:
        """Resets the CodeAct Agent."""
        super().reset()
        self.plan_approved = False
        self.plan = None
        self.waiting_for_feedback = False

    def step(self, state: State) -> Action:
        # Check if waiting for feedback
        if self.waiting_for_feedback:
            last_user_message = state.get_last_user_message()
            if last_user_message:
                self.waiting_for_feedback = False
                return self._handle_plan_feedback(last_user_message, state)
            ac = MessageAction(
                content='Waiting for your feedback on the plan...',
                wait_for_response=True
            )
            ac._cause = PLAN_CAUSE_CODE
            return ac

        # Create and get approval for plan if not done yet
        if not self.plan_approved:
            if not self.plan:
                # Generate initial plan
                plan_messages = self._get_messages(state)
                prompt = (
                    text
                ) = f"""You are an AI task planner responsible for creating detailed execution plans for complex tasks that will be solved by multiple AI language models with varying capabilities. Your goal is to analyze the given task, create a comprehensive plan, and present it in a structured format.

Please create a detailed execution plan for this task. Follow these steps:

1. Analyze the task complexity and requirements.
2. Rank the task into tiers based on the required model capabilities.
3. Create a step-by-step breakdown of actions to be taken.
4. Identify potential risks or considerations.
5. Define expected outcomes at each step.

Task ranking for model:
- Tier1 (Smartest): Required for complex architectural analysis, design pattern recognition, and high-level system relationships. Needs deep understanding of software engineering principles and ability to make sophisticated connections.
- Tier2: Suitable for analyzing individual components, documenting functionality, and explaining code flow. Requires good comprehension of programming concepts but less architectural expertise.
- Tier3 (Smallest, 2b-8b): Can handle basic code documentation, syntax explanation, simple function execution, summarize output. Suitable for line-by-line explanations and basic code structure documentation.

Use the following structure for your analysis and plan:

<execution_plan>

<step_by_step_breakdown>
Provide a detailed breakdown of actions to be taken, please use code language if needed. For each main step, to ensure thoroughness:
1. [Step 1]
   a. [Substep 1a] `[code if any]` [model name assign use @] [Detailed explanation of the expected outcome for the next step]
   b. [SubStep 1b] `[code if any]` [model name assign use @] [Detailed explanation of the expected outcome for the next step]
   c. [Subtep 1c] `[code if any]` [model name assign use @] [Detailed explanation of the expected outcome for the next step]
2. [Step 2]
   a. [Substep 2a] `[code if any]` [model name assign use @] [Detailed explanation of the expected outcome for the next step]
   b. [Substep 2b] `[code if any]` [model name assign use @] [Detailed explanation of the expected outcome for the next step]
   c. [Substep 2c] `[code if any]` [model name assign use @] [Detailed explanation of the expected outcome for the next step]
[Continue with additional steps as needed]
</step_by_step_breakdown>

<potential_risks>
List and briefly explain any potential risks or considerations:
1. [Risk 1]: [Explanation]
2. [Risk 2]: [Explanation]
[Continue with additional risks as needed]
</potential_risks>

<expected_outcomes>
Define the expected outcomes at each main step:
1. [Outcome for Step 1]: [Description]
2. [Outcome for Step 2]: [Description]
[Continue with outcomes for each main step]
</expected_outcomes>
</execution_plan>

Please ensure that your plan is comprehensive, clear, and addresses all aspects of the task. Pay special attention to providing detailed substeps for each main action to be taken.

User prompt will be provided next.
                                         """
                plan_messages.append(
                    Message(role='system', content=[TextContent(text=prompt)])
                )

                plan_params = {
                    'messages': self.llm.format_messages_for_llm(plan_messages),
                }
                plan_response = self.llm.completion(**plan_params)
                self.plan = plan_response.choices[0].message.content

                # Ask user to review plan
                self.waiting_for_feedback = True
                ac = MessageAction(
                    content=f'Here is my proposed execution plan:\n\n{self.plan}\n\n'
                    f'Please review this plan and respond with:\n'
                    f"- 'approve' to proceed with execution\n"
                    f"- 'modify <suggestions>' to request changes to the plan\n"
                    f"- 'reject' to cancel execution",
                    wait_for_response=True,
                )
                ac._cause = PLAN_CAUSE_CODE
                return ac

        # Proceed with normal execution once plan is approved
        return self._execute_step(state)

    def _handle_plan_feedback(self, feedback: str, state: State) -> Action:
        """Handle user feedback on the execution plan"""
        feedback = feedback.lower().strip()

        if 'approve' in feedback:
            self.plan_approved = True
            ac = MessageAction(
                content='Plan approved. Proceeding with execution.',
                wait_for_response=False,
            )
            ac._cause = 100
            return ac
        elif 'reject' in feedback:
            self.plan = None
            return AgentFinishAction(thought='Plan rejected. Ending execution.')
        else:
            # Handle modification request
            modification_request = feedback.replace('modify', '').strip()
            self.plan = None

            # Create new plan based on feedback
            plan_messages = self._get_messages(state)
            plan_messages.append(
                Message(
                    role='user',
                    content=[
                        TextContent(
                            text=f'Based on the following feedback:\n{modification_request}\n\n'
                            'Please create a revised execution plan that outlines:\n'
                            '1. The overall approach to solving the task\n'
                            '2. Step-by-step breakdown of actions to be taken\n'
                            '3. Any potential risks or considerations\n'
                            '4. Expected outcomes at each step'
                        )
                    ],
                )
            )

            plan_params = {
                'messages': self.llm.format_messages_for_llm(plan_messages),
            }
            plan_response = self.llm.completion(**plan_params)
            self.plan = plan_response.choices[0].message.content

            # Request approval for new plan
            self.waiting_for_feedback = True
            ac = MessageAction(
                content=f'Here is my revised execution plan:\n\n{self.plan}\n\n'
                f'Please review this plan and respond with:\n'
                f"- 'approve' to proceed with execution\n"
                f"- 'modify <suggestions>' to request further changes\n"
                f"- 'reject' to cancel execution",
                wait_for_response=True,
            )
            ac._cause = PLAN_CAUSE_CODE
            return ac

    def _execute_step(self, state: State) -> Action:
        """Execute a single step after plan is approved"""
        logger.debug(f'Executing step with plan: {self.plan}')
        messages = self._get_messages(state)

        # Add plan context to the messages
        

        params: dict = {
            'messages': self.llm.format_messages_for_llm(messages),
        }

        if self.function_calling_active:
            logger.debug('Function calling is active, adding tools')
            params['tools'] = self.tools
            params['parallel_tool_calls'] = False
        else:
            logger.debug('Function calling not active, adding stop tokens')
            params['stop'] = [
                '</execute_ipython>',
                '</execute_bash>',
                '</execute_browse>',
                '</file_edit>',
            ]

        # Log the messages being sent to LLM
        logger.debug('Messages being sent to LLM:')
        for msg in messages:
            logger.debug(
                f'Role: {msg.role}, Content: {truncate_content(str(msg.content), 100)}'
            )

        try:
            response = self.llm.completion(**params)
            logger.debug(f'LLM Response: {response}')

            if self.function_calling_active:
                actions = codeact_function_calling.response_to_actions(response)
                for action in actions:
                    self.pending_actions.append(action)
                return self.pending_actions.popleft()
            else:
                return self.action_parser.parse(response)
        except Exception as e:
            logger.error(f'Error during execution: {str(e)}')
            logger.debug(f'Full error details: {traceback.format_exc()}')
            raise

    def _get_messages(self, state: State) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Initializes with system prompt and optional initial user message
        2. Processes events (Actions and Observations) into messages
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            state (State): The current state object containing conversation history and other metadata

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt
                - Initial user message (if configured)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        messages: list[Message] = [
            Message(
                role='system',
                content=[
                    TextContent(
                        text=self.system_prompt,
                        cache_prompt=self.llm.is_caching_prompt_active(),  # Cache system prompt
                    )
                ],
            )
        ]
        if self.initial_user_message:
            messages.append(
                Message(
                    role='user',
                    content=[TextContent(text=self.initial_user_message)],
                )
            )

        plan_context = Message(
            role='user',
            content=[
                TextContent(
                    # cache_prompt=self.llm.is_caching_prompt_active(),

                    text=f'Following the approved plan:\n{self.plan}\n\nPlease execute the next step.' if self.plan_approved
                    else 'Please provide an execution plan for the task.' if not self.plan
                    else 'Previous plan: ' + self.plan,
                )
            ],
        )
        messages.append(plan_context)

        pending_tool_call_action_messages: dict[str, Message] = {}
        tool_call_id_to_message: dict[str, Message] = {}
        events = list(state.history)
        for event in events:
            # create a regular message from an event
            if isinstance(event, Action):
                messages_to_add = self.get_action_message(
                    action=event,
                    pending_tool_call_action_messages=pending_tool_call_action_messages,
                )
            elif isinstance(event, Observation):
                messages_to_add = self.get_observation_message(
                    obs=event,
                    tool_call_id_to_message=tool_call_id_to_message,
                )
            else:
                raise ValueError(f'Unknown event type: {type(event)}')

            # Check pending tool call action messages and see if they are complete
            _response_ids_to_remove = []
            for (
                response_id,
                pending_message,
            ) in pending_tool_call_action_messages.items():
                assert pending_message.tool_calls is not None, (
                    'Tool calls should NOT be None when function calling is enabled & the message is considered pending tool call. '
                    f'Pending message: {pending_message}'
                )
                if all(
                    tool_call.id in tool_call_id_to_message
                    for tool_call in pending_message.tool_calls
                ):
                    # If complete:
                    # -- 1. Add the message that **initiated** the tool calls
                    messages_to_add.append(pending_message)
                    # -- 2. Add the tool calls **results***
                    for tool_call in pending_message.tool_calls:
                        messages_to_add.append(tool_call_id_to_message[tool_call.id])
                        tool_call_id_to_message.pop(tool_call.id)
                    _response_ids_to_remove.append(response_id)

            logger.debug(f'Pending tool call action messages: {pending_tool_call_action_messages}')
            logger.debug(f'Tool call ID to message: {tool_call_id_to_message}')
            logger.debug(f'Messages to add: {messages_to_add}')
            logger.debug(f'Response IDs to remove: {_response_ids_to_remove}')
            # Cleanup the processed pending tool messages
            for response_id in _response_ids_to_remove:
                pending_tool_call_action_messages.pop(response_id)

            for message in messages_to_add:
                # add regular message
                if message:
                    # handle error if the message is the SAME role as the previous message
                    # litellm.exceptions.BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'detail': 'Only supports u/a/u/a/u...'}
                    # there shouldn't be two consecutive messages from the same role
                    # NOTE: we shouldn't combine tool messages because each of them has a different tool_call_id
                    if (
                        messages
                        and messages[-1].role == message.role
                        and message.role != 'tool'
                    ):
                        messages[-1].content.extend(message.content)
                    else:
                        messages.append(message)

        if self.llm.is_caching_prompt_active():
            # NOTE: this is only needed for anthropic
            # following logic here:
            # https://github.com/anthropics/anthropic-quickstarts/blob/8f734fd08c425c6ec91ddd613af04ff87d70c5a0/computer-use-demo/computer_use_demo/loop.py#L241-L262
            breakpoints_remaining = 3  # remaining 1 for system/tool
            for message in reversed(messages):
                if message.role == 'user' or message.role == 'tool':
                    if breakpoints_remaining > 0:
                        message.content[
                            -1
                        ].cache_prompt = True  # Last item inside the message content
                        breakpoints_remaining -= 1
                    else:
                        break

        if not self.function_calling_active:
            # The latest user message is important:
            # we want to remind the agent of the environment constraints
            latest_user_message = next(
                islice(
                    (
                        m
                        for m in reversed(messages)
                        if m.role == 'user'
                        and any(isinstance(c, TextContent) for c in m.content)
                    ),
                    1,
                ),
                None,
            )
            # do not add this for function calling
            if latest_user_message:
                reminder_text = f'\n\nENVIRONMENT REMINDER: You have {state.max_iterations - state.iteration} turns left to complete the task. When finished reply with <finish></finish>.'
                latest_user_message.content.append(TextContent(text=reminder_text))

        logger.debug(f'Final get Messages: {messages}')

        return messages
