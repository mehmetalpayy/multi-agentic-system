"""Supervisor agent that orchestrates a team of sub-agents."""

import asyncio
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from src.history import HistoryBase, InMemoryChatHistory
from src.prompts import SUPERVISOR_PROMPT_TEMPLATE
from src.utils import (
    AgentTool,
    AgentTools,
    ConversationMessage,
    Logger,
    ParticipantRole,
)

from .base import Agent, AgentOptions
from .lead_agent import LeadAgent


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    return uuid.uuid4().hex


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SupervisorAgentOptions(AgentOptions):
    """Configuration options for the supervisor agent and its team."""

    lead_agent: LeadAgent | None = None
    team: Sequence[Agent] = Field(default_factory=list)
    storage: HistoryBase | None = None
    instructions: str | None = None
    trace: bool | None = None
    extra_tools: AgentTools | list[AgentTool] | None = None
    context_schema: Any | None = None

    def validate(self) -> None:
        """Validate supervisor configuration and tool options."""
        if not isinstance(self.lead_agent, LeadAgent):
            raise ValueError("Supervisor must be LeadAgent")

        if self.extra_tools:
            if not isinstance(self.extra_tools, (AgentTools, list)):
                raise ValueError(
                    "extra_tools must be Tools object or list of Tool objects"
                )

            tools_to_check = (
                self.extra_tools.tools
                if isinstance(self.extra_tools, AgentTools)
                else self.extra_tools
            )
            if not all(isinstance(tool, AgentTool) for tool in tools_to_check):
                raise ValueError(
                    "extra_tools must be Tools object or list of Tool objects"
                )

        if self.lead_agent.tool_config:
            raise ValueError(
                "Supervisor tools are managed by SupervisorAgent. Use extra_tools for additional tools."
            )


class SupervisorAgent(Agent):
    """Supervisor agent that orchestrates interactions between multiple agents.

    Manages communication, task delegation, and response aggregation between a team of agents. Supports parallel
    processing of messages and maintains conversation history.
    """

    DEFAULT_TOOL_MAX_RECURSIONS = 10

    def __init__(self, options: SupervisorAgentOptions) -> None:
        """Initialise the supervisor with a lead agent, team, storage, and orchestration settings."""
        options.validate()
        options.name = options.lead_agent.name
        options.description = options.lead_agent.description
        super().__init__(options)

        self.lead_agent = options.lead_agent
        self.team = options.team
        self.storage = options.storage or InMemoryChatHistory()
        self.instructions = options.instructions
        self.trace = options.trace
        self.context_schema = options.context_schema

        self.user_id = ""
        self.session_id = ""

        self._configure_supervisor_tools(options.extra_tools)
        self._configure_prompt()

    def _configure_prompt(self) -> None:
        """Configure the lead_agent's prompt template."""
        Logger.debug(
            "Supervisor tool configured | name=%s description=%s",
            getattr(self.supervisor_tools, "name", "unknown"),
            getattr(self.supervisor_tools, "description", ""),
        )
        tools_str = "\n".join(
            f"{tool.name}:{tool.func_description}"
            for tool in self.supervisor_tools.tools
        )
        Logger.debug("Supervisor tools string | value=%s", tools_str)
        agent_list_str = "\n".join(
            f"{agent.name}: {agent.description}" for agent in self.team
        )
        # Logger.info(f"Agent list strings: \n {agent_list_str}")

        instructions_section = ""
        if self.instructions:
            instructions_section = f"""
            <specific_instructions>
            {self.instructions}
            </specific_instructions>

            """
        current_date = datetime.now().strftime("%d %B %Y")
        self.prompt_template = SUPERVISOR_PROMPT_TEMPLATE.format(
            name=self.name,
            description=self.description,
            current_date=current_date,
            instructions_section=instructions_section,
            agent_list_str=agent_list_str,
            tools_str=tools_str,
        )

    def _configure_supervisor_tools(
        self, extra_tools: AgentTools | list[AgentTool] | None
    ) -> None:
        """Configure the tools available to the lead_agent."""
        self.supervisor_tools = AgentTools(
            [
                AgentTool(
                    name="send_messages",
                    description="Send messages to multiple agents in parallel.",
                    properties={
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "recipient": {
                                        "type": "string",
                                        "description": "Agent name to send message to.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Message content.",
                                    },
                                },
                                "required": ["recipient", "content"],
                            },
                            "description": "Array of messages for different agents.",
                            "minItems": 1,
                        }
                    },
                    required=["messages"],
                    func=self._send_messages_tool,
                )
            ]
        )

        if extra_tools:
            if isinstance(extra_tools, AgentTools):
                self.supervisor_tools.tools.extend(extra_tools.tools)
                if extra_tools.callbacks:
                    self.supervisor_tools.callbacks = extra_tools.callbacks
            else:
                self.supervisor_tools.tools.extend(extra_tools)

        self.lead_agent.tool_config = {
            "tool": self.supervisor_tools,
            "toolMaxRecursions": self.DEFAULT_TOOL_MAX_RECURSIONS,
        }

    async def _send_messages_tool(
        self, messages: Any = None, **kwargs: dict[str, Any]
    ) -> str:
        """Adapter used by the lead agent tool to forward messages to team agents."""
        if messages is None and "messages" in kwargs:
            messages = kwargs["messages"]

        normalized: list[dict[str, Any]] = []
        if isinstance(messages, list):
            normalized = messages

        return await self.send_messages(normalized)

    async def send_message(
        self,
        agent: Agent,
        content: str,
        user_id: str,
        session_id: str,
    ) -> str:
        """Send a message to a specific agent and process the response."""
        try:
            # chat_history = await self.storage.fetch_chat_messages(
            #     user_id=user_id, session_id=session_id, agent_id=agent.id
            # )

            user_message = ConversationMessage(
                role=ParticipantRole.USER.value,
                content=[{"text": content}],
            )

            Logger.info(
                "Agent input | agent=%s | text=%s",
                agent.name,
                content,
            )

            agent_run_id = generate_run_id()
            start_time = time.perf_counter()
            response = await agent.process_request(
                input_text=content,
                user_id=user_id,
                session_id=session_id,
                chat_history=[],
                additional_params={
                    "run_id": agent_run_id,
                },
            )

            assistant_message = ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": response.text if response else ""}],
            )

            await self.storage.save_chat_messages(
                user_id, session_id, agent.id, [user_message, assistant_message]
            )
            Logger.info(
                "Agent response | agent=%s | text=%s",
                agent.name,
                response.text,
            )

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            Logger.info(
                f"{agent.name} request execution took {elapsed_time:0.4f} seconds"
            )
            return f"{agent.name}: {response.text}"

        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            Logger.error(f"Error in send_message: {error}")
            raise error

    async def send_messages(self, messages: list[dict[str, str]]) -> str:
        """Dispatch messages to all matching agents and aggregate their responses."""
        if not messages:
            return "No messages to send."
        try:
            Logger.info(
                "Supervisor dispatch start | total_messages=%d | payload=%s",
                len(messages),
                messages,
            )
            tasks = [
                self.send_message(
                    agent,
                    message.get("content"),
                    self.user_id,
                    self.session_id,
                )
                for agent in self.team
                for message in messages
                if (
                    agent.name == message.get("recipient")
                    or agent.name == f"{message.get('recipient')} Agent"
                    or agent.name == message.get("recipient").replace(" Agent", "")
                )
            ]

            if not tasks:
                Logger.info("Supervisor dispatch result | no matching agents")
                return f"No agent matches for the request:{str(messages)}"

            Logger.info("Supervisor dispatch created | task_count=%d", len(tasks))
            responses = await asyncio.gather(*tasks)
            Logger.info(
                "Supervisor dispatch done | response_count=%d",
                len(responses),
            )
            return "".join(responses)

        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            Logger.error(f"Error in send_messages: {error}")
            raise error

    def format_history(self, chat_history: list[ConversationMessage]) -> str:
        """Format conversation history."""
        return "".join(
            f"{user_msg.role}:{user_msg.content[0].get('text', '')}\n"
            f"{asst_msg.role}:{asst_msg.content[0].get('text', '')}\n"
            for user_msg, asst_msg in zip(
                chat_history[::2], chat_history[1::2], strict=False
            )
            if self.id not in asst_msg.content[0].get("text", "")
        )

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Process a user request through the lead_agent agent."""
        self.user_id = user_id
        self.session_id = session_id
        try:
            Logger.info(
                "Supervisor process start | user_id=%s | session_id=%s | input=%s",
                self.user_id,
                self.session_id,
                input_text,
            )
            agents_history = await self.storage.fetch_chat(
                user_id=self.user_id, session_id=self.session_id
            )
            agents_memory = self.format_history(agents_history)
            Logger.info(
                "Supervisor memory prepared | history_messages=%d",
                len(agents_history),
            )
            # Logger.debug(f"Agent history: {agents_memory}")
            summary = await self.storage.fetch_summary(
                user_id=user_id, session_id=session_id
            )
            self.lead_agent.set_system_prompt(
                template=self.prompt_template,
                variables={
                    "AGENTS_MEMORY": agents_memory,
                    "summary": summary,
                },
            )

            Logger.info("Supervisor calling lead agent")
            lead_response = await self.lead_agent.process_request(
                input_text=input_text,
                user_id=self.user_id,
                session_id=self.session_id,
                chat_history=chat_history,
            )
            Logger.info(
                "Supervisor received lead response | text=%s",
                lead_response.text,
            )
            return lead_response

        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            Logger.error(f"Error in process_request: {error}")
            raise error
