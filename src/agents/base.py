"""Base abstractions and helpers for orchestration agents."""

import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from re import Match
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from src.callbacks import AgentCallbacks
from src.prompts import AGENT_SYSTEM_PROMPT_TEMPLATE
from src.utils import ConversationMessage, ParticipantRole, TemplateVariables


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AgentOptions:
    """Configuration options shared by all agents."""

    id: str | None
    name: str
    description: str
    save_chat: bool = False
    callbacks: AgentCallbacks | None = None


class Agent(ABC):
    """Abstract base class for all orchestration agents."""

    def __init__(self, options: AgentOptions) -> None:
        """Initialise common agent attributes and default system prompt."""
        self.id = options.id or self.generate_unique_id()
        self.name = options.name
        self.description = options.description
        self.save_chat = options.save_chat

        current_date = datetime.now().strftime("%d %B %Y")

        self.callbacks = (
            options.callbacks if options.callbacks is not None else AgentCallbacks()
        )

        self.system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=self.name,
            description=self.description,
            current_date=current_date,
        )

    @staticmethod
    def generate_unique_id() -> str:
        """Return a compact unique identifier."""
        return str(uuid.uuid4().hex[:24])

    @abstractmethod
    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Handle an incoming payload using the agent's configuration."""
        raise NotImplementedError

    async def prepare_chat_history(
        self, chat_history: list[ConversationMessage]
    ) -> list[dict[str, str]]:
        """Format prior conversation with new input for downstream models."""
        messages = [
            {
                "role": (
                    "user"
                    if (
                        (
                            msg.role.value
                            if isinstance(msg.role, ParticipantRole)
                            else msg.role
                        )
                        in ("user", ParticipantRole.USER.value)
                    )
                    else "assistant"
                ),
                "content": msg.content[0]["text"] if msg.content else "",
            }
            for msg in chat_history or []
        ]
        return messages

    def set_system_prompt(
        self,
        template: str | None = None,
        variables: TemplateVariables | None = None,
    ) -> str:
        """Set or update the system prompt, optionally applying template variables."""
        if template is None and variables is None:
            return self.system_prompt
        elif template and variables is None:
            self.system_prompt = template
            return self.system_prompt
        self.system_prompt = self.replace_placeholders(template, variables)
        return self.system_prompt

    @staticmethod
    def replace_placeholders(template: str, variables: TemplateVariables) -> str:
        """Replace {{var}} placeholders in the given template using the provided variables."""

        def replace(match: Match[str]) -> str:
            key = match.group(1)
            value = variables.get(key)
            if value is not None:
                return "\n".join(value) if isinstance(value, list) else str(value)
            return match.group(0)

        return re.sub(r"{{(\w+)}}", replace, template)
