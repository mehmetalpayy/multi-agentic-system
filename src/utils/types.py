"""Shared type definitions for orchestration, including conversation messages and template variables."""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


class AgentProviderType(Enum):
    """Agent Provider Types."""

    OPENAI = "OPENAI"
    LITELLM = "LITELLM"


class ParticipantRole(Enum):
    """Roles that a conversation participant may take."""

    ASSISTANT = "assistant"
    USER = "user"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ConversationMessage:
    """Internal representation of a single conversation message."""

    role: ParticipantRole | str
    content: list[Any] = Field(default_factory=list)
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @staticmethod
    def _extract_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("text", "content", "output_text"):
                text = value.get(key)
                if isinstance(text, str):
                    return text
            return None
        if isinstance(value, list):
            parts = []
            for item in value:
                text = ConversationMessage._extract_text(item)
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
            return None
        return None

    @property
    def text(self) -> str | None:
        """Return the primary text payload if present."""
        if not self.content:
            return None
        first = self.content[0]
        if isinstance(first, dict):
            if "text" in first:
                return self._extract_text(first["text"])
            if "content" in first:
                return self._extract_text(first["content"])
        if isinstance(first, str):
            return first
        return self._extract_text(first)

    def to_dict(self) -> dict[str, Any]:
        """Convert the message to a JSON-serialisable dictionary."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "role": self.role.value if isinstance(self.role, Enum) else self.role,
            "content": self.content,
        }


TemplateVariables = dict[str, str | list[str]]
