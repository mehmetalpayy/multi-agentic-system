"""Simple in-memory implementation of the history backend."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.utils import ConversationMessage, Logger

from .base import HistoryBase


class InMemoryChatHistory(HistoryBase):
    """Simple in-memory chat history store; cleared when the process exits."""

    @classmethod
    async def connect(cls) -> None:
        """No-op connect for in-memory backend."""
        return None

    @classmethod
    async def disconnect(cls) -> None:
        """No-op disconnect for in-memory backend."""
        return None

    def __init__(self) -> None:
        """Initialise the in-memory storage."""
        self._conversations: defaultdict[str, list[ConversationMessage]] = defaultdict(
            list
        )
        self._summaries: dict[str, str] = {}

    @staticmethod
    def _scoped_key(session_id: str, agent_id: str | None = None) -> str:
        """Build a storage key scoped by optional agent id."""
        if agent_id:
            return f"{session_id}::{agent_id}"
        return session_id

    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        new_message: ConversationMessage,
        max_history_size: int | None = None,
    ) -> list[ConversationMessage]:
        """Persist a chat message in memory and return the (optionally trimmed) history."""
        key = self._scoped_key(session_id)
        history = self._conversations[key]
        history.append(new_message)

        if max_history_size is not None:
            trimmed, _ = await self.trim_conversation(
                history,
                max_history_size=max_history_size,
            )
            self._conversations[key] = trimmed
            history = trimmed

        Logger.debug(
            "Chat message stored for session=%s (total=%d)",
            session_id,
            len(history),
        )
        return list(history)

    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
        max_history_size: int | None = None,
    ) -> list[ConversationMessage]:
        """Fetch chat messages for a session with an optional size limit."""
        history = await self.fetch_chat_messages(user_id, session_id, agent_id)
        if max_history_size is not None and len(history) > max_history_size:
            return history[-max_history_size:]
        return history

    async def fetch_chat_messages(
        self,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[ConversationMessage]:
        """Return the in-memory history for the given session."""
        key = self._scoped_key(session_id, agent_id)
        history = self._conversations.get(key, [])
        return list(history)

    async def fetch_summary(
        self,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
    ) -> str | None:
        """Return summary for a session when available."""
        key = self._scoped_key(session_id, agent_id)
        return self._summaries.get(key)

    async def save_chat_messages(
        self,
        user_id: str,
        session_id: str,
        agent_id: str | None,
        new_messages: list[ConversationMessage],
        max_history_size: int | None = None,
    ) -> list[ConversationMessage]:
        """Save multiple chat messages and return the updated conversation."""
        history: list[ConversationMessage] = []
        target_session_id = self._scoped_key(session_id, agent_id)
        for message in new_messages:
            history = await self.save_chat_message(
                user_id=user_id,
                session_id=target_session_id,
                new_message=message,
                max_history_size=max_history_size,
            )
        return history

    async def save_memory(
        self,
        user_id: str,
        namespace: str,
        memory: str,
        embedding: list[float],
    ) -> Any:
        """In-memory backend does not persist vector memories."""
        _ = (user_id, namespace, memory, embedding)
        return None

    async def search_memory(
        self,
        user_id: str,
        namespace: str,
        embedding: list[float],
        limit: int = 3,
    ) -> list[dict[str, str]]:
        """In-memory backend returns no vector-memory matches."""
        _ = (user_id, namespace, embedding, limit)
        return []
