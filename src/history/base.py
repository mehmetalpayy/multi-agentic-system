"""Abstract interfaces and helpers for conversation history backends."""

from abc import ABC, abstractmethod

from src.utils import ConversationMessage


class HistoryBase(ABC):
    """Abstract base class representing the interface for an agent."""

    @classmethod
    @abstractmethod
    async def connect(cls) -> None:
        """Initializes the connection to the history backend."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    async def disconnect(cls) -> None:
        """Closes the connection to the history backend."""
        raise NotImplementedError

    @staticmethod
    def generate_key(session_id: str) -> str:
        """Generate a unique key for a conversation."""
        return session_id

    def is_same_role_as_last_message(
        self, conversation: list[ConversationMessage], new_message: ConversationMessage
    ) -> bool:
        """Check if the new message is consecutive with the last message in the conversation.

        Args:
            conversation (list[ConversationMessage]): The existing conversation.
            new_message (ConversationMessage): The new message to check.

        Returns:
            bool: True if the new message is consecutive, False otherwise.
        """
        if not conversation:
            return False
        return conversation[-1].role == new_message.role

    async def trim_conversation(
        self,
        conversation: list[ConversationMessage],
        max_history_size: int | None = None,
        trim_to_size: int | None = None,
    ) -> tuple[list[ConversationMessage], list[ConversationMessage]]:
        """Trims a conversation based on a threshold or a hard limit.

        Args:
            conversation (list[ConversationMessage]): The conversation to trim.
            max_history_size (Optional[int]): The maximum size or the threshold to trigger trimming.
            trim_to_size (Optional[int]): If provided, the conversation is trimmed to this size
                once `max_history_size` is reached.

        Returns:
            tuple[list[ConversationMessage], list[ConversationMessage]]: A tuple of
                (trimmed_conversation, removed_messages).
        """
        if trim_to_size is not None and max_history_size is not None:
            if len(conversation) >= max_history_size:
                return conversation[-trim_to_size:], conversation[:-trim_to_size]
            return conversation, []

        if max_history_size is not None and len(conversation) > max_history_size:
            return conversation[-max_history_size:], conversation[:-max_history_size]

        return conversation, []

    @abstractmethod
    async def save_chat_message(
        self,
        user_id: str,
        session_id: str,
        new_message: ConversationMessage,
        max_history_size: int | None = None,
    ) -> list[ConversationMessage]:
        """Save a new chat message.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            new_message (ConversationMessage): The new message to save.
            max_history_size (Optional[int]): The maximum history size.

        Returns:
            list[ConversationMessage]: The trimmed conversation if successful, otherwise an empty list.
        """

    @abstractmethod
    async def fetch_chat(
        self,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
        max_history_size: int | None = None,
    ) -> list[ConversationMessage]:
        """Fetch chat messages.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (Optional[int]): The agent identifier to scope history to, if applicable.
            max_history_size (Optional[int]): The maximum number of messages to fetch.

        Returns:
            list[ConversationMessage]: The fetched chat messages.
        """

    @abstractmethod
    async def fetch_chat_messages(
        self,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[ConversationMessage]:
        """Fetch all chat messages for a user and session.

        Args:
            user_id (str): The user ID.
            session_id (str): The session ID.
            agent_id (Optional[int]): The agent identifier to scope history to, if applicable.

        Returns:
            list[ConversationMessage]: All chat messages for the user and session.
        """
