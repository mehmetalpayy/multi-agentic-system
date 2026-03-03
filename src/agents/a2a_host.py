"""Agent that delegates work to remote A2A agents over HTTP."""

import uuid
from typing import Any

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, Task, TaskState, TextPart
from a2a.utils import get_message_text, get_text_parts
from pydantic import Field
from pydantic.dataclasses import ConfigDict, dataclass

from src.utils import ConversationMessage, Logger, ParticipantRole

from .base import Agent, AgentOptions


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class A2AHostOptions(AgentOptions):
    """Configuration options for the A2A host agent."""

    remote_agent_address: str = Field(default_factory=str)
    httpx_client: Any | None = None
    streaming: bool = False


class A2AHost(Agent):
    """Agent that proxies requests to a remote A2A-compatible agent."""

    def __init__(self, options: A2AHostOptions) -> None:
        """Initialise the A2A host with HTTP client and remote agent address."""
        super().__init__(options)

        self.httpx_client = options.httpx_client
        self.remote_url = options.remote_agent_address
        self.streaming = options.streaming
        self._current_run_id: str | None = None

        self.agent_card = None
        self.factory: ClientFactory | None = None

    async def _fetch_agent_card(self) -> Any:
        resolver = A2ACardResolver(self.httpx_client, self.remote_url)
        return await resolver.get_agent_card()

    async def create(self) -> "A2AHost":
        """Fetch the remote agent card and prepare the A2A client factory."""
        if self.factory and self.agent_card:
            return self

        agent_card = await self._fetch_agent_card()

        self.name = agent_card.name
        self.description = agent_card.description
        self.agent_card = agent_card

        config = ClientConfig(
            httpx_client=self.httpx_client,
            streaming=self.streaming,
        )
        self.factory = ClientFactory(config)
        return self

    async def send_message(self, text: str, user_id: str, session_id: str) -> str:
        """Send a single text message to the remote agent and return its reply."""
        if not self.factory or not self.agent_card:
            raise RuntimeError("ClientFactory not initialized.")

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(kind="text", text=text))],
            message_id=uuid.uuid4().hex,
        )

        client = self.factory.create(self.agent_card)

        async for event in client.send_message(
            message,
            request_metadata={"user_id": user_id, "session_id": session_id},
        ):
            final = await self.handle_event(event)
            if final is not None:
                return final

        return "No response received."

    async def handle_event(self, event: Any) -> str | None:
        """Handle an A2A event and return a final user-facing response if available."""
        task = self.extract_task(event)
        if not task or not task.status:
            return None

        await self.handle_streaming_tokens(task)

        state = task.status.state

        if state == TaskState.completed:
            return await self.extract_completed_text(task)

        if state == TaskState.input_required:
            return self.extract_input_required_text(task)

        return None

    @staticmethod
    def extract_task(event: Any) -> Task | None:
        """Extract a Task instance from an A2A event or event tuple."""
        if isinstance(event, Task):
            return event
        if isinstance(event, tuple) and event and isinstance(event[0], Task):
            return event[0]
        return None

    async def handle_streaming_tokens(self, task: Task) -> None:
        """Emit only reasoning tokens and always record token usage from agent messages."""
        status = task.status
        message = status.message if status else None
        if not message or message.role != Role.agent:
            return

        for part in message.parts or []:
            if not isinstance(part.root, TextPart):
                continue

            metadata = part.root.metadata or {}

            if token_usage := metadata.get("token_usage"):
                Logger.info("%s token_usage=%s", self.name, token_usage)
                await self.callbacks.on_agent_end(
                    agent_name=self.name,
                    run_id=self._current_run_id,
                    metadata={"token_usage": token_usage},
                )

            if metadata.get("content_type") != "reasoning":
                continue

            await self.callbacks.on_llm_new_token(
                token=part.root.text,
                format="reasoning",
                name=self.name,
                run_id=task.id,
            )

    async def extract_completed_text(self, task: Task) -> str | None:
        """Extract final response text and record token usage from completed task artifacts."""
        for artifact in task.artifacts or []:
            for part in artifact.parts or []:
                if isinstance(part.root, TextPart):
                    metadata = part.root.metadata or {}

                    if token_usage := metadata.get("token_usage"):
                        Logger.info("%s token_usage=%s", self.name, token_usage)
                        await self.callbacks.on_agent_end(
                            agent_name=self.name,
                            run_id=self._current_run_id,
                            metadata={"token_usage": token_usage},
                        )

            texts = get_text_parts(artifact.parts or [])
            if texts:
                return "\n".join(texts)

        return None

    @staticmethod
    def extract_input_required_text(task: Task) -> str | None:
        """Extract the user-facing message from an input-required task state."""
        message = task.status.message
        if not message:
            return None

        text = get_message_text(message)
        return text.strip() or None

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Adapt a generic agent request to a single A2A text message."""
        run_id = (additional_params or {}).get("run_id")
        self._current_run_id = run_id
        await self.callbacks.on_agent_start(
            agent_name=self.name,
            payload_input=input_text,
            messages=chat_history,
            run_id=run_id,
        )
        return await self.single_response(
            input_text, user_id, session_id, run_id=run_id
        )

    async def single_response(
        self, user_input: str, user_id: str, session_id: str, run_id: str | None = None
    ) -> ConversationMessage:
        """Wrap the remote agent's reply in a ConversationMessage."""
        text = await self.send_message(user_input, user_id, session_id)
        Logger.info(f"Text: {text}")
        message = ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": text}],
        )
        await self.callbacks.on_agent_end(
            agent_name=self.name,
            response=text,
            messages=[{"role": "assistant", "content": text}],
            run_id=run_id,
        )
        return message
