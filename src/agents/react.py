"""LangGraph-based React-style agent implementation."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import AzureChatOpenAI
from pydantic import Field
from pydantic.dataclasses import dataclass

from env import secrets
from src.core import settings
from src.utils import ConversationMessage, Logger, ParticipantRole

from .base import Agent, AgentOptions


@dataclass
class ReactAgentOptions(AgentOptions):
    """Configuration options specific to the ReactAgent."""

    retriever: Any | None = None
    client: Any | None = None
    inference_config: dict[str, Any] | None = None
    tools: list[Any] = Field(default_factory=list)
    custom_system_prompt: dict[str, Any] | None = None
    model: str = "gpt-4.1"
    streaming: bool = False
    additional_model_request_fields: dict[str, Any] | None = None


class ReactAgent(Agent):
    """Agent that delegates reasoning and acting to a LangGraph ReAct graph."""

    def __init__(self, options: ReactAgentOptions) -> None:
        """Initialise the React agent with tools, model, and inference configuration."""
        super().__init__(options)
        self.retriever = options.retriever
        self.tools = options.tools

        default_inference_config = {
            "max_tokens": settings.AGENTS_LLM_MAX_TOKENS,
            "temperature": settings.AGENTS_LLM_TEMPERATURE,
            "top_p": settings.AGENTS_LLM_TOP_P,
        }

        self.inference_config = default_inference_config.copy()
        if options.inference_config:
            self.inference_config.update(options.inference_config)

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get("template"),
                options.custom_system_prompt.get("variables"),
            )

        self.additional_model_request_fields: dict[str, Any] | None = (
            options.additional_model_request_fields or {}
        )
        if self.additional_model_request_fields.get("reasoning"):
            # Logger.warn("Removing top_p for reasoning/thinking mode")
            self.inference_config.pop("top_p", None)
            self.inference_config.pop("temperature", None)

        self.streaming = options.streaming
        self.model = options.model

        self.client = options.client or AzureChatOpenAI(
            azure_endpoint=settings.AZURE_ENDPOINT,
            api_key=secrets.AZURE_OPENAI_API_KEY,
            azure_deployment=settings.AZURE_DEPLOYMENT_NAME,
            api_version=settings.AZURE_API_VERSION,
            **self.inference_config,
            **self.additional_model_request_fields,
        )
        self.token_usage_callback = UsageMetadataCallbackHandler()
        self.graph = create_agent(
            self.client, tools=self.tools, system_prompt=self.system_prompt
        )

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Process a user request through the React-style LangGraph agent."""
        try:
            run_id = (additional_params or {}).get("run_id")
            messages = await self.prepare_chat_history(chat_history)

            final_messages = [
                *messages,
                {"role": "user", "content": input_text},
            ]
            await self.callbacks.on_agent_start(
                agent_name=self.name,
                payload_input=input_text,
                messages=final_messages,
                run_id=run_id,
            )

            if self.streaming:
                return await self.streaming_response(
                    final_messages,
                    user_id=user_id,
                    session_id=session_id,
                    run_id=run_id,
                )
            else:
                return await self.single_response(
                    final_messages,
                    user_id=user_id,
                    session_id=session_id,
                    run_id=run_id,
                )

        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            Logger.error(f"Error in API call: {str(error)}")
            raise error

    async def single_response(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
        run_id: str | None = None,
    ) -> ConversationMessage:
        """Produce a single assistant message using the LangGraph graph."""
        try:
            graph_output = await self.graph.ainvoke(
                {"messages": messages},
                config={"callbacks": [self.token_usage_callback]},
                context={
                    "user_id": user_id,
                    "session_id": session_id,
                },
            )
            assistant_message = graph_output["messages"][-1].content
            total_usage = self.token_usage_callback.usage_metadata
            Logger.info("%s token_usage=%s", self.name, total_usage)
            await self.callbacks.on_agent_end(
                agent_name=self.name,
                response=assistant_message,
                messages=messages,
                run_id=run_id,
                metadata={"token_usage": total_usage} if total_usage else None,
            )
            Logger.info(f"{self.name} response={assistant_message}")

            if not isinstance(assistant_message, str):
                raise ValueError("Unexpected response format from API")

            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": assistant_message}],
            )

        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            Logger.error(f"Error in API call: {str(error)}")
            error_message = "Unknown error occurred while processing the request."

            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value, content=[{"text": error_message}]
            )

    async def streaming_response(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
        run_id: str | None = None,
    ) -> ConversationMessage:
        """Stream response chunks and collect reasoning/tool events."""
        content_parts: list[str] = []

        async for event in self.graph.astream_events(
            {"messages": messages},
            version="v2",
            config={"callbacks": [self.token_usage_callback]},
            context={"user_id": user_id, "session_id": session_id},
        ):
            if event.get("event") != "on_chat_model_stream":
                continue

            chunk = event.get("data", {}).get("chunk")
            if not chunk:
                continue
            content = getattr(chunk, "content", None)

            if isinstance(content, list):
                await self._handle_stream_blocks(
                    blocks=content, content_parts=content_parts, run_id=run_id
                )
                continue

        total_usage = self.token_usage_callback.usage_metadata
        Logger.info("%s token_usage=%s", self.name, total_usage)
        await self.callbacks.on_agent_end(
            agent_name=self.name,
            response="".join(content_parts).strip(),
            messages=messages,
            run_id=run_id,
            metadata={"token_usage": total_usage} if total_usage else None,
        )
        return ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": "".join(content_parts).strip()}],
        )

    async def _handle_stream_blocks(
        self,
        blocks: list[dict[str, Any]],
        content_parts: list[str],
        run_id: str | None,
    ) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue

            btype = block.get("type")
            if btype == "reasoning":
                for summary in block.get("summary", []):
                    if not isinstance(summary, dict):
                        continue
                    text = summary.get("text")
                    if text:
                        await self.callbacks.on_llm_new_token(
                            token=text,
                            format="reasoning",
                            name=self.name,
                            run_id=run_id,
                        )
            elif btype == "text":
                text = block.get("text")
                if text:
                    content_parts.append(text)
