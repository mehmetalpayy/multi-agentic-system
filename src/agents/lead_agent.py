"""Lead agent that coordinates tool-using LLM calls via LiteLLM."""

import uuid
from dataclasses import dataclass
from typing import Any

from litellm import acompletion
from litellm.utils import CustomStreamWrapper, ModelResponse

from env import secrets
from src.core import settings
from src.utils import (
    AgentProviderType,
    AgentTools,
    ConversationMessage,
    Logger,
    ParticipantRole,
)

from .base import Agent, AgentOptions


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    return uuid.uuid4().hex


@dataclass
class LeadAgentOptions(AgentOptions):
    """Configuration options for the lead (coordinator) agent."""

    inference_config: dict[str, Any] | None = None
    retriever: Any | None = None
    tool_config: dict[str, Any] | None = None
    custom_system_prompt: dict[str, Any] | None = None
    model: str = "gpt-4.1"
    streaming: bool = False
    additional_model_request_fields: dict[str, Any] | None = None


class LeadAgent(Agent):
    """Agent responsible for coordinating tools and delegating via LiteLLM."""

    def __init__(self, options: LeadAgentOptions) -> None:
        """Initialise the lead agent with model, tools, and inference configuration."""
        super().__init__(options)
        self.retriever = options.retriever
        self.tool_config: dict[str, Any] | None = options.tool_config

        self.inference_config = {
            "max_tokens": settings.AGENTS_LLM_MAX_TOKENS,
            "temperature": settings.AGENTS_LLM_TEMPERATURE,
            "top_p": settings.AGENTS_LLM_TOP_P,
        }

        if options.inference_config:
            self.inference_config.update(options.inference_config)

        self.additional_model_request_fields: dict[str, Any] | None = (
            options.additional_model_request_fields or {}
        )

        if self.additional_model_request_fields.get("reasoning_effort"):
            # Logger.warn("Removing top_p for reasoning_effort mode")
            self.inference_config.pop("top_p", None)
            self.inference_config.pop("temperature", None)

        if options.custom_system_prompt:
            self.set_system_prompt(
                options.custom_system_prompt.get("template"),
                options.custom_system_prompt.get("variables"),
            )

        self.azure_config = {
            "api_base": settings.AZURE_ENDPOINT,
            "api_key": secrets.AZURE_OPENAI_API_KEY,
            "api_version": settings.AZURE_API_VERSION,
        }

        self.default_max_recursions: int = 10

        self.streaming = options.streaming
        self.model = options.model or f"azure/{settings.AZURE_DEPLOYMENT_NAME}"

    def _get_max_recursions(self) -> int:
        """Get the maximum number of recursions based on tool configuration."""
        if not self.tool_config:
            return 1
        return self.tool_config.get("toolMaxRecursions", self.default_max_recursions)

    async def process_request(
        self,
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Process a user request, optionally coordinating tools via LiteLLM."""
        Logger.debug("LeadAgent.process_request | start | streaming=%s", self.streaming)
        messages = await self.prepare_chat_history(chat_history)
        Logger.debug(
            "LeadAgent.process_request | prepared_chat_history=%s",
            messages,
        )
        final_system_prompt = self.system_prompt

        messages = [
            {"role": "system", "content": final_system_prompt},
            *messages,
            {"role": "user", "content": input_text},
        ]
        Logger.debug(f"{self.name} messages history={messages}")

        input_data: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.azure_config,
            **self.inference_config,
            **self.additional_model_request_fields,
            "drop_params": True,
        }

        if self.tool_config:
            # Logger.debug(
            #     f"LeadAgent.process_request | tools_enabled | count={self.tool_config}"
            # )
            tools_payload = self._format_tools_for_litellm()
            # Logger.debug(f"Tools: {tools_payload}")
            input_data["tools"] = tools_payload
            # input_data["tool_choice"] = "auto"

            final_message = ""
            tool_use = True
            max_recursions = self._get_max_recursions()

            while tool_use and max_recursions > 0:
                run_id = generate_run_id()
                await self.callbacks.on_agent_start(
                    agent_name=self.name,
                    payload_input=input_text,
                    messages=input_data["messages"],
                    run_id=run_id,
                )
                response = await (
                    self.streaming_response(input_data, run_id=run_id)
                    if self.streaming
                    else self.single_response(input_data, run_id=run_id)
                )
                if not response.choices:
                    Logger.error(
                        "LeadAgent.process_request | empty_response | response=%r",
                        response,
                    )
                tool_calls = (
                    response.choices[0].message.tool_calls
                    if response.choices and response.choices[0].message.tool_calls
                    else []
                )
                if tool_calls:
                    serializable_tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                    Logger.info(
                        "LeadAgent.process_request | dispatching %d tool call(s)",
                        len(serializable_tool_calls),
                    )
                    input_data["messages"].append(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                            "tool_calls": serializable_tool_calls,
                        }
                    )
                    Logger.debug(
                        "LeadAgent.process_request | assistant message appended with tool calls"
                    )

                    tools_manager: AgentTools = self.tool_config["tool"]
                    tool_responses = await tools_manager.tool_handler(
                        AgentProviderType.LITELLM.value,
                        response,
                        [],
                    )
                    if not tool_responses:
                        Logger.warn(
                            "LeadAgent.process_request | empty_tool_response | tool_calls=%r",
                            serializable_tool_calls,
                        )

                    for tool_response in tool_responses:
                        input_data["messages"].append(tool_response)
                    tool_use = True

                else:
                    final_message = response.choices[0].message.content or ""
                    tool_use = False
                    if not final_message:
                        Logger.warn(
                            "LeadAgent.process_request | empty_final_message | message=%r",
                            response.choices[0].message,
                        )

                max_recursions -= 1

            await self.callbacks.on_agent_end(agent_name=self.name)
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": final_message}],
            )
        else:
            Logger.info("LeadAgent.process_request | tools_disabled_path")
            run_id = generate_run_id()
            await self.callbacks.on_agent_start(
                agent_name=self.name,
                payload_input=input_text,
                messages=input_data["messages"],
                run_id=run_id,
            )
            response = await self.single_response(input_data, run_id=run_id)
            response_content = response.choices[0].message.content or ""
            await self.callbacks.on_agent_end(agent_name=self.name)
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": response_content}],
            )

    async def single_response(
        self, input_data: dict, run_id: str | None = None
    ) -> ModelResponse | CustomStreamWrapper:
        """Call the underlying model once and record token usage."""
        try:
            Logger.debug(
                "LeadAgent.single_response | dispatch | keys=%s",
                list(input_data.keys()),
            )

            response = await acompletion(**input_data)
            # Logger.info(f"{self.name} response={response}")
            usage = getattr(response, "usage", None)
            Logger.info("%s token_usage=%s", self.name, usage)
            await self.callbacks.on_agent_end(
                agent_name=self.name,
                response=response,
                run_id=run_id,
                metadata={"token_usage": usage} if usage else None,
            )

            return response
        except Exception as error:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            Logger.error(f"Error invoking model via litellm: {error}")
            raise error

    async def streaming_response(
        self, input_data: dict, run_id: str | None = None
    ) -> ModelResponse:
        """Stream model output and tool calls into a single response."""
        Logger.debug("LeadAgent.streaming_response | starting")

        accumulated_content = ""
        current_tool_calls: dict[int, dict[str, Any]] = {}
        completed_tool_calls: list[dict[str, Any]] = []
        last_chunk = None

        stream = await acompletion(
            **{
                **input_data,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        async for chunk in stream:
            last_chunk = chunk
            if not chunk.choices:
                Logger.warn(
                    "LeadAgent.streaming_response | chunk_without_choices | chunk=%r",
                    chunk,
                )
                continue
            delta = chunk.choices[0].delta

            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                await self.callbacks.on_llm_new_token(
                    token=reasoning_content,
                    format="reasoning",
                    name=self.name,
                    run_id=run_id,
                )

            if delta.content:
                if not isinstance(delta.content, str):
                    Logger.warn(
                        "LeadAgent.streaming_response | non_string_delta_content | type=%s | content=%r",
                        type(delta.content).__name__,
                        delta.content,
                    )
                chunk_content = delta.content
                accumulated_content += chunk_content
                await self.callbacks.on_llm_new_token(
                    token=chunk_content,
                    format="response",
                    name=self.name,
                    run_id=run_id,
                )

            if delta.tool_calls:
                current_tool_calls = self._handle_tool_calls(
                    delta.tool_calls,
                    current_tool_calls,
                    completed_tool_calls,
                )
            elif delta.content is None and not reasoning_content:
                Logger.debug(
                    "LeadAgent.streaming_response | empty_delta | delta=%r",
                    delta,
                )

        tool_calls = completed_tool_calls + [
            {
                "id": data["id"],
                "type": "function",
                "function": {
                    "name": data["function"]["name"],
                    "arguments": data["function"]["arguments"],
                },
            }
            for data in current_tool_calls.values()
            if data["id"] and data["function"]["name"]
        ]

        usage = getattr(last_chunk, "usage", None) if last_chunk else None
        Logger.info("%s token_usage=%s", self.name, usage)

        message = {
            "role": "assistant",
            "content": accumulated_content or None,
            "tool_calls": tool_calls or None,
        }
        response_dict = {
            "id": getattr(last_chunk, "id", "stream") if last_chunk else "stream",
            "object": (
                getattr(last_chunk, "object", "chat.completion")
                if last_chunk
                else "chat.completion"
            ),
            "created": getattr(last_chunk, "created", None) if last_chunk else None,
            "model": input_data["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                }
            ],
        }
        await self.callbacks.on_agent_end(
            agent_name=self.name,
            response=response_dict,
            run_id=run_id,
            metadata={"token_usage": usage} if usage else None,
        )
        return ModelResponse(**response_dict)

    def _handle_tool_calls(
        self,
        tool_calls_delta: list,
        current_tool_calls: dict[int, dict[str, Any]],
        completed_tool_calls: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """Handle tool calls from stream."""
        for tool_call_delta in tool_calls_delta:
            index = tool_call_delta.index
            if index not in current_tool_calls:
                current_tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }

            active_call = current_tool_calls[index]
            if (
                tool_call_delta.id
                and active_call["id"]
                and tool_call_delta.id != active_call["id"]
            ):
                completed_tool_calls.append(active_call)
                current_tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                active_call = current_tool_calls[index]

            if tool_call_delta.id:
                active_call["id"] = tool_call_delta.id
            if tool_call_delta.function:
                if tool_call_delta.function.name:
                    active_call["function"]["name"] = tool_call_delta.function.name
                if tool_call_delta.function.arguments:
                    active_call["function"]["arguments"] += (
                        tool_call_delta.function.arguments
                    )

        return current_tool_calls

    def _format_tools_for_litellm(self) -> list[dict[str, Any]]:
        if not self.tool_config:
            return []
        tools = []
        try:
            agent_tools_container = self.tool_config.get("tool")
            if isinstance(agent_tools_container, AgentTools):
                for tool in agent_tools_container.tools:
                    tools.append(tool.to_litellm_format())
        except Exception as e:
            Logger.error(f"Error formatting tools for litellm: {e}")
        return tools
