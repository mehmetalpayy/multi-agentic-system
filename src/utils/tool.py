"""Tool abstractions and utilities for orchestration agents."""

import inspect
import json
import re
from collections.abc import Callable
from functools import wraps
from typing import Any, get_type_hints

from litellm.utils import ModelResponse
from pydantic.dataclasses import dataclass

from .logger import Logger
from .types import (
    AgentProviderType,
)


@dataclass
class AgentToolResult:
    """Structured representation of a single tool invocation result."""

    tool_use_id: str
    content: Any

    def to_openai_format(self) -> dict:
        """Convert the result into an OpenAI-compatible tool message."""
        return {
            "tool_call_id": self.tool_use_id,
            "role": "tool",
            "content": self.content,
        }


class AgentTool:
    """Wrapper that turns a Python callable into a structured tool definition."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        properties: dict[str, dict[str, Any]] | None = None,
        required: list[str] | None = None,
        func: Callable | None = None,
        enum_values: dict[str, list] | None = None,
    ) -> None:
        """Initialise a tool from a callable and its schema.

        Args:
            name (str): Public name of the tool exposed to the LLM.
            description (Optional[str]): Human‑readable description; if omitted, derived from the function docstring.
            properties (Optional[dict[str, dict[str, Any]]]): JSON‑schema style parameter definitions.
            required (Optional[list[str]]): Names of required parameters; defaults to all properties.
            func (Callable): Underlying Python function implementing the tool.
            enum_values (Optional[dict[str, list]]): Optional enum constraints per parameter name.
        """
        self.name = name
        # Extract docstring if description not provided
        if description is None:
            docstring = inspect.getdoc(func)
            if docstring:
                # Get the first paragraph of the docstring (before any parameter descriptions)
                self.func_description = docstring.split("\n\n")[0].strip()
            else:
                self.func_description = f"Function to {name}"
        else:
            self.func_description = description
        self.enum_values = enum_values or {}

        if not func:
            raise ValueError("Function must be provided")

        # Extract properties from the function if not passed
        self.properties = properties or self._extract_properties(func)
        self.required = required or list(self.properties.keys())
        self.func = self._wrap_function(func)

        # Add enum values to properties if they exist
        for prop_name, enum_vals in self.enum_values.items():
            if prop_name in self.properties:
                self.properties[prop_name]["enum"] = enum_vals

    def _extract_properties(self, func: Callable) -> dict[str, dict[str, Any]]:
        """Extract properties from the function's signature and type hints."""
        # Get function's type hints and signature
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Parse docstring for parameter descriptions
        docstring = inspect.getdoc(func) or ""
        param_descriptions = {}

        # Extract parameter descriptions using regex
        param_matches = re.finditer(r":param\s+(\w+)\s*:\s*([^:\n]+)", docstring)
        for match in param_matches:
            param_name = match.group(1)
            description = match.group(2).strip()
            param_descriptions[param_name] = description

        properties = {}
        for param_name, _param in sig.parameters.items():
            # Skip 'self' parameter for class methods
            if param_name in ("self", "runtime"):
                continue

            param_type = type_hints.get(param_name, Any)

            # Convert Python types to JSON schema types
            type_mapping = {
                int: "integer",
                float: "number",
                str: "string",
                bool: "boolean",
                list: "array",
                dict: "object",
            }

            json_type = type_mapping.get(param_type, "string")

            # Use docstring description if available, else create a default one
            description = param_descriptions.get(
                param_name, f"The {param_name} parameter"
            )

            properties[param_name] = {"type": json_type, "description": description}

        return properties

    def _wrap_function(self, func: Callable) -> Callable:
        """Wrap the function to preserve its metadata and handle async/sync functions."""

        @wraps(func)
        async def wrapper(**kwargs):
            result = func(**kwargs)
            if inspect.iscoroutine(result):
                return await result
            return result

        return wrapper

    def to_openai_format(self) -> dict[str, Any]:
        """Convert generic tool definition to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": self.name.lower().replace("_tool", ""),
                "description": self.func_description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                    "additionalProperties": False,
                },
            },
        }

    def to_litellm_format(self) -> dict[str, Any]:
        """Convert generic tool definition to LiteLLM format."""
        return self.to_openai_format()


class AgentTools:
    """Container for a collection of agent tools and their callbacks."""

    def __init__(self, tools: list[AgentTool]) -> None:
        """Initialise the tool container with tools and optional callbacks.

        Args:
            tools (list[AgentTool]): List of tool definitions to expose.
        """
        self.tools: list[AgentTool] = tools

    async def tool_handler(
        self,
        provider_type: str,
        response: ModelResponse,
        _conversation: list[dict[str, Any]],
    ) -> dict | list:
        """Process tool calls from different providers and return formatted results."""
        try:
            tool_results = []
            tool_calls = self._get_tool_calls(provider_type, response)

            for tool_call in tool_calls:
                result = await self._process_tool(
                    tool_call.get("name"),
                    tool_call.get("input", {}),
                )
                tool_result = AgentToolResult(tool_call.get("id"), result)
                if provider_type in [
                    AgentProviderType.OPENAI.value,
                    AgentProviderType.LITELLM.value,
                ]:
                    tool_results.append(tool_result.to_openai_format())

            if not tool_results:
                return []

            if provider_type in [
                AgentProviderType.OPENAI.value,
                AgentProviderType.LITELLM.value,
            ]:
                return tool_results
            return {"role": "user", "content": tool_results}

        except Exception as err:
            raise ValueError(f"Error processing tool response: {str(err)}") from err

    def _get_tool_calls(
        self, provider_type: str, response: ModelResponse
    ) -> list[dict]:
        """Extract tool calls from response based on provider type."""
        tool_calls = []

        if provider_type in [
            AgentProviderType.OPENAI.value,
            AgentProviderType.LITELLM.value,
        ]:
            if response.choices and response.choices[0].message.tool_calls:
                for call in response.choices[0].message.tool_calls:
                    try:
                        parsed_args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        parsed_args = {}
                    tool_calls.append(
                        {
                            "name": call.function.name,
                            "id": call.id,
                            "input": parsed_args,
                        }
                    )

        return tool_calls

    async def _process_tool(self, tool_name: str, input_data: dict[str, Any] | None):
        try:
            tool = next(tool for tool in self.tools if tool.name == tool_name)
            call_kwargs: dict[str, Any] = dict(input_data or {})
            Logger.info(
                "Tool call start | tool=%s | input=%s",
                tool_name,
                call_kwargs,
            )
            result = await tool.func(**call_kwargs)
            Logger.info(
                "Tool call end | tool=%s | output=%s",
                tool_name,
                result,
            )
            return result
        except StopIteration:
            Logger.warn(
                "Tool call skipped | reason=tool_not_found | tool=%s",
                tool_name,
            )
            return f"Tool '{tool_name}' not found"
        except Exception as error:  # noqa: BLE001
            Logger.error(
                "Tool call failed | tool=%s | input=%s | error=%s",
                tool_name,
                input_data,
                error,
            )
            raise

    def to_litellm_format(self) -> list[dict[str, Any]]:
        """Convert all tools to LiteLLM format."""
        return [tool.to_litellm_format() for tool in self.tools]
