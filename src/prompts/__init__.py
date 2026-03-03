"""Prompt templates used by orchestration agents for tools, supervision, and summarisation."""

from .agent_prompts import (
    AGENT_SYSTEM_PROMPT_TEMPLATE,
    MATH_AGENT_SYSTEM_PROMPT,
    SUPERVISOR_PROMPT_TEMPLATE,
    WEATHER_AGENT_SYSTEM_PROMPT,
)

__all__ = [
    "SUPERVISOR_PROMPT_TEMPLATE",
    "AGENT_SYSTEM_PROMPT_TEMPLATE",
    "MATH_AGENT_SYSTEM_PROMPT",
    "WEATHER_AGENT_SYSTEM_PROMPT",
]
