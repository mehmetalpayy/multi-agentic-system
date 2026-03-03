"""Agent implementations and option types for orchestration layer."""

from .a2a_host import A2AHost, A2AHostOptions
from .base import Agent, AgentOptions
from .lead_agent import LeadAgent, LeadAgentOptions
from .react import ReactAgent, ReactAgentOptions
from .supervisor import SupervisorAgent, SupervisorAgentOptions

__all__ = [
    "A2AHost",
    "A2AHostOptions",
    "Agent",
    "AgentOptions",
    "ReactAgent",
    "ReactAgentOptions",
    "LeadAgent",
    "LeadAgentOptions",
    "SupervisorAgent",
    "SupervisorAgentOptions",
]
