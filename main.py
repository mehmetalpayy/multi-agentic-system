"""Minimal example wiring for two React agents and one supervisor."""

import asyncio
import uuid

from src.agents import (
    LeadAgent,
    LeadAgentOptions,
    ReactAgent,
    ReactAgentOptions,
    SupervisorAgent,
    SupervisorAgentOptions,
)
from src.history import InMemoryChatHistory
from src.prompts import MATH_AGENT_SYSTEM_PROMPT, WEATHER_AGENT_SYSTEM_PROMPT
from src.tools import (
    add_numbers,
    divide_numbers,
    multiply_numbers,
    square_root,
    subtract_numbers,
    weather_lookup_tool,
)
from src.utils import ConversationMessage, ParticipantRole


async def process_request(
    supervisor: SupervisorAgent,
    storage: InMemoryChatHistory,
    user_input: str,
    user_id: str,
    session_id: str,
) -> str:
    """Call supervisor and return the final response text."""
    await storage.save_chat_message(
        user_id=user_id,
        session_id=session_id,
        new_message=ConversationMessage(
            role=ParticipantRole.USER.value,
            content=[{"text": user_input}],
        ),
    )
    chat_history = await storage.fetch_chat_messages(
        user_id=user_id, session_id=session_id
    )
    model_history = chat_history[:-1] if chat_history else []

    response = await supervisor.process_request(
        input_text=user_input,
        user_id=user_id,
        session_id=session_id,
        chat_history=model_history,
    )

    response_text = response.text or ""
    await storage.save_chat_message(
        user_id=user_id,
        session_id=session_id,
        new_message=ConversationMessage(
            role=ParticipantRole.ASSISTANT.value,
            content=[{"text": response_text}],
        ),
    )

    return response_text


async def run_interactive_session() -> None:
    storage = InMemoryChatHistory()

    weather_agent = ReactAgent(
        ReactAgentOptions(
            id=None,
            name="Weather Agent",
            description="Provides basic weather information using weather tools.",
            model="gpt-5.1",
            tools=[weather_lookup_tool],
            streaming=False,
            custom_system_prompt={"template": WEATHER_AGENT_SYSTEM_PROMPT},
        )
    )

    math_agent = ReactAgent(
        ReactAgentOptions(
            id=None,
            name="Math Agent",
            description="Handles arithmetic and basic math operations.",
            model="gpt-5.1",
            tools=[
                add_numbers,
                subtract_numbers,
                multiply_numbers,
                divide_numbers,
                square_root,
            ],
            streaming=False,
            custom_system_prompt={"template": MATH_AGENT_SYSTEM_PROMPT},
        )
    )

    lead_agent = LeadAgent(
        LeadAgentOptions(
            id=None,
            name="Lead Agent",
            model="azure/responses/gpt-5.1",
            description="Routes tasks to specialist agents and combines outputs.",
            streaming=False,
        )
    )

    supervisor = SupervisorAgent(
        SupervisorAgentOptions(
            id=None,
            name="Supervisor",
            description="Coordinates multi-agent execution.",
            lead_agent=lead_agent,
            team=[weather_agent, math_agent],
            storage=storage,
            trace=True,
        )
    )

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    print("Interactive session started. Type 'exit' to quit.")
    while True:
        user_input = (await asyncio.to_thread(input, "You: ")).strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Session ended.")
            break

        response_text = await process_request(
            supervisor=supervisor,
            storage=storage,
            user_input=user_input,
            user_id=user_id,
            session_id=session_id,
        )
        print(f"\nSupervisor: {response_text}")


def main() -> None:
    asyncio.run(run_interactive_session())


if __name__ == "__main__":
    main()
