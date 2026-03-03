<h1 align="center"><strong>Multi-Agentic System</strong></h1>

## Overview

Welcome to the **multi-agentic-system** repository.

This project is a lightweight multi-agent orchestration setup where:

- a **Supervisor Agent** coordinates execution,
- a **Lead Agent** decides routing/tool usage,
- specialized **React Agents** (e.g. Weather, Math) handle domain tasks,
- chat state is persisted in **InMemoryChatHistory** for interactive sessions.

The default demo runs in terminal and supports continuous conversation until you type `exit`.

## Task Description

The objective of this repository is to provide a clean, extensible baseline for:

- orchestrating multiple LLM agents,
- delegating tasks with tool-calling,
- tracking conversation history,
- and observing execution through structured logs and token usage metadata.

## Repository Structure

1. **`main.py`**: Interactive terminal app; wires agents and runs request loop.
2. **`src/agents/`**: Agent implementations:
   - `supervisor.py` (orchestration and dispatch)
   - `lead_agent.py` (LiteLLM-based planning/tool loop)
   - `react.py` (LangChain/LangGraph style specialist agents)
   - `a2a_host.py` (A2A hosting support)
3. **`src/tools/`**: Built-in tools (weather + math).
4. **`src/prompts/`**: System prompts for supervisor, generic agent, weather, math.
5. **`src/history/`**: History abstraction + in-memory backend.
6. **`src/core/settings.py`**: App/runtime config (`BaseSettings`).
7. **`env.py`**: Secret config (`AZURE_OPENAI_API_KEY`).
8. **`src/utils/`**: Logger, tool abstractions, shared types.
9. **`src/callbacks/`**: Agent callback interface and hooks.
10. **`Makefile`**: `run`, `lint`, `format` shortcuts.
11. **`pyproject.toml`**: Dependencies and Ruff configuration.

## System Flow

### 1) User Request Flow

1. User enters input in terminal.
2. Input is saved to history storage.
3. `SupervisorAgent` receives request and prepares context.
4. `LeadAgent` decides whether to call `send_messages` tool.
5. Supervisor dispatches messages to relevant specialist agents.
6. Specialist agents run with their own tools/prompts.
7. Lead consolidates responses and returns final answer.
8. Final assistant response is saved to history and printed.

### 2) Tool Execution Flow

1. Lead emits tool calls.
2. `AgentTools.tool_handler(...)` executes mapped Python tools.
3. Tool input/output is logged (`Tool call start/end`).
4. Tool outputs are injected back into LLM conversation.

### 3) History Flow

1. User-facing history is stored by session.
2. Agent-internal messages are stored in agent-scoped keys.
3. `main.py` avoids duplicating the current user message in prompt history.

## Installation and Setup

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd multi-agentic-system
```

### 2. Create Environment File

Copy the example and fill secrets:

```bash
cp .env.example .env
```

Minimum required:

```env
AZURE_OPENAI_API_KEY=...
```

Recommended runtime config values (read by `src/core/settings.py`):

```env
AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_DEPLOYMENT_NAME=<model_name>
AZURE_API_VERSION=<api_version>

AGENTS_LLM_MAX_TOKENS=4096
AGENTS_LLM_TEMPERATURE=0.1
AGENTS_LLM_TOP_P=0.9

LOG_LEVEL=INFO
LOG_FILE=default.log
```

### 3. Install Dependencies

With `uv`:

```bash
uv sync
```

## Run

Start interactive session:

```bash
make run
```

You can then chat continuously:

- `You: What's the weather in Istanbul right now?`
- `You: Multiply 17 by 25`
- `You: exit`

## Development Commands

Lint:

```bash
make lint
```

Format:

```bash
make format
```

## Default Example Agents

`main.py` wires these by default:

1. **Weather Agent** (`ReactAgent`)
   - Tool: `weather_lookup_tool`
   - Prompt: `WEATHER_AGENT_SYSTEM_PROMPT`
2. **Math Agent** (`ReactAgent`)
   - Tools: `add_numbers`, `subtract_numbers`, `multiply_numbers`, `divide_numbers`, `square_root`
   - Prompt: `MATH_AGENT_SYSTEM_PROMPT`
3. **Lead Agent**
   - Model: `azure/responses/gpt-5.1`
4. **Supervisor Agent**
   - Routes across team and returns final response.

## Logging and Observability

Logs are configured in `src/core/settings.py` and written to:

- console
- `default.log` (or `LOG_FILE`)

Useful log lines include:

- supervisor lifecycle (`process start`, `dispatch start/done`)
- tool-level input/output (`Tool call start/end`)
- agent token usage (`token_usage=...`)

## Troubleshooting

### 1) `NotFoundError: Resource not found`

Usually Azure config mismatch:

- wrong `AZURE_ENDPOINT`
- wrong `AZURE_DEPLOYMENT_NAME`
- unsupported `AZURE_API_VERSION`

### 2) Missing input/parameter errors from Azure Responses API

Ensure model/provider path matches the request style in `LeadAgent` and your Azure deployment supports it.

### 3) Duplicate history or duplicated prompts

History is now separated by agent scope and user-session scope. If this reappears, check `src/history/in_memory.py` keying logic.

## Contributing

1. Create a branch.
2. Implement changes.
3. Run:
   - `make lint`
   - `make format`
4. Open a PR with:
   - what changed,
   - why it changed,
   - and how it was tested.
