# Emergency Severity Index Agentic Triage System

This project is a FastAPI backend for running and evaluating Emergency Severity Index triage agents.

It supports two execution modes:

- a single-agent baseline
- a configuration-driven multi-agent system (MAS) built as a constrained LangGraph workflow

The backend exposes APIs for:

- starting single-agent runs
- starting MAS runs
- running single-agent test batches
- running MAS test batches
- streaming execution and test events over Server-Sent Events
- reading run outputs, metrics, and traces

## What the project does

The system takes a structured triage case as input and produces an ESI prediction.

The single-agent path runs one `AgentKernel` instance end-to-end.

The MAS path orchestrates multiple configured specialist agents as graph nodes. In the current workflow, those agents collaborate through structured handoffs and graph state rather than through an unrestricted shared conversation.

The backend persists runs, events, metrics, handoffs, and test results in SQLite by default.

## Seeded test data

On startup, the application seeds a small synthetic test set into the database.

- `10` fake single-agent test cases are seeded
- `10` fake MAS test cases are seeded
- the cases are synthetic and deidentified

These are loaded automatically during app startup from the seed files in `app/`.

## Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment configuration

Create a `.env` file in the project root.

### OpenAI / GPT

If you want to run the OpenAI-backed models, set:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

If you use Azure OpenAI instead of the standard OpenAI API, set:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

### MedGemma via self-hosted vLLM

The codebase currently uses legacy setting names for the vLLM endpoint. Even though the runtime wrapper is now named `vllm_chat`, the `.env` keys are still:

```env
LLAMA_SERVER_BASE_URL=http://your-vllm-host:8000/v1
LLAMA_SERVER_API_KEY=your_optional_api_key
LLAMA_SERVER_SERIAL_REQUESTS=false
LLAMA_SERVER_TIMEOUT_S=60
```

Use these when your MedGemma models are served behind a vLLM-compatible OpenAI-style endpoint.

### Dr7 / Doctor Seven

If you want to use Doctor Seven / Dr7-hosted MedGemma, set:

```env
DR7_API_KEY=your_dr7_api_key
DR7_MEDICAL_BASE_URL=https://dr7.ai/api/v1/medical
```

## How to run the project

Start the API server with either of the following:

```bash
python run.py
```

or:

```bash
uvicorn app.main:app --reload
```

The server starts on:

```text
http://localhost:8000
```

FastAPI interactive API docs are available at:

```text
http://localhost:8000/docs
```

## What happens on startup

When the app starts, it:

- creates the database tables if they do not exist
- applies runtime schema upgrades
- seeds the deidentified single-agent and MAS test cases

By default the database is:

```text
app.db
```

## Model selection

The model registry is defined in:

- [app/agentic/model_registry.py](/Users/adil/Documents/University/MultiAgentResearch/UseCase1ESI/app/agentic/model_registry.py:1)

This is where the backend maps model IDs to providers such as:

- OpenAI
- Dr7
- vLLM-hosted MedGemma

## If you want to host your own model

If you want to point the project at your own hosted model endpoint, the main files to check are:

- [app/config.py](/Users/adil/Documents/University/MultiAgentResearch/UseCase1ESI/app/config.py:1)
  This is where the environment variables for the endpoint URL, API key, and timeout are defined.
- [app/agentic/model_registry.py](/Users/adil/Documents/University/MultiAgentResearch/UseCase1ESI/app/agentic/model_registry.py:1)
  This is where model IDs are registered and routed to the correct provider wrapper.

If your hosted endpoint is OpenAI-compatible, updating the base URL and model mapping is usually enough.

If your hosted endpoint uses a different request or response format, you would also need to update the relevant wrapper in:

- [app/agentic/models/vllm_chat.py](/Users/adil/Documents/University/MultiAgentResearch/UseCase1ESI/app/agentic/models/vllm_chat.py:1)
- [app/agentic/models/medgemma_medical_chat.py](/Users/adil/Documents/University/MultiAgentResearch/UseCase1ESI/app/agentic/models/medgemma_medical_chat.py:1)

## Test execution APIs

The main test endpoints are:

- `POST /api/tests/runs/start` for single-agent batch tests
- `GET /api/tests/runs/{run_id}/stream` for single-agent test streaming
- `POST /api/mas-tests/runs/start` for MAS batch tests
- `GET /api/mas-tests/runs/{run_id}/stream` for MAS test streaming

The operational run endpoints are mounted under:

- `/api/agent-runs`
- `/api/mas-runs`

There is also a compatibility alias for:

- `/api/swarm-runs`

# Research CLI

The database-free research toolkit can be exercised from a clone with `PYTHONPATH=src python -m mas_slm_research.cli`. See [CLI.md](CLI.md) for validation, inspection, single-case runs, paired comparison, and the scripted offline ESI fixture. The legacy API remains in this repository until the backend-retirement milestone.
