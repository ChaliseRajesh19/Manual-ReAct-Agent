# Manual ReAct Agent (LangGraph StateGraph)

A hand-built ReAct-style AI assistant for BCT exam preparation (Tribhuvan University), implemented with LangGraph's low-level `StateGraph` instead of using a prebuilt agent helper.

The project includes:
- A FastAPI backend (`/ask`)
- A Streamlit chat frontend
- Multiple subject tools (syllabus, explainer, planner, predictor, past questions, routine)
- FAISS-based retrieval over local PDF content

## Why this project

This project was built to understand how a ReAct loop works internally:
- how message state is stored
- how tool calls are detected and routed
- how control returns from tools back to the LLM node

## Project structure

```text
.
├── api.py                  # FastAPI server
├── app.py                  # Streamlit UI
├── client.py               # Streamlit -> API client
├── main.py                 # CLI + graph invocation entry
├── graph/graph.py          # StateGraph definition + routing
├── nodes/
│   ├── agent_node.py       # LLM node
│   └── tool_node.py        # ToolNode wrapper
├── tools/                  # Tool implementations
├── ingest.py               # Build FAISS indexes from PDFs
├── embedding_utils.py      # Embedding model loader
├── llm_utils.py            # Groq LLM loader
├── schemas.py              # API request/response models
├── data/                   # Source PDFs (syllabus, past questions, routines)
└── faiss_index/            # Saved vector indexes
```

## Architecture flow

1. User sends a prompt.
2. `agent_node` calls the LLM with system prompt + message history.
3. If the LLM returns tool calls, graph routes to `tools`.
4. `ToolNode` executes requested tool(s).
5. Graph returns to `agent_node` with tool output in messages.
6. If no more tool calls, graph ends and final answer is returned.

## Tech stack

- Python 3.11
- LangChain + LangGraph
- Groq chat models (`llama-3.1-8b-instant` or `llama-3.3-70b-versatile`)
- FAISS (`faiss-cpu`) for retrieval
- HuggingFace sentence-transformer embeddings
- FastAPI + Uvicorn
- Streamlit
- Tavily search (routine lookup)

## Setup

### 1) Clone and create virtual environment

```bash
git clone <your-repo-url>
cd Manaul_React_Agent
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Create `.env`

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
MODEL_NAME=llama-3.1-8b-instant
```

Optional for observability:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=manual-react-agent
```

### 4) Add PDF data

Place source PDFs in these folders:
- `data/syllabus/`
- `data/past_questions/`
- `data/routines/` (optional if using web lookup only)

### 5) Build vector indexes

```bash
python ingest.py
```

Force rebuild:

```bash
python ingest.py --rebuild
```

## Running the project

### Run backend API

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API endpoint:

```http
POST /ask
Content-Type: application/json

{
	"question": "Predict likely questions for Engineering Economics",
	"session_id": "user-123"
}
```

### Run Streamlit frontend

```bash
streamlit run app.py
```

Note: `client.py` currently points to deployed API URL (`https://manual-react-agent.onrender.com/ask`).
For local development, update it to:

```python
API = "http://127.0.0.1:8000/ask"
```

### Run CLI mode

```bash
python main.py
```

## Docker

Build image:

```bash
docker build -t manual-react-agent .
```

Run container:

```bash
docker run --env-file .env -p 8000:8000 manual-react-agent
```

## Errors and problems faced while building (and fixes)

This section captures the major issues observed during development and common failures in this codebase.

### 1) `Import "fastapi" could not be resolved`

Symptom:
- Editor/linter shows unresolved import in `api.py`.

Cause:
- Dependencies not installed in the active environment.

Fix:
```bash
pip install -r requirements.txt
```
Also ensure VS Code is using the same virtual environment interpreter.

### 2) Missing API keys (`GROQ_API_KEY`, `TAVILY_API_KEY`)

Symptom:
- Runtime errors when invoking LLM or Tavily search.

Cause:
- `.env` not created or keys missing.

Fix:
- Add required keys in `.env`.
- Restart the app after changing environment variables.

### 3) FAISS index load failure / missing `index.faiss`

Symptom:
- Tool call fails when loading syllabus or past question indexes.

Cause:
- Index folder not created yet, or only one index exists.

Fix:
```bash
python ingest.py
```
Ensure PDFs exist in `data/syllabus/` and `data/past_questions/` first.

### 4) `No PDF files found ... Nothing to index.`

Symptom:
- Ingest script prints that no PDFs were found.

Cause:
- Empty data folders.

Fix:
- Add actual PDF files to required folders.
- Re-run `python ingest.py --rebuild`.

### 5) Streamlit UI cannot get response / request error

Symptom:
- Frontend returns an error text from `client.py`.

Cause:
- API URL points to production while backend is running locally, or API is down.

Fix:
- Set local API URL in `client.py` during local development:
	- `http://127.0.0.1:8000/ask`
- Confirm backend is running before launching Streamlit.

### 6) FastAPI request validation error (`422 Unprocessable Entity`)

Symptom:
- POST request fails with validation errors.

Cause:
- Missing required fields in request body.

Fix:
- Send both `question` and `session_id` exactly as defined in `schemas.py`.

### 7) ReAct loop recursion/tool-loop issues

Symptom:
- Agent repeatedly calls tools or hits graph recursion limit.

Cause:
- Prompt/tool-instruction mismatch, model behavior, or overly broad tool usage rules.

Fix:
- Tighten system prompt and enforce one tool call per turn where appropriate.
- Keep recursion limit controlled and monitor traces (LangSmith helps here).

### 8) Model sometimes outputs tool-call text instead of structured tool calls

Symptom:
- The model writes a pseudo tool call in plain text.

Cause:
- LLM function-calling inconsistency in longer contexts.

Fix/workaround:
- Keep prompts strict and concise.
- Use shorter context windows and lower temperature for tool-heavy tasks.

### 9) Current code issue in planner tool: undefined `memory`

Symptom:
- Planner tool may throw `NameError: name 'memory' is not defined`.

Cause:
- `planner_fetcher` passes `chat_history: memory`, but `memory` is not defined in that scope.

Fix options:
- Remove `chat_history` from planner input, or
- Explicitly pass chat history into the tool through state/arguments.

### 10) Token/context pressure with large retrieved text

Symptom:
- Slow responses or occasional model failures on long retrieved chunks.

Cause:
- Too much context passed to model.

Fix:
- Trim chunk size and retrieved document length.
- Keep `k` small in retriever and reduce prompt verbosity.

## Known limitations

- Tool-calling reliability still depends on model behavior.
- Planner tool currently has a scoping bug (`memory` variable).
- Frontend currently defaults to remote API endpoint in `client.py`.

## Future improvements

- Add automated tests for each tool and graph routing.
- Add better error handling and user-friendly fallback messages.
- Add proper session persistence (database/Redis) instead of in-memory dict.
- Add retry + timeout guards around external APIs.

## License

Add your preferred license here (MIT/Apache-2.0/etc.).