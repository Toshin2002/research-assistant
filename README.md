# Autonomous Research Agent

An AI agent that takes a research goal, builds a multi-step plan, searches the web, and synthesizes findings into a structured markdown report — all served through an async REST API.

Built with **LangGraph**, **Groq**, and **FastAPI**.

---

## How It Works

The agent follows a **ReAct loop** (Reason → Act → Observe) powered by LangGraph:

```
START → plan → act → observe → reflect → act → ... → report → END
```

| Node | What it does |
|---|---|
| **plan** | Breaks the goal into 3–6 ordered research steps |
| **act** | Decides which tool to call and with what arguments |
| **observe** | Executes the tool, distills the raw output into a concise note |
| **reflect** | Evaluates whether enough information has been gathered |
| **report** | Synthesizes all notes into a structured markdown report |

The loop continues until the LLM is satisfied with the research or the iteration limit is reached.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Web search | Tavily API |
| Page fetching | requests + BeautifulSoup |
| API | FastAPI (async) |
| Database | SQLite via SQLAlchemy async + aiosqlite |
| Validation | Pydantic v2 |
| Tests | pytest + pytest-asyncio |

---

## Project Structure

```
AI_Agent/
├── app/
│   ├── agent/
│   │   ├── graph.py      # LangGraph StateGraph definition
│   │   ├── nodes.py      # plan, act, observe, reflect, report nodes
│   │   ├── runner.py     # Runs the graph as a background task
│   │   ├── state.py      # AgentState TypedDict
│   │   └── tools.py      # web_search, fetch_page, TOOL_REGISTRY
│   ├── api/
│   │   └── routes.py     # REST endpoints
│   ├── db/
│   │   ├── models.py     # SQLAlchemy ORM models
│   │   ├── repository.py # All database logic (Repository pattern)
│   │   └── session.py    # Async engine and session factory
│   ├── schemas/
│   │   └── run.py        # Pydantic request/response models
│   ├── config.py         # Settings loaded from .env
│   └── main.py           # FastAPI app entry point
├── tests/
│   ├── test_agent.py     # Unit tests for all agent nodes
│   └── test_api.py       # Integration tests for all API endpoints
├── .env.example
├── pytest.ini
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/runs` | Start a new research run |
| `GET` | `/runs` | List all runs (paginated) |
| `GET` | `/runs/{run_id}` | Get run details and steps |
| `DELETE` | `/runs/{run_id}` | Delete a run |

### Start a run

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"goal": "Compare the top AI coding assistants in 2024", "max_iterations": 5}'
```

**Response (202 Accepted):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "goal": "Compare the top AI coding assistants in 2024",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Get run result

```bash
curl http://localhost:8000/runs/{run_id}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "goal": "Compare the top AI coding assistants in 2024",
  "status": "completed",
  "report": "## Executive Summary\n...",
  "iteration_count": 5,
  "steps": [...]
}
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autonomous-research-agent.git
cd autonomous-research-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
MAX_ITERATIONS=5
DATABASE_URL=sqlite+aiosqlite:///research.db
```

- **Groq API key** — free at [console.groq.com](https://console.groq.com)
- **Tavily API key** — free tier available at [tavily.com](https://tavily.com)

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

---

## Running Tests

```bash
pytest
```

The test suite has 39 tests covering all agent nodes and API endpoints. All Groq API calls are mocked — no API keys needed to run tests.

```
tests/test_agent.py   — unit tests for plan, act, observe, reflect, report nodes
tests/test_api.py     — integration tests using in-memory SQLite
```

---

## Key Design Decisions

**Prompt-based tool calling over native Groq tool-calling API**
The LLM is instructed via the system prompt to respond with a JSON object `{"tool": "...", "arguments": {...}}`. This avoids format inconsistencies with open-source models and keeps tool execution fully under our control.

**Separated act and observe nodes**
The `act` node only decides what to do. The `observe` node executes it and distills the result. Single responsibility makes each node independently testable.

**Repository pattern for database access**
All SQL lives in `RunRepository`. Routes never touch the ORM directly. This makes the database layer swappable and easy to test with dependency injection.

**Background task execution**
The agent runs in a FastAPI `BackgroundTask` so the API returns `202 Accepted` immediately. The client polls `GET /runs/{id}` for the result.
