# Tutoristic Graph

An AI tutor that actually remembers you. Your learning path, what you've mastered,
what you're stuck on, and every concept the tutor has taught you live in a
knowledge graph in [FalkorDB](https://docs.falkordb.com/) — not in a chat log
that evaporates when you close the tab.

Built for GraphHacks, Track 02: Agent Memory & Coordination.

## How it works

Three agents take turns, coordinated through a LangGraph state machine and a
shared graph memory (this is the hackathon part):

- **Planner** — reads your goal ("I want to learn recursion"), decides a path of
  concepts ordered by prerequisites, and creates a learning page for the goal.
- **Instructor** — teaches the current concept, picking a strategy based on what
  worked for you before (`Concept -[:TAUGHT_BY]-> Strategy` edges).
- **Examiner** — grades your answer against the concept. Score 70+ and it marks
  the concept mastered and moves you forward; otherwise it hands you back to the
  instructor to re-teach it a different way.

Every turn writes to the graph: interactions, status changes (`locked → learning
→ mastered`), agent handoffs, strategies that succeeded or failed. Come back
tomorrow in a new session and the tutor picks up exactly where you left off,
because none of that state lived in the conversation.

Each user gets their own graph (`tutor_<user_id>`), so your curriculum is
isolated from everyone else's.

## Getting started

You'll need Docker, Python 3.11+, and Node 18+.

**1. Start FalkorDB**

```bash
docker run -d --name falkor-db -p 6379:6379 falkordb/falkordb:latest
```

**2. Backend**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your-key-here   # from console.groq.com
uvicorn app.main:app --port 8000
```

Optional: `FALKORDB_URL` (defaults to `redis://localhost:6379`).

**3. Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. Your user id is generated and stored in
`localStorage` on first visit.

## What you get in the UI

- **Chat** — talk to the tutor like a tutor. Ask a goal, answer its questions,
  get explanations with worked examples. History survives reloads and new
  sessions.
- **Learning pages** — every goal becomes its own page in the sidebar with its
  own to-do list. Studying Spanish and decorators at once? They don't get
  jumbled together.
- **Graph view** — your knowledge as a React Flow graph: prerequisites on top,
  dependents below, colored by status. Drag nodes (positions persist), add or
  delete concepts and edges by hand, or hit **Auto-link** and let the LLM infer
  `REQUIRES` edges from the concepts' descriptions.
- **To-do** — click any item to cycle `locked → learning → mastered`. Mastered
  concepts go green in the graph.
- **Reset** — wipes the graph and history when you want a clean slate.

## API at a glance

| Endpoint | What it does |
| --- | --- |
| `POST /chat` | One tutor turn: `{user_id, message, goal?, is_reply?, session_id?}` |
| `GET /state/{user_id}` | Graph, to-do, pages, chat history, active agent |
| `PUT /todo` | Manually set a concept's status |
| `POST /topic` | Create a learning page by hand |
| `POST/DELETE /graph/node`, `/graph/edge` | Edit the graph |
| `PUT /graph/positions` | Save node positions from the UI |
| `POST /graph/analyze` | LLM auto-links related concepts |
| `POST /reset/{user_id}` | Wipe everything for that user |

## Project layout

```
backend/app/
  main.py                 FastAPI endpoints
  agents/tutor_graph.py   LangGraph planner / instructor / examiner
  memory/graph_memory.py  All FalkorDB reads and writes
  core/config.py          Settings (Groq key, model, FalkorDB URL)
frontend/
  app/page.tsx            Page state: chat, graph, pages
  components/Chat.tsx     Chat with markdown rendering
  components/GraphView.tsx React Flow graph + dagre layout + editing
  components/Sidebar.tsx  Learning pages and to-do
  lib/api.ts              API client
```

## Notes

- The Groq model defaults to `openai/gpt-oss-120b`; change `model` in
  `backend/app/core/config.py` if you want something else.
- FalkorDB speaks Redis protocol, so any `redis://` endpoint works — including
  Falkor Cloud.
