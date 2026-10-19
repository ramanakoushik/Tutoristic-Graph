from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.tutor_graph import TutorGraph
from app.memory.graph_memory import GraphMemory

app = FastAPI(title="Tutoristic Graph API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mem = GraphMemory()
tutor = TutorGraph()


class ChatRequest(BaseModel):
    user_id: str
    message: str
    goal: Optional[str] = None
    is_reply: Optional[bool] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    state: Dict[str, Any]
    next_agent: Optional[str] = None
    score: Optional[int] = None
    feedback: Optional[str] = None


class TodoUpdate(BaseModel):
    user_id: str
    concept_id: str
    status: str = Field(pattern=r"^(locked|learning|mastered)$")


class CreateTopicRequest(BaseModel):
    user_id: str
    goal: str
    title: Optional[str] = None


class ActivatePageRequest(BaseModel):
    user_id: str
    page_id: str


class NodeCreate(BaseModel):
    user_id: str
    labels: List[str]
    properties: Dict[str, Any]


class NodeDelete(BaseModel):
    user_id: str
    node_id: str


class EdgeCreate(BaseModel):
    user_id: str
    source_id: str
    target_id: str
    edge_type: str
    properties: Optional[Dict[str, Any]] = None


class EdgeDelete(BaseModel):
    user_id: str
    source_id: str
    target_id: str
    edge_type: str


class PositionsSave(BaseModel):
    user_id: str
    positions: List[Dict[str, Any]]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        res = tutor.run_turn(
            user_id=req.user_id,
            message=req.message,
            goal=req.goal,
            is_reply=req.is_reply,
            session_id=req.session_id,
        )
        return ChatResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state/{user_id}")
async def get_state(user_id: str):
    try:
        return mem.get_state(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/todo")
async def update_todo(body: TodoUpdate):
    try:
        mem.set_concept_status(body.user_id, body.concept_id, body.status)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/topic")
async def create_topic(body: CreateTopicRequest):
    try:
        pid = mem.create_page(body.user_id, body.goal, body.title)
        return {"ok": True, "page_id": pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/page/activate")
async def activate_page(body: ActivatePageRequest):
    try:
        mem.activate_page(body.user_id, body.page_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/node")
async def add_graph_node(body: NodeCreate):
    try:
        nid = mem.add_node(body.user_id, body.labels, body.properties)
        return {"ok": True, "node_id": nid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/graph/node")
async def delete_graph_node(body: NodeDelete):
    try:
        ok = mem.delete_node(body.user_id, body.node_id)
        return {"ok": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/edge")
async def add_graph_edge(body: EdgeCreate):
    try:
        ok = mem.add_edge(body.user_id, body.source_id, body.target_id, body.edge_type, body.properties)
        return {"ok": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/graph/edge")
async def delete_graph_edge(body: EdgeDelete):
    try:
        ok = mem.delete_edge(body.user_id, body.source_id, body.target_id, body.edge_type)
        return {"ok": ok}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/graph/positions")
async def save_graph_positions(body: PositionsSave):
    try:
        mem.save_positions(body.user_id, body.positions)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/analyze")
async def analyze_graph_links(user_id: str):
    try:
        created = mem.analyze_links(user_id)
        return {"ok": True, "created": created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{user_id}")
async def reset_user(user_id: str):
    try:
        mem.reset(user_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

