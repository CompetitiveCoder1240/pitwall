"""
PitWall — FastAPI Backend (v3 — LangGraph Agentic Engine)
==========================================================
Serves the F1 2026 Race Strategy & Regulation Assistant via streaming SSE.

Architecture:
  - LangGraph StateGraph orchestrates Router → [Regulation, Telemetry, Strategy] → Synthesis
  - Hybrid BM25+Vector search for FIA 2026 PDF regulations
  - SQLite queries for 2022-2025 F1 telemetry (pit stops, tyre deg, circuit data)
  - Deterministic Python race strategy calculator
  - OpenRouter cloud LLM for generation (0 MB local RAM)

Run from the project root: uvicorn backend.api:app --reload
"""

import os
import json
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
os.environ.pop("DATABASE_URL", None)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pitwall_graph = None
session_histories: dict[str, list] = defaultdict(list)


# ---------------------------------------------------------------------------
# FastAPI lifespan: build LangGraph once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global pitwall_graph
    from backend.graph import build_pitwall_graph
    pitwall_graph = build_pitwall_graph()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PitWall API",
    description="F1 2026 Race Strategy & Regulation Assistant — LangGraph Agentic Engine (v3)",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "pipeline": "ready" if pitwall_graph else "loading", "version": "v3-langgraph"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Streams the LLM response token-by-token as Server-Sent Events (SSE).
    
    The LangGraph engine orchestrates:
      1. Router node → classifies intent
      2. Regulation/Telemetry/Strategy nodes → fetch data in parallel
      3. Synthesis node → merges all context and generates the answer
    """
    if not pitwall_graph:
        raise HTTPException(status_code=503, detail="LangGraph pipeline not ready yet.")
    if not req.session_id or not req.message:
        raise HTTPException(status_code=400, detail="session_id and message are required.")

    history = session_histories[req.session_id]

    async def event_stream():
        full_response = ""
        try:
            # Invoke the LangGraph with the user's input and chat history
            result = await pitwall_graph.ainvoke({
                "user_input": req.message,
                "chat_history": history,
                "needs_regulations": False,
                "needs_telemetry": False,
                "needs_strategy": False,
                "regulation_context": "",
                "telemetry_data": "",
                "strategy_analysis": "",
                "final_response": "",
            })

            full_response = result.get("final_response", "")

            # Stream the response in chunks for a streaming UX feel
            chunk_size = 12
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Persist this exchange (trim to last 8 messages)
            history.append(HumanMessage(content=req.message))
            history.append(AIMessage(content=full_response))
            if len(history) > 8:
                session_histories[req.session_id] = history[-8:]

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clears in-memory chat history for a given session."""
    session_histories.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
