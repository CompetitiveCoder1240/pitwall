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
import re
import asyncio
import sqlite3
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
import jwt

from backend.logger import logger
from backend.auth import (
    get_password_hash, verify_password, 
    create_access_token, create_refresh_token, get_current_user,
    SECRET_KEY, ALGORITHM, DB_PATH
)

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
    logger.info("Starting up FastAPI application...")
    from backend.graph import build_pitwall_graph
    pitwall_graph = build_pitwall_graph()
    logger.info("LangGraph pipeline successfully loaded.")
    yield
    logger.info("Shutting down FastAPI application...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PitWall API",
    description="F1 2026 Race Strategy & Regulation Assistant — LangGraph Agentic Engine (v3)",
    version="3.0.0",
    lifespan=lifespan,
)

# Setup Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class RefreshRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "pipeline": "ready" if pitwall_graph else "loading", "version": "v3-langgraph"}


# ---------------------------------------------------------------------------
# Security Helpers
# ---------------------------------------------------------------------------
def sanitize_input(text: str) -> str:
    """Removes non-printable characters and limits length to 1000 characters."""
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    result = text[:1000].strip()
    logger.debug(f"Input sanitized - Original length: {len(text)}, Sanitized length: {len(result)}")
    return result

def check_prompt_injection(text: str):
    """Basic heuristic to block common prompt injection attempts."""
    patterns = [
        r"ignore (?:all )?previous instructions",
        r"you are now (?:a|an)?",
        r"system prompt",
        r"bypass rules",
        r"new instructions:",
        r"disregard previous",
        r"roleplay as"
    ]
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            logger.warning(f"Prompt injection detected! Matched pattern: {pattern}")
            raise HTTPException(status_code=400, detail="Security alert: Prompt injection detected.")

def validate_password(password: str):
    if len(password) < 8:
        logger.warning("Password validation failed for registration")
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
    if len(password) > 72:
        logger.warning("Password validation failed for registration")
        raise HTTPException(status_code=400, detail="Password cannot be longer than 72 characters.")
    if not re.search(r"[A-Z]", password):
        logger.warning("Password validation failed for registration")
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter.")
    if not re.search(r"[0-9]", password):
        logger.warning("Password validation failed for registration")
        raise HTTPException(status_code=400, detail="Password must contain at least one number.")
    if not re.search(r"[^a-zA-Z0-9]", password):
        logger.warning("Password validation failed for registration")
        raise HTTPException(status_code=400, detail="Password must contain at least one special character.")

@app.post("/register")
async def register_user(user: UserCreate):
    validate_password(user.password)
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    
    if SUPABASE_URL:
        import psycopg2
        try:
            conn = psycopg2.connect(SUPABASE_URL)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", 
                           (user.username, get_password_hash(user.password)))
            conn.commit()
            conn.close()
            logger.info(f"New user registered in Supabase: {user.username}")
        except psycopg2.IntegrityError:
            if 'conn' in locals(): conn.close()
            logger.warning(f"Registration failed - username already exists: {user.username}")
            raise HTTPException(status_code=400, detail="Username already registered")
    else:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                           (user.username, get_password_hash(user.password)))
            conn.commit()
            logger.info(f"New user registered: {user.username}")
        except sqlite3.IntegrityError:
            conn.close()
            logger.warning(f"Registration failed - username already exists: {user.username}")
            raise HTTPException(status_code=400, detail="Username already registered")
        conn.close()
    return {"message": "User registered successfully"}

@app.post("/login", response_model=Token)
async def login_for_access_token(user: UserLogin):
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    db_user = None
    
    if SUPABASE_URL:
        import psycopg2
        conn = psycopg2.connect(SUPABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = %s", (user.username,))
        db_user = cursor.fetchone()
        conn.close()
    else:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (user.username,))
        db_user = cursor.fetchone()
        conn.close()
    
    if not db_user or not verify_password(user.password, db_user[0]):
        logger.warning(f"Failed login attempt for: {user.username}")
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
        
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    logger.info(f"User logged in: {user.username}")
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/refresh", response_model=Token)
async def refresh_access_token(req: RefreshRequest):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(req.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        if username is None or token_type != "refresh":
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    access_token = create_access_token(data={"sub": username})
    refresh_token = create_refresh_token(data={"sub": username})
    
    logger.info(f"Token refreshed for user: {username}")
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest, current_user: str = Depends(get_current_user)):
    """
    Streams the LLM response token-by-token as Server-Sent Events (SSE).
    
    The LangGraph engine orchestrates:
      1. Router node → classifies intent
      2. Regulation/Telemetry/Strategy nodes → fetch data in parallel
      3. Synthesis node → merges all context and generates the answer
    """
    logger.info(f"Incoming chat request - Session: {req.session_id} | User: {current_user} | Length: {len(req.message)}")
    if not pitwall_graph:
        raise HTTPException(status_code=503, detail="LangGraph pipeline not ready yet.")
    if not req.session_id or not req.message:
        raise HTTPException(status_code=400, detail="session_id and message are required.")

    clean_message = sanitize_input(req.message)
    if not clean_message:
        raise HTTPException(status_code=400, detail="Empty or invalid message after sanitization.")
    check_prompt_injection(clean_message)

    history = session_histories[req.session_id]

    async def event_stream():
        full_response = ""
        try:
            # Invoke the LangGraph with the user's input and chat history
            result = await pitwall_graph.ainvoke({
                "user_input": clean_message,
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
            logger.info(f"Response generated for {current_user} - Length: {len(full_response)}")

            # Stream the response in chunks for a streaming UX feel
            chunk_size = 12
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i + chunk_size]
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                await asyncio.sleep(0.015)  # Simulate network streaming delay

            # Persist this exchange (trim to last 8 messages)
            history.append(HumanMessage(content=clean_message))
            history.append(AIMessage(content=full_response))
            if len(history) > 8:
                session_histories[req.session_id] = history[-8:]

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Error during graph execution or streaming: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': 'An internal server error occurred while processing your request. Please try again.'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
def clear_session(session_id: str, current_user: str = Depends(get_current_user)):
    """Clears in-memory chat history for a given session."""
    session_histories.pop(session_id, None)
    logger.info(f"Session cleared: {session_id} by user: {current_user}")
    return {"status": "cleared", "session_id": session_id}
