"""
PitWall — FastAPI Backend (v2)
Serves the F1 2026 Parent-Child RAG pipeline via streaming SSE.

Changes from v1:
  - Hybrid Search: BM25 (sparse) + ChromaDB (dense) via EnsembleRetriever
  - k = 6 for both retrievers
  - Section-aware parent vault (section code + name in metadata)
  - Async retriever wrapper: vector search + reranking runs in a thread pool
    so the event loop stays free for concurrent requests
  - FlashRank cross-encoder reranker stays on top of ensemble output

Run from the project root: uvicorn backend.api:app --reload
"""

import os
import asyncio
import pickle
import json
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Core LangChain components
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Hybrid search components
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# For async retriever wrapper
from langchain_core.retrievers import BaseRetriever

load_dotenv()

# ---------------------------------------------------------------------------
# Paths — always resolved relative to the project root (two levels up from
# this file: backend/api.py → backend/ → project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_parent_child_db")
PARENTS_PKL = PROJECT_ROOT / "parents.pkl"
BM25_PKL = PROJECT_ROOT / "bm25_corpus.pkl"

# ---------------------------------------------------------------------------
# Global pipeline state — built once at startup, reused for every request
# ---------------------------------------------------------------------------
rag_chain = None
session_histories: dict[str, list] = defaultdict(list)


# ---------------------------------------------------------------------------
# Async Retriever Wrapper — runs synchronous retrieval in a thread pool
# so that vector search, BM25 search, and FlashRank reranking don't block
# the FastAPI event loop during concurrent requests.
# ---------------------------------------------------------------------------
class AsyncRetrieverWrapper(BaseRetriever):
    """Wraps a synchronous retriever to run in asyncio.to_thread()."""
    sync_retriever: Any

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronous fallback — delegates to the wrapped retriever."""
        return self.sync_retriever.invoke(query)

    async def _aget_relevant_documents(
        self, query: str, **kwargs
    ) -> List[Document]:
        """Async path — offloads retrieval to a background thread."""
        return await asyncio.to_thread(self.sync_retriever.invoke, query)


def build_rag_pipeline():
    """Construct the full Parent-Child RAG chain with hybrid search."""

    print("[PitWall] Loading embedding model (CPU)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )

    print(f"[PitWall] Loading Chroma vector store from {CHROMA_DIR}...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    print(f"[PitWall] Loading parent vault from {PARENTS_PKL}...")
    with open(PARENTS_PKL, "rb") as f:
        parent_vault = pickle.load(f)

    print(f"[PitWall] Loading BM25 corpus from {BM25_PKL}...")
    with open(BM25_PKL, "rb") as f:
        bm25_corpus = pickle.load(f)

    # ── Dense retriever: ChromaDB vector search (k=6) ────────────────────
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # ── Sparse retriever: BM25 keyword search (k=6) ─────────────────────
    bm25_docs = [
        Document(page_content=item["text"], metadata=item["metadata"])
        for item in bm25_corpus
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 6

    # ── Hybrid search: 40% BM25 + 60% Vector via EnsembleRetriever ──────
    print("[PitWall] Building hybrid search (40% BM25 + 60% Vector)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    # ── Cross-encoder reranker on top of ensemble output ─────────────────
    compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    # ── Async wrapper: offload retrieval to thread pool ──────────────────
    async_retriever = AsyncRetrieverWrapper(sync_retriever=reranked_retriever)

    print("[PitWall] Connecting to OpenRouter LLM...")
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_ENDPOINT"),
        temperature=0,
        streaming=True,
    )

    # --- Prompt: question contextualization for multi-turn follow-ups ---
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question which can be "
            "understood without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, async_retriever, contextualize_q_prompt
    )

    # --- Prompt: F1 technical consultant answer generation ---
    system_prompt = (
        "You are an expert F1 technical consultant advising an engineer or fan. "
        "Analyze the following complete regulations context to fulfill the user's request. "
        "Provide a detailed response quoting or referencing specific data limits where available. "
        "If the answer cannot be found or reasonably deduced from the context, explicitly say: "
        "'I cannot locate this rule in the active 2026 regulations.' "
        "Do not extrapolate using unverified outside technical knowledge."
        "\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # --- Hierarchical resolver: child match → full parent text ---
    def fetch_and_format_parents(child_docs):
        seen_parents = set()
        formatted_context = []
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                # v2: parent_vault stores dicts with "text" key
                parent_entry = parent_vault.get(parent_id)
                if isinstance(parent_entry, dict):
                    parent_text = parent_entry.get("text", doc.page_content)
                else:
                    # Backward compat with v1 format (plain strings)
                    parent_text = parent_entry or doc.page_content
                formatted_context.append(parent_text)
        return "\n\n---\n\n".join(formatted_context)

    # --- Full LCEL chain ---
    chain = (
        RunnablePassthrough.assign(context=history_aware_retriever)
        | RunnablePassthrough.assign(
            context=(lambda x: fetch_and_format_parents(x["context"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    print("[PitWall] RAG pipeline ready (v2 — hybrid search + async).\n")
    return chain


# ---------------------------------------------------------------------------
# FastAPI lifespan: build pipeline once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global rag_chain
    rag_chain = build_rag_pipeline()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PitWall API",
    description="F1 2026 Regulations RAG Consultant — Backend (v2)",
    version="2.0.0",
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
    """Liveness probe used by the Streamlit sidebar health badge."""
    return {"status": "ok", "pipeline": "ready" if rag_chain else "loading"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Streams the LLM response token-by-token as Server-Sent Events (SSE).
    Chat history is maintained in-memory per session (last 8 messages).

    The retriever runs in a background thread (via AsyncRetrieverWrapper)
    so the event loop stays free for concurrent requests.
    """
    if not rag_chain:
        raise HTTPException(
            status_code=503, detail="RAG pipeline not ready yet.")
    if not req.session_id or not req.message:
        raise HTTPException(
            status_code=400, detail="session_id and message are required.")

    history = session_histories[req.session_id]

    async def event_stream():
        full_response = ""
        try:
            async for chunk in rag_chain.astream({
                "input": req.message,
                "chat_history": history,
            }):
                full_response += chunk
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
