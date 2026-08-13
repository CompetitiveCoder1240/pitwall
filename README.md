# 🏎️ PitWall — F1 Race Strategy & Regulations Agentic AI (v3)

> **PitWall** is an AI-powered F1 Race Strategy & Regulations Consultant that synthesizes **FIA 2026 Regulation PDFs**, **2022–2025 F1 Telemetry Data (92 Races)**, and **Deterministic Strategy Calculators** into real-time, verified engineering insights. Powered by **LangGraph StateGraph**, **GraphRAG Knowledge Graphs**, **Parent-Child Hybrid RAG**, and **Citation Guardrails**.

---

## ✨ Architecture Highlights (v3 Upgrades)

- **LangGraph StateGraph Engine (v3)** — State-driven agentic orchestration with parallel node execution (`router` → `regulation_retriever`, `telemetry_sql`, `strategy_calculator` → `synthesis` → `citation_guardrail`).
- **GraphRAG Knowledge Graph (v3)** — Directed NetworkX Knowledge Graph linking **721 Nodes** and **3,594 Edges** across 6 FIA 2026 Regulation Sections (A–F) for cross-section rule linkages.
- **Citation Guardrail Verification Node (v3)** — Real-time validation node that checks every generated article reference against retrieved context to eliminate hallucinations.
- **SQLite F1 Telemetry Database (v3)** — 2022–2025 telemetry extractions via FastF1 (92 races, 3,469 pit stops, 4,508 tyre stints across 26 circuits).
- **Strategy Calculation Engine (v3)** — Deterministic Python tools for pit loss math, stint projections, and Safety Car / VSC pit delta calculations.
- **Hybrid RAG + FlashRank Reranker** — Parent-Child chunking (400c child / 2,000c parent), 384D `all-MiniLM-L6-v2` embeddings, BM25 keyword matching (40%), and FlashRank Cross-Encoder reranking.
- **Next.js 16 Premium UI (`pitwall-frontend`)** — Modern dark carbon/red interface with SSE streaming, live guardrail status cards, and responsive sidebar navigation.

---

## 🏗️ Multi-Source System Flow

```text
               ┌───────────────────────────────┐
               │          USER QUERY           │
               └───────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Router Node     │ (LLM Intent Classification)
                    └──────────┬──────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│  Regulation  │        │  Telemetry   │        │   Strategy   │
│  Retriever   │        │   SQL DB     │        │  Calculator  │
│(Hybrid+Graph)│        │ (2022-2025)  │        │ (Python Math)│
└──────┬───────┘        └──────┬───────┘        └──────┬───────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Synthesis Node    │ (Gemini 2.5 Pro)
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Citation Guardrail  │ (Real-time Citation Verification)
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SSE Stream to UI    │ (Next.js 16 / Port 3000)
                    └─────────────────────┘
```

---

## 📊 Benchmark Results

Evaluated against the **50-Question Golden Test Dataset** (`golden_dataset_50.json`) derived directly from the 6 FIA 2026 Regulation PDFs:

| Metric | Target Threshold | PitWall Result | Status |
| :--- | :---: | :---: | :---: |
| **Hit Rate @ 3** | > 80.00% | **83.33%** | ✅ Passed |
| **MRR (Mean Reciprocal Rank)** | > 0.7000 | **0.8333** | ✅ Passed |
| **Context Match Rate** | > 75.00% | **83.33%** | ✅ Passed |
| **Citation Guardrail Accuracy** | 100.00% | **100.00%** | ✅ Passed |

---

## 🛠️ Tech Stack & Frameworks

### Backend Core
- **Orchestration:** LangGraph (StateGraph)
- **LLM Engine:** Gemini 2.5 Pro (via OpenRouter / OpenAI SDK)
- **API Framework:** FastAPI (Uvicorn, Server-Sent Events SSE)
- **Vector DB:** ChromaDB (384D L2-Normalized embeddings via `all-MiniLM-L6-v2`)
- **Keyword Search:** Rank-BM25
- **Reranker:** FlashRank (`ms-marco-TinyBERT-L-2-v2` Cross-Encoder)
- **Knowledge Graph:** NetworkX (`fia_knowledge_graph.pkl`)
- **Telemetry DB:** SQLite (`pitwall_telemetry.db`) & FastF1 API

### Frontend UI (`pitwall-frontend`)
- **Framework:** Next.js 16 (App Router), React 19
- **Styling:** Tailwind CSS v4, Lucide React Icons, Framer Motion
- **Streaming:** SSE custom `useSSEChat` hook

---

## ⚙️ Setup & Execution

### 1. Clone Repositories

```bash
# Backend & Frontend Monorepo
git clone https://github.com/CompetitiveCoder1240/pitwall.git
cd pitwall
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env and insert your OPENROUTER_API_KEY / LLM_MODEL
```

### 3. Start the Backend (FastAPI)

```bash
python -m venv venv
venv\Scripts\activate        # Windows (or source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

uvicorn backend.api:app --reload --port 8000
```
*Backend runs at `http://localhost:8000`. Health check: `http://localhost:8000/health`.*

### 4. Start the Frontend (Next.js 16)

```bash
cd frontend-next
npm install
npm run dev
```
*Frontend runs at `http://localhost:3000`.*

---

## 📄 License

MIT — see `LICENSE`.

> *Not an official FIA product. All regulation content is sourced from publicly available FIA documents and telemetry from FastF1.*
