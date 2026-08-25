"""
PitWall — LangGraph Agentic State Graph
=========================================
Orchestrates multi-source retrieval and strategy calculation via a
LangGraph StateGraph.

Nodes:
  1. router_node       — Analyzes user intent, decides which tools to invoke
  2. regulation_node   — Hybrid BM25+Vector search on FIA 2026 PDF regulations
  3. telemetry_node    — SQLite queries on 2022-2025 F1 telemetry data
  4. strategy_node     — Deterministic pit/tyre strategy calculator
  5. synthesis_node    — Merges all tool outputs into a single LLM-generated answer

Edges:
  START → router_node → [regulation_node, telemetry_node, strategy_node] → synthesis_node → END
"""

from __future__ import annotations

import os
import asyncio
import pickle
from pathlib import Path
from typing import Any, Annotated, TypedDict

from backend.logger import logger
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from backend.tools import query_telemetry, calculate_strategy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_parent_child_db")
PARENTS_PKL = PROJECT_ROOT / "parents.pkl"
BM25_PKL = PROJECT_ROOT / "bm25_corpus.pkl"

# Load .env from project root (not cwd)
load_dotenv(PROJECT_ROOT / ".env")
os.environ.pop("DATABASE_URL", None)


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class PitWallState(TypedDict):
    """Central state dictionary passed between LangGraph nodes."""
    user_input: str
    chat_history: list
    # Router decisions
    needs_regulations: bool
    needs_telemetry: bool
    needs_strategy: bool
    # Tool outputs
    regulation_context: str
    telemetry_data: str
    strategy_analysis: str
    # Final output
    final_response: str


# ---------------------------------------------------------------------------
# Build RAG Components (called once at startup)
# ---------------------------------------------------------------------------
def build_retriever():
    """Build the hybrid BM25+Vector retriever with FlashRank reranking."""
    from chromadb.config import Settings
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if PINECONE_API_KEY:
        from langchain_pinecone import PineconeVectorStore
        logger.info("Loading Pinecone VectorStore...")
        vectorstore = PineconeVectorStore(
            index_name="pitwall-regulations",
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
    else:
        logger.info(f"Loading Chroma from {CHROMA_DIR}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

    logger.info(f"Loading BM25 corpus from {BM25_PKL}...")
    with open(BM25_PKL, "rb") as f:
        bm25_corpus = pickle.load(f)

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    bm25_docs = [
        Document(page_content=item["text"], metadata=item["metadata"])
        for item in bm25_corpus
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 6

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )

    compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    logger.info("Loading parent vault...")
    with open(PARENTS_PKL, "rb") as f:
        parent_vault = pickle.load(f)

    return reranked_retriever, parent_vault, bm25_retriever


def _extract_text(content) -> str:
    """Safely extract text from an LLM response content field.
    
    Gemini models may return content as a list of parts instead of a plain string.
    This helper normalises both formats into a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", str(part))
            for part in content
        )
    return str(content)


def build_llm():
    """Build the Gemini-connected LLM via Google AI Studio."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL", "gemini-3.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        max_output_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------
def make_router_node(llm):
    """Create the router node that classifies user intent."""

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert F1 intent classifier for PitWall.\n"
            "Analyze the user's input and determine which data sources are needed.\n"
            "Return EXACTLY three comma-separated boolean values (true/false) for:\n"
            "needs_regulations, needs_telemetry, needs_strategy\n\n"
            "Rules:\n"
            "- needs_regulations = true if the question asks about FIA rules, technical/sporting/financial "
            "specifications, wing dimensions, weight limits, parc fermé, penalties, cost cap, or regulations.\n"
            "- needs_telemetry = true if the question asks about past F1 race data (2022-2025), pit stop durations, "
            "tyre degradation, circuit-specific performance, or lap times.\n"
            "- needs_strategy = true if the question involves pit stop strategy decisions, stint projections, "
            "tyre choice optimization, or race time calculations.\n\n"
            "Examples:\n"
            "Q: 'What is the minimum car weight?' → true, false, false\n"
            "Q: 'What is the average pit stop time at Monza?' → false, true, false\n"
            "Q: 'Should I pit under safety car at Silverstone on lap 30?' → true, true, true\n"
            "Q: 'Can I replace a front wing under Parc Fermé and what is my pit loss?' → true, true, true\n"
            "Q: 'Compare tyre degradation for soft vs medium at Spa' → false, true, false\n"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def router_node(state: PitWallState) -> dict:
        """Classify user intent and set routing flags."""
        try:
            response = llm.invoke(
                router_prompt.format_messages(
                    input=state["user_input"],
                    chat_history=state.get("chat_history", []),
                )
            )
            text = _extract_text(response.content).strip().lower()

            # Parse the three boolean values
            parts = [p.strip() for p in text.split(",")]
            needs_reg = "true" in parts[0] if len(parts) > 0 else True
            needs_tel = "true" in parts[1] if len(parts) > 1 else False
            needs_str = "true" in parts[2] if len(parts) > 2 else False

            # Fallback: if nothing is detected, default to regulations
            if not needs_reg and not needs_tel and not needs_str:
                needs_reg = True
        except Exception as e:
            logger.error(f"Router LLM call failed: {e}", exc_info=True)
            needs_reg = True
            needs_tel = False
            needs_str = False

        logger.info(f"Router Decision -> Regulations: {needs_reg} | Telemetry: {needs_tel} | Strategy: {needs_str}")

        return {
            "needs_regulations": needs_reg,
            "needs_telemetry": needs_tel,
            "needs_strategy": needs_str,
        }

    return router_node


def make_regulation_node(retriever, parent_vault: dict, bm25_fallback_retriever=None):
    """Create the regulation retrieval node with GraphRAG knowledge graph traversal and fail-safe fallback."""
    import pickle
    import networkx as nx

    graph_path = os.path.join(PROJECT_ROOT, "data", "fia_knowledge_graph.pkl")
    if not os.path.exists(graph_path):
        graph_path = os.path.join(PROJECT_ROOT, "f1-rag-chatbot", "data", "fia_knowledge_graph.pkl")

    kg = None
    if os.path.exists(graph_path):
        try:
            with open(graph_path, "rb") as f:
                kg = pickle.load(f)
            logger.info(f"Loaded GraphRAG Knowledge Graph ({kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges)")
        except Exception as e:
            logger.warning(f"Could not load Knowledge Graph: {e}")

    def fetch_parent_context(child_docs: list[Document]) -> str:
        """Resolve child chunks → full parent regulation text."""
        seen_parents = set()
        formatted = []
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                entry = parent_vault.get(parent_id)
                if isinstance(entry, dict):
                    text = entry.get("text", doc.page_content)
                else:
                    text = entry or doc.page_content
                formatted.append(text)
        return "\n\n---\n\n".join(formatted) if formatted else ""

    def traverse_knowledge_graph(user_query: str, primary_context: str) -> str:
        """Traverse GraphRAG edges to find connected cross-section rules."""
        if not kg:
            return ""

        query_lower = user_query.lower()
        matched_nodes = []

        # Find matching entity or article nodes in graph
        for node, data in kg.nodes(data=True):
            node_str = str(node).lower()
            if node_str in query_lower or (len(node_str) > 4 and node_str in primary_context.lower()):
                matched_nodes.append(node)

        if not matched_nodes:
            return ""

        graph_links = []
        visited_nodes = set()

        for start_node in matched_nodes[:3]:
            # 1-hop successors and predecessors
            neighbors = list(kg.successors(start_node)) + list(kg.predecessors(start_node))
            for nbr in neighbors[:4]:
                if nbr not in visited_nodes and nbr != start_node:
                    visited_nodes.add(nbr)
                    edge_data = kg.get_edge_data(start_node, nbr) or kg.get_edge_data(nbr, start_node) or {}
                    rel = edge_data.get("relation", "CONNECTED_TO")
                    sample = kg.nodes[nbr].get("text_sample", "")
                    if sample:
                        graph_links.append(f"• Linked Node [{start_node}] --({rel})--> [{nbr}]: {sample[:200]}...")
                    else:
                        graph_links.append(f"• Linked Node [{start_node}] --({rel})--> [{nbr}]")

        if not graph_links:
            return ""

        return "\n═══ KNOWLEDGE GRAPH CROSS-SECTION LINKAGES ═══\n" + "\n".join(graph_links[:5])

    def regulation_node(state: PitWallState) -> dict:
        """Retrieve FIA 2026 regulation context via hybrid search + GraphRAG traversal."""
        if not state.get("needs_regulations", False):
            return {"regulation_context": ""}

        logger.info("Executing Regulation Node (Hybrid Search)...")

        # Fail-safe retrieval execution
        child_docs = []
        try:
            child_docs = retriever.invoke(state["user_input"])
        except Exception as e:
            logger.warning(f"Hybrid retrieval error ({e}). Falling back to BM25 search.")
            if bm25_fallback_retriever:
                try:
                    child_docs = bm25_fallback_retriever.invoke(state["user_input"])
                except Exception as e2:
                    logger.error(f"BM25 fallback error ({e2})")
                    child_docs = []

        primary_context = fetch_parent_context(child_docs)

        # GraphRAG 1-hop/2-hop cross-section traversal
        graph_context = traverse_knowledge_graph(state["user_input"], primary_context)
        combined_context = primary_context + ("\n\n" + graph_context if graph_context else "")

        logger.info(f"Regulation Node retrieved {len(child_docs)} chunks, context length: {len(combined_context)}")

        return {"regulation_context": combined_context}

    return regulation_node


def make_telemetry_node():
    """Create the telemetry database query node."""

    def telemetry_node(state: PitWallState) -> dict:
        """Query 2022-2025 telemetry data from SQLite."""
        if not state.get("needs_telemetry", False):
            return {"telemetry_data": ""}

        logger.info("Executing Telemetry Node...")

        # Use the LLM-friendly tool interface — pass the user's question
        # and let the tool extract circuit names via pattern matching
        user_input = state["user_input"].lower()

        # Simple circuit name extraction from the user query
        from backend.tools import _query_db
        circuits = _query_db("SELECT DISTINCT circuit_name FROM circuit_summaries")
        circuit_names = [c["circuit_name"] for c in circuits]

        matched_circuit = None
        for name in circuit_names:
            if name.lower() in user_input:
                matched_circuit = name
                break

        if not matched_circuit:
            # Try partial matching
            for name in circuit_names:
                for word in name.lower().split():
                    if len(word) > 3 and word in user_input:
                        matched_circuit = name
                        break
                if matched_circuit:
                    break

        if not matched_circuit:
            return {"telemetry_data": "No specific circuit identified in the query. Available circuits can be listed with the telemetry tool."}

        # Get circuit summary
        summary = query_telemetry.invoke({
            "circuit_name": matched_circuit,
            "query_type": "summary",
        })

        result = summary
        logger.info(f"Telemetry Node result length: {len(result)}")

        return {"telemetry_data": result}

    return telemetry_node


def make_strategy_node():
    """Create the strategy calculator node."""

    def strategy_node(state: PitWallState) -> dict:
        """Run pit/tyre strategy calculations."""
        if not state.get("needs_strategy", False):
            return {"strategy_analysis": ""}

        logger.info("Executing Strategy Node...")

        user_input = state["user_input"].lower()

        # Extract circuit name
        from backend.tools import _query_db
        circuits = _query_db("SELECT DISTINCT circuit_name FROM circuit_summaries")
        circuit_names = [c["circuit_name"] for c in circuits]

        matched_circuit = None
        for name in circuit_names:
            if name.lower() in user_input:
                matched_circuit = name
                break
        if not matched_circuit:
            for name in circuit_names:
                for word in name.lower().split():
                    if len(word) > 3 and word in user_input:
                        matched_circuit = name
                        break
                if matched_circuit:
                    break

        if not matched_circuit:
            return {"strategy_analysis": "Could not identify a specific circuit for strategy calculation."}

        # Extract lap numbers from query (simple regex-free parsing)
        import re
        lap_matches = re.findall(r"lap\s*(\d+)", user_input)
        total_matches = re.findall(r"(\d+)\s*(?:total\s*)?laps", user_input)

        current_lap = int(lap_matches[0]) if lap_matches else 25
        total_laps = int(total_matches[0]) if total_matches else 55

        # Detect flag condition
        if "safety car" in user_input or "sc " in user_input:
            flag = "safety_car"
        elif "vsc" in user_input or "virtual" in user_input:
            flag = "vsc"
        else:
            flag = "green"

        # Detect compound
        if "soft" in user_input:
            compound = "SOFT"
        elif "hard" in user_input:
            compound = "HARD"
        else:
            compound = "MEDIUM"

        result = calculate_strategy.invoke({
            "circuit_name": matched_circuit,
            "current_lap": current_lap,
            "total_laps": total_laps,
            "flag_condition": flag,
            "target_compound": compound,
        })

        logger.info(f"Strategy Node result length: {len(result)}")

        return {"strategy_analysis": result}

    return strategy_node


def make_synthesis_node(llm):
    """Create the synthesis node that merges all tool outputs."""

    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are PitWall, an expert F1 Race Strategy and Regulations Consultant.\n"
            "You have access to multiple data sources and must synthesize them into a "
            "clear, actionable response.\n\n"
            "AVAILABLE CONTEXT (use only what is provided, do not fabricate data):\n\n"
            "{context_block}\n\n"
            "INSTRUCTIONS:\n"
            "- If regulation context is provided, cite specific FIA articles and sections.\n"
            "- If telemetry data is provided, reference the exact numbers.\n"
            "- If strategy analysis is provided, integrate the pit loss and stint projections.\n"
            "- Structure your response with clear sections when multiple data sources are used.\n"
            "- If you cannot answer from the provided context, explicitly say so.\n"
            "- Be concise but thorough. Use a professional race engineer tone."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def synthesis_node(state: PitWallState) -> dict:
        """Merge all tool outputs and generate final response."""
        # Build context block from available tool outputs
        context_parts = []

        if state.get("regulation_context"):
            context_parts.append(
                "═══ FIA 2026 REGULATION CONTEXT ═══\n" + state["regulation_context"]
            )
        if state.get("telemetry_data"):
            context_parts.append(
                "═══ TELEMETRY DATA (2022-2025) ═══\n" + state["telemetry_data"]
            )
        if state.get("strategy_analysis"):
            context_parts.append(
                "═══ STRATEGY ANALYSIS ═══\n" + state["strategy_analysis"]
            )

        if not context_parts:
            context_block = "No specific data retrieved. Answer based on general F1 knowledge."
        else:
            context_block = "\n\n".join(context_parts)

        # Generate response
        logger.info("Executing Synthesis Node...")
        try:
            response = llm.invoke(
                synthesis_prompt.format_messages(
                    context_block=context_block,
                    input=state["user_input"],
                    chat_history=state.get("chat_history", []),
                )
            )
            final_response = _extract_text(response.content)
        except Exception as e:
            logger.error(f"Synthesis LLM call failed: {e}", exc_info=True)
            final_response = "I'm sorry, an error occurred while generating your response. Please try again."

        return {"final_response": final_response}

    return synthesis_node


def make_citation_guardrail_node():
    """Create the citation guardrail verification node."""

    def citation_guardrail_node(state: PitWallState) -> dict:
        """
        Validate every regulation citation in the final_response against
        the retrieved regulation_context to detect hallucinations.
        """
        final_response = state.get("final_response", "")
        regulation_context = state.get("regulation_context", "")

        if not final_response or not regulation_context:
            return {"final_response": final_response}

        import re
        # Find all article citations like "Article B3.5.3.a", "Article 40.2", "Article C3.2"
        cited_articles = re.findall(r"Article\s+([A-Z0-9\.]+)", final_response, re.IGNORECASE)

        if not cited_articles:
            return {"final_response": final_response}

        verified_count = 0
        unverified_citations = []

        for article_num in cited_articles:
            # Clean trailing punctuation
            clean_num = article_num.rstrip(".,;")
            # Check if this article number or base section appears in regulation_context
            if clean_num in regulation_context or clean_num.lower() in regulation_context.lower():
                verified_count += 1
            else:
                unverified_citations.append(clean_num)

        # Build guardrail validation badge/footer
        total_cited = len(cited_articles)
        if unverified_citations:
            # Replace hallucinated citations in text or append warning
            warning_msg = (
                f"\n\n⚠️ [Citation Guardrail]: {total_cited - len(unverified_citations)}/{total_cited} "
                f"citations verified. Unverified citation(s): {', '.join(unverified_citations)}."
            )
            updated_response = final_response + warning_msg
        else:
            badge_msg = (
                f"\n\n🛡️ [Citation Guardrail]: {verified_count}/{total_cited} "
                "citations verified against official 2026 FIA rulebooks."
            )
            updated_response = final_response + badge_msg

        logger.info(f"Citation Guardrail: {verified_count}/{total_cited} citations verified")

        return {"final_response": updated_response}

    return citation_guardrail_node


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------
def build_pitwall_graph():
    """
    Construct and compile the full PitWall LangGraph StateGraph.

    Returns:
        compiled_graph: The compiled LangGraph ready for .invoke() or .astream()
    """
    logger.info("Building LangGraph StateGraph...")

    # Build components
    retriever, parent_vault, bm25_retriever = build_retriever()
    llm = build_llm()

    # Create nodes
    router = make_router_node(llm)
    regulation = make_regulation_node(retriever, parent_vault, bm25_retriever)
    telemetry = make_telemetry_node()
    strategy = make_strategy_node()
    synthesis = make_synthesis_node(llm)
    citation_guardrail = make_citation_guardrail_node()

    # Define graph
    graph = StateGraph(PitWallState)

    # Add nodes
    graph.add_node("router", router)
    graph.add_node("regulation_retriever", regulation)
    graph.add_node("telemetry_sql", telemetry)
    graph.add_node("strategy_calculator", strategy)
    graph.add_node("synthesis", synthesis)
    graph.add_node("citation_guardrail", citation_guardrail)

    # Define edges
    graph.set_entry_point("router")

    # After router, always run all three tool nodes
    graph.add_edge("router", "regulation_retriever")
    graph.add_edge("router", "telemetry_sql")
    graph.add_edge("router", "strategy_calculator")

    # All tool nodes feed into synthesis
    graph.add_edge("regulation_retriever", "synthesis")
    graph.add_edge("telemetry_sql", "synthesis")
    graph.add_edge("strategy_calculator", "synthesis")

    # Synthesis feeds into Citation Guardrail node
    graph.add_edge("synthesis", "citation_guardrail")
    graph.add_edge("citation_guardrail", END)

    # Compile
    compiled = graph.compile()
    logger.info("LangGraph compiled successfully.")

    return compiled

