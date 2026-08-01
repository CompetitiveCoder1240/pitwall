"""
PitWall — RAG Pipeline Evaluation Script
Runs Ragas evaluation metrics over golden_dataset_50.json using the live RAG pipeline.
"""

import os
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# Core LangChain components
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_parent_child_db")
PARENTS_PKL = PROJECT_ROOT / "parents.pkl"
DATASET_JSON = PROJECT_ROOT / "golden_dataset_50.json"
RESULTS_JSON = PROJECT_ROOT / "evaluation_results.json"

print("[PitWall Eval] 1. Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
)

print("[PitWall Eval] 2. Loading Chroma vector store...")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

print("[PitWall Eval] 3. Loading Parent Vault...")
with open(PARENTS_PKL, "rb") as f:
    parent_vault = pickle.load(f)

# Two-stage retriever: Vector search (k=5) -> FlashRank rerank (top_n=3)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

print("[PitWall Eval] 4. Connecting to OpenRouter LLM...")
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_ENDPOINT"),
    temperature=0,
)

# System prompt
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


def fetch_parent_texts(child_docs):
    seen = set()
    texts = []
    for doc in child_docs:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            parent_text = parent_vault.get(pid, doc.page_content)
            texts.append(parent_text)
    return texts


def format_context_string(texts):
    return "\n\n---\n\n".join(texts)


# Assemble chain
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_context_string(fetch_parent_texts(x["retrieved_children"]))
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Load dataset
print(f"[PitWall Eval] 5. Loading Golden Dataset from {DATASET_JSON} ...")
with open(DATASET_JSON, "r", encoding="utf-8") as f:
    golden_data = json.load(f)

print(f"[PitWall Eval] Running pipeline over {len(golden_data)} test items...")

eval_samples = []
for idx, item in enumerate(golden_data, 1):
    q = item["question"]
    gt = item["ground_truth"]
    print(f"[{idx}/{len(golden_data)}] Querying: {q[:60]}...")

    # Step A: Retrieve child docs & extract parent texts
    child_docs = retriever.invoke(q)
    parent_texts = fetch_parent_texts(child_docs)

    # Step B: Generate RAG answer with rate-limit retry protection
    response = None
    for attempt in range(5):
        try:
            response = rag_chain.invoke({
                "input": q,
                "retrieved_children": child_docs,
                "chat_history": []
            })
            break
        except Exception as err:
            print(f"   [Rate limit / API error: {err}. Pausing 5s before retry...]")
            time.sleep(5)

    if response is None:
        response = "Error generating response due to rate limit."

    eval_samples.append({
        "question": q,
        "answer": response,
        "contexts": parent_texts,
        "ground_truth": gt,
        "expected_article": item.get("expected_article", "")
    })

    # Small pause to stay well within free OpenRouter rate limits
    time.sleep(2)

print(f"\n[PitWall Eval] Completed {len(eval_samples)} pipeline evaluations.")

# Save raw generation outputs to disk first
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(eval_samples, f, indent=2, ensure_ascii=False)

print(f"[PitWall Eval] Raw predictions saved to {RESULTS_JSON}")

# Convert to HuggingFace Dataset for Ragas
eval_dataset = Dataset.from_list([
    {
        "question": s["question"],
        "answer": s["answer"],
        "contexts": s["contexts"],
        "ground_truth": s["ground_truth"]
    }
    for s in eval_samples
])

print("\n[PitWall Eval] 6. Computing Ragas Evaluation Metrics...")
try:
    scores = evaluate(
        dataset=eval_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=embeddings
    )
    print("\n=========================================")
    print("       PITWALL RAG EVALUATION SCORES     ")
    print("=========================================")
    print(scores)
    print("=========================================\n")
except Exception as e:
    print(f"[PitWall Eval] Ragas metric calculation note: {e}")
