"""
PitWall — Data Ingestion Script (v2)
Reads PDFs from /data, builds the Parent-Child vector index with section-aware
metadata, serializes the BM25 corpus, and saves everything to the project root.

Changes from v1:
  - Child chunk size: 250 → 300 (overlap stays at 50)
  - Section metadata: extracted from PDF filenames (A–F)
  - BM25 corpus: serialized to bm25_corpus.pkl for hybrid search

Run ONCE from the project root before starting the backend:
    python scripts/ingest.py
"""

import os
import pickle
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# ---------------------------------------------------------------------------
# Paths — always resolved relative to the project root (two levels up from
# this file: scripts/ingest.py → scripts/ → project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CHROMA_DIR   = str(PROJECT_ROOT / "chroma_parent_child_db")
PARENTS_PKL  = PROJECT_ROOT / "parents.pkl"
BM25_PKL     = PROJECT_ROOT / "bm25_corpus.pkl"

# ---------------------------------------------------------------------------
# Section label mapping — extracted from PDF filenames
# ---------------------------------------------------------------------------
SECTION_MAP = {
    "Section A": ("A", "General Provisions"),
    "Section B": ("B", "Sporting"),
    "Section C": ("C", "Technical"),
    "Section D": ("D", "Financial - F1 Teams"),
    "Section E": ("E", "Financial - PU Manufacturers"),
    "Section F": ("F", "Operational"),
}


def detect_section(source_path: str) -> tuple[str, str]:
    """Extract section code and name from the PDF filename."""
    for key, (code, name) in SECTION_MAP.items():
        if key.lower() in source_path.lower():
            return code, name
    return "Unknown", "Unknown"


# ---------------------------------------------------------------------------
# 1. Load raw FIA PDFs
# ---------------------------------------------------------------------------
print(f"1. Loading FIA PDFs from {DATA_DIR} ...")
loader = PyPDFDirectoryLoader(str(DATA_DIR))
docs   = loader.load()
print(f"   Loaded {len(docs)} pages.")

# ---------------------------------------------------------------------------
# 2. Parent chunks — large structural blocks (full articles)
# ---------------------------------------------------------------------------
parent_splitter = RecursiveCharacterTextSplitter(
    separators=["\nARTICLE ", "\nC3.", "\n\n", "\n"],
    chunk_size=3000,
    chunk_overlap=200,
)
parent_docs = parent_splitter.split_documents(docs)
print(f"2. Generated {len(parent_docs)} large Parent chunks.")

# ---------------------------------------------------------------------------
# 3. Child chunks — micro snippets for precise vector matching (300 chars)
# ---------------------------------------------------------------------------
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

child_docs    = []
parent_vault  = {}   # parent_id → full parent text
bm25_corpus   = []   # list of dicts: {"text": ..., "metadata": {...}}

print("3. Constructing Parent-Child relationships with section metadata...")

section_counts = {}

for parent in parent_docs:
    parent_id = str(uuid.uuid4())

    # Detect section from the original PDF source path
    source = parent.metadata.get("source", "")
    section_code, section_name = detect_section(source)

    # Track section distribution
    section_counts[section_code] = section_counts.get(section_code, 0) + 1

    # Store parent with section metadata
    parent_vault[parent_id] = {
        "text": parent.page_content,
        "section": section_code,
        "section_name": section_name,
    }

    for chunk in child_splitter.split_text(parent.page_content):
        metadata = {
            "parent_id": parent_id,
            "section": section_code,
            "section_name": section_name,
        }

        child_docs.append(Document(
            page_content=chunk,
            metadata=metadata,
        ))

        # Also save for BM25 corpus
        bm25_corpus.append({
            "text": chunk,
            "metadata": metadata,
        })

print(f"   Generated {len(child_docs)} child vectors (300c chunks).")
print(f"   Section distribution:")
for code in sorted(section_counts.keys()):
    print(f"     Section {code}: {section_counts[code]} parent chunks")

# ---------------------------------------------------------------------------
# 4. Embed child vectors into ChromaDB
# ---------------------------------------------------------------------------
print("4. Indexing child vectors into ChromaDB (CPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
)
Chroma.from_documents(
    documents=child_docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

# ---------------------------------------------------------------------------
# 5. Serialize parent vault to disk
# ---------------------------------------------------------------------------
print(f"5. Serializing parent vault to {PARENTS_PKL} ...")
with open(PARENTS_PKL, "wb") as f:
    pickle.dump(parent_vault, f)

# ---------------------------------------------------------------------------
# 6. Serialize BM25 corpus to disk
# ---------------------------------------------------------------------------
print(f"6. Serializing BM25 corpus ({len(bm25_corpus)} chunks) to {BM25_PKL} ...")
with open(BM25_PKL, "wb") as f:
    pickle.dump(bm25_corpus, f)

print("\n=== Ingestion complete! ===")
print("Run the backend next: uvicorn backend.api:app --reload")
