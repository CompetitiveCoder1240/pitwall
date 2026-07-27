"""
PitWall — Data Ingestion Script
Reads PDFs from /data, builds the Parent-Child vector index, and saves
chroma_parent_child_db/ + parents.pkl to the project root.

Run ONCE from the project root before starting the backend:
    python scripts/ingest.py
"""

import os
import pickle
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
# 3. Child chunks — micro snippets for precise vector matching
# ---------------------------------------------------------------------------
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

child_docs   = []
parent_vault = {}   # parent_id → full parent text

print("3. Constructing Parent-Child relationships...")
for parent in parent_docs:
    parent_id = str(uuid.uuid4())
    parent_vault[parent_id] = parent.page_content

    for chunk in child_splitter.split_text(parent.page_content):
        child_docs.append(Document(
            page_content=chunk,
            metadata={"parent_id": parent_id},
        ))

print(f"   Generated {len(child_docs)} micro Child vectors.")

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

print("\n=== Ingestion complete! ===")
print("Run the backend next: uvicorn backend.api:app --reload")
