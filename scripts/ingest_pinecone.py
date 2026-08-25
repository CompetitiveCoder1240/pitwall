import os
import pickle
import uuid
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARENTS_PKL = PROJECT_ROOT / "parents.pkl"
BM25_PKL = PROJECT_ROOT / "bm25_corpus.pkl"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

SECTION_MAP = {
    "Section A": ("A", "General Provisions"),
    "Section B": ("B", "Sporting"),
    "Section C": ("C", "Technical"),
    "Section D": ("D", "Financial - F1 Teams"),
    "Section E": ("E", "Financial - PU Manufacturers"),
    "Section F": ("F", "Operational"),
}

def detect_section(source_path: str) -> tuple[str, str]:
    for key, (code, name) in SECTION_MAP.items():
        if key.lower() in source_path.lower():
            return code, name
    return "Unknown", "Unknown"

if __name__ == "__main__":
    if not PINECONE_API_KEY:
        print("ERROR: PINECONE_API_KEY not found.")
        exit(1)

    print(f"1. Loading FIA PDFs from {DATA_DIR} ...")
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    docs = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\nARTICLE ", "\nC3.", "\n\n", "\n"],
        chunk_size=3000,
        chunk_overlap=200,
    )
    parent_docs = parent_splitter.split_documents(docs)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    child_docs = []
    parent_vault = {}
    bm25_corpus = []

    for parent in parent_docs:
        parent_id = str(uuid.uuid4())
        source = parent.metadata.get("source", "")
        section_code, section_name = detect_section(source)

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
            child_docs.append(Document(page_content=chunk, metadata=metadata))
            bm25_corpus.append({"text": chunk, "metadata": metadata})

    print(f"Indexing {len(child_docs)} chunks to Pinecone...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    PineconeVectorStore.from_documents(
        documents=child_docs,
        embedding=embeddings,
        index_name="pitwall-regulations"
    )

    print("Serializing parent vault and BM25 corpus...")
    with open(PARENTS_PKL, "wb") as f: pickle.dump(parent_vault, f)
    with open(BM25_PKL, "wb") as f: pickle.dump(bm25_corpus, f)

    print("Pinecone ingestion complete!")
