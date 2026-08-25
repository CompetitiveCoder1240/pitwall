import os
import pickle
import uuid
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
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
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    import time
    
    vectorstore = PineconeVectorStore(
        index_name="pitwall-regulations",
        embedding=embeddings
    )
    
    batch_size = 50
    total_batches = (len(child_docs) + batch_size - 1) // batch_size
    print(f"Uploading in {total_batches} batches to respect Google's 100 RPM rate limit...")
    
    for i in range(0, len(child_docs), batch_size):
        batch = child_docs[i:i+batch_size]
        print(f"  -> Uploading batch {i//batch_size + 1}/{total_batches} ({len(batch)} chunks)...")
        try:
            vectorstore.add_documents(batch)
            time.sleep(1.5)  # Sleep 1.5s to ensure we stay under 100 requests per minute
        except Exception as e:
            print(f"Rate limit hit! Sleeping for 15 seconds before retrying...")
            time.sleep(15)
            vectorstore.add_documents(batch)
    print("Serializing parent vault and BM25 corpus...")
    with open(PARENTS_PKL, "wb") as f: pickle.dump(parent_vault, f)
    with open(BM25_PKL, "wb") as f: pickle.dump(bm25_corpus, f)

    print("Pinecone ingestion complete!")
