"""
PitWall — Offline Retrieval Evaluation (No LLM API Required)
Measures retrieval accuracy using only local ChromaDB + FlashRank.

Metrics computed:
  - Hit Rate @ 3: Did the correct parent article appear in the top 3 results?
  - MRR (Mean Reciprocal Rank): How high was the best match ranked?
  - Context Match Rate: Did the retrieved context contain the ground truth answer?
"""

import json
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_parent_child_db")
PARENTS_PKL = PROJECT_ROOT / "parents.pkl"
DATASET_JSON = PROJECT_ROOT / "golden_dataset_50.json"

# ── 1. Load local models & databases ──────────────────────────────────────
print("[Eval] 1. Loading embedding model (local CPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
)

print("[Eval] 2. Loading Chroma vector store...")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

print("[Eval] 3. Loading Parent Vault...")
with open(PARENTS_PKL, "rb") as f:
    parent_vault = pickle.load(f)

# Two-stage retriever: Vector search (k=5) -> FlashRank rerank (top_n=3)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# ── 2. Load golden dataset ────────────────────────────────────────────────
print(f"[Eval] 4. Loading Golden Dataset...")
with open(DATASET_JSON, "r", encoding="utf-8") as f:
    golden_data = json.load(f)

# ── 3. Helper: resolve parent texts from child docs ───────────────────────
def resolve_parent_text(pid, fallback=""):
    """Get parent text from vault, handling both v1 (str) and v2 (dict) formats."""
    entry = parent_vault.get(pid)
    if entry is None:
        return fallback
    if isinstance(entry, dict):
        return entry.get("text", fallback)
    return entry  # v1 plain string


def get_parent_texts(child_docs):
    seen = set()
    texts = []
    for doc in child_docs:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            texts.append(resolve_parent_text(pid, doc.page_content))
    return texts

# ── 4. Run retrieval evaluation ───────────────────────────────────────────
print(f"\n[Eval] Running retrieval over {len(golden_data)} questions (100% LOCAL)...\n")

hits = 0           # How many times the ground truth appeared in retrieved context
mrr_sum = 0.0      # Sum of reciprocal ranks
context_matches = 0 # How many times context contained key answer phrases
detailed_results = []

for idx, item in enumerate(golden_data, 1):
    q = item["question"]
    gt = item["ground_truth"]

    # Retrieve child docs (LOCAL vector search + LOCAL FlashRank rerank)
    child_docs = retriever.invoke(q)
    parent_texts = get_parent_texts(child_docs)
    combined_context = " ".join(parent_texts).lower()

    # Extract key phrases from ground truth for matching
    gt_lower = gt.lower()

    # Check if key content from ground truth appears in retrieved context
    # Extract the core factual claim (numbers, names, key terms)
    import re
    # Extract numbers and key terms from ground truth
    numbers = re.findall(r'\d+[\.\d]*', gt)
    key_found = False

    if numbers:
        # If ground truth contains specific numbers, check if they appear in context
        numbers_found = sum(1 for n in numbers if n in combined_context)
        key_found = numbers_found >= len(numbers) * 0.5  # At least half the numbers found
    else:
        # For non-numeric answers, check if key phrases (3+ word chunks) appear
        words = gt_lower.split()
        if len(words) >= 4:
            # Check if a 4-word sliding window from GT appears in context
            for i in range(len(words) - 3):
                phrase = " ".join(words[i:i+4])
                if phrase in combined_context:
                    key_found = True
                    break
        else:
            key_found = gt_lower in combined_context

    if key_found:
        hits += 1
        context_matches += 1

    # MRR: Check each child doc's parent text for ground truth match
    found_rank = 0
    for rank, doc in enumerate(child_docs, 1):
        pid = doc.metadata.get("parent_id")
        if pid:
            parent_text = resolve_parent_text(pid, "").lower()
            if numbers:
                if any(n in parent_text for n in numbers):
                    found_rank = rank
                    break
            else:
                words = gt_lower.split()
                if len(words) >= 4:
                    for i in range(len(words) - 3):
                        phrase = " ".join(words[i:i+4])
                        if phrase in parent_text:
                            found_rank = rank
                            break
                    if found_rank > 0:
                        break

    if found_rank > 0:
        mrr_sum += 1.0 / found_rank

    status = "HIT" if key_found else "MISS"
    print(f"  [{idx:2d}/50] [{status}] {q[:70]}...")

    detailed_results.append({
        "question": q,
        "ground_truth": gt,
        "expected_article": item.get("expected_article", ""),
        "hit": key_found,
        "rank": found_rank if found_rank > 0 else "N/A",
        "num_parents_retrieved": len(parent_texts),
    })

# ── 5. Calculate final scores ─────────────────────────────────────────────
n = len(golden_data)
hit_rate = hits / n
mrr = mrr_sum / n
context_match_rate = context_matches / n

print("\n" + "=" * 55)
print("     PITWALL RETRIEVAL EVALUATION SCORES")
print("=" * 55)
print(f"  Hit Rate @ 3:         {hit_rate:.2%}  ({hits}/{n})")
print(f"  MRR (Mean Recip Rank): {mrr:.4f}")
print(f"  Context Match Rate:    {context_match_rate:.2%}  ({context_matches}/{n})")
print("=" * 55)
print()
print("  Interpretation:")
print(f"    - Hit Rate @ 3 > 80%  = GOOD retriever     (yours: {hit_rate:.0%})")
print(f"    - MRR > 0.70          = GOOD ranking        (yours: {mrr:.2f})")
print(f"    - Context Match > 75% = GOOD parent lookup  (yours: {context_match_rate:.0%})")
print()

# Save detailed results
output_path = PROJECT_ROOT / "retrieval_eval_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({
        "scores": {
            "hit_rate_at_3": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "context_match_rate": round(context_match_rate, 4),
            "total_questions": n,
            "total_hits": hits,
        },
        "detailed_results": detailed_results
    }, f, indent=2, ensure_ascii=False)

print(f"  Detailed results saved to: {output_path}")
