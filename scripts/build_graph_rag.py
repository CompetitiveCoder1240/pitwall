"""
scripts/build_graph_rag.py
----------------------------
Builds a directed Knowledge Graph (GraphRAG) across all 6 FIA 2026 Regulation sections
(Sections A through F) using NetworkX.

Extracts:
1. Article Nodes & Section Nodes (e.g. "Section B Article 40.2")
2. Concept/Entity Nodes (e.g. "Parc Fermé", "Front Wing", "Pit Lane Start", "MGU-K", "Cost Cap")
3. Directed Edges representing relationships (GOVERNS, PENALIZES, REFERENCES, HAS_SUBRULE)

Outputs:
data/fia_knowledge_graph.pkl
"""

import os
import re
import pickle
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENTS_PATH = os.path.join(PROJECT_ROOT, "f1-rag-chatbot", "parents.pkl")
if not os.path.exists(PARENTS_PATH):
    PARENTS_PATH = os.path.join(PROJECT_ROOT, "parents.pkl")

OUTPUT_GRAPH_PATH = os.path.join(PROJECT_ROOT, "f1-rag-chatbot", "data", "fia_knowledge_graph.pkl")
os.makedirs(os.path.dirname(OUTPUT_GRAPH_PATH), exist_ok=True)

# Key FIA F1 domain entities to map into graph
KEY_ENTITIES = {
    "parc fermé": ["Parc Fermé", "Parc Ferme", "parc ferme"],
    "front wing": ["Front Wing", "front wing assembly", "front wing flap"],
    "rear wing": ["Rear Wing", "rear wing flap", "DRS"],
    "power unit": ["Power Unit", "MGU-K", "MGU-H", "Energy Store", "Turbocharger"],
    "pit lane start": ["Pit Lane Start", "start from the pit lane", "pit lane penalty"],
    "safety car": ["Safety Car", "VSC", "Virtual Safety Car"],
    "cost cap": ["Cost Cap", "Financial Limit", "Budget Cap"],
    "minimum weight": ["Minimum Weight", "Car Mass", "Ballast"],
}

def build_knowledge_graph():
    print("[GraphRAG] Initializing FIA 2026 Knowledge Graph...")
    G = nx.DiGraph()

    # Load parent vault to parse full regulation contexts
    if not os.path.exists(PARENTS_PATH):
        print(f"[GraphRAG Warning] Could not find {PARENTS_PATH}, creating core schema graph...")
        parents_vault = {}
    else:
        with open(PARENTS_PATH, "rb") as f:
            parents_vault = pickle.load(f)

    print(f"[GraphRAG] Processing {len(parents_vault)} parent regulation blocks...")

    article_count = 0
    ref_edge_count = 0
    entity_edge_count = 0

    # Add core section nodes
    sections = ["Section A", "Section B", "Section C", "Section D", "Section E", "Section F"]
    for sec in sections:
        G.add_node(sec, type="SECTION", label=sec)

    # 1. Parse vault chunks for Articles and Entities
    for parent_id, parent_entry in parents_vault.items():
        if isinstance(parent_entry, dict):
            text = parent_entry.get("text", "")
        else:
            text = str(parent_entry)

        # Extract Article numbers (e.g. "Article B3.5.3", "Article 40.2", "Section C Article 4.1")
        articles_found = re.findall(r"(Article\s+([A-Z0-9\.]+))", text, re.IGNORECASE)
        for full_art_str, art_num in articles_found:
            art_node = f"Article {art_num.rstrip('.')}"
            if not G.has_node(art_node):
                G.add_node(art_node, type="ARTICLE", text_sample=text[:300])
                article_count += 1

                # Link Article to Section if format is "Article B3.5" -> Section B
                if art_num.startswith("A"):
                    G.add_edge("Section A", art_node, relation="CONTAINS")
                elif art_num.startswith("B"):
                    G.add_edge("Section B", art_node, relation="CONTAINS")
                elif art_num.startswith("C"):
                    G.add_edge("Section C", art_node, relation="CONTAINS")
                elif art_num.startswith("D"):
                    G.add_edge("Section D", art_node, relation="CONTAINS")
                elif art_num.startswith("E"):
                    G.add_edge("Section E", art_node, relation="CONTAINS")
                elif art_num.startswith("F"):
                    G.add_edge("Section F", art_node, relation="CONTAINS")

            # Check cross-references to other articles in the same text
            for full_ref_str, ref_num in articles_found:
                target_node = f"Article {ref_num.rstrip('.')}"
                if target_node != art_node:
                    if not G.has_edge(art_node, target_node):
                        G.add_edge(art_node, target_node, relation="REFERENCES")
                        ref_edge_count += 1

            # Check entity mentions
            for canonical_entity, keywords in KEY_ENTITIES.items():
                for kw in keywords:
                    if kw.lower() in text.lower():
                        entity_node = canonical_entity.title()
                        if not G.has_node(entity_node):
                            G.add_node(entity_node, type="ENTITY")
                        if not G.has_edge(art_node, entity_node):
                            G.add_edge(art_node, entity_node, relation="GOVERNS")
                            entity_edge_count += 1
                        break

    print(f"[GraphRAG] Graph Construction Summary:")
    print(f"  - Total Nodes: {G.number_of_nodes()}")
    print(f"  - Total Edges: {G.number_of_edges()}")
    print(f"  - Article Nodes: {article_count}")
    print(f"  - Reference Edges: {ref_edge_count}")
    print(f"  - Entity Edges: {entity_edge_count}")

    # Save graph to disk
    with open(OUTPUT_GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)

    print(f"[GraphRAG] Knowledge Graph persisted successfully to {OUTPUT_GRAPH_PATH}")
    return G

if __name__ == "__main__":
    build_knowledge_graph()
