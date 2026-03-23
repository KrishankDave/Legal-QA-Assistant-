# hybrid_retriever.py
# This file handles the smart search part of our project.
# It combines TWO search methods:
#   1. ChromaDB (dense search) - finds sections by MEANING
#   2. BM25 (sparse search)    - finds sections by KEYWORDS
# Then merges both results using RRF (Reciprocal Rank Fusion)
# giving us the best of both worlds.

import json
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

PERSIST_DIRECTORY_PATH = os.getenv("PERSIST_DIRECTORY_PATH")
IPC_COLLECTION_NAME    = os.getenv("IPC_COLLECTION_NAME")
IPC_JSON_PATH          = os.getenv("IPC_JSON_PATH")


# ── Step 1: Load IPC JSON for BM25 index ──────────────────────────────────────
# BM25 works directly on raw text so we load the JSON again here
print("Loading IPC data for BM25 index...")

base_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_path, IPC_JSON_PATH), "r", encoding="utf-8") as f:
    ipc_data = json.load(f)

# Build the text corpus — one string per section
# We combine title + description so BM25 can match on both
corpus = [
    f"Section {d['Section']} {d['section_title']} {d['section_desc']}"
    for d in ipc_data
]

# Tokenise the corpus (split each section into individual words)
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Build the BM25 index
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 index built on {len(corpus)} IPC sections.")


# ── Step 2: Connect to ChromaDB (dense search) ────────────────────────────────
print("Connecting to ChromaDB...")

db = Chroma(
    collection_name   = IPC_COLLECTION_NAME,
    persist_directory = os.path.join(base_path, PERSIST_DIRECTORY_PATH),
    embedding_function= HuggingFaceEmbeddings()
)

print("ChromaDB connected.\n")


# ── Step 3: RRF Fusion function ───────────────────────────────────────────────
# RRF = Reciprocal Rank Fusion
# It combines two ranked lists into one fair combined ranking.
# Formula: score = 1 / (rank + 60)
# Sections appearing near the top of BOTH lists get the highest score.

def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    """
    Combines dense and sparse result lists using RRF scoring.

    Args:
        dense_ids  : list of section numbers from ChromaDB search
        sparse_ids : list of section numbers from BM25 search
        k          : smoothing constant (default 60, standard in literature)

    Returns:
        List of section numbers sorted by combined RRF score (best first)
    """
    scores = {}

    # Score dense results (ChromaDB)
    for rank, sec_id in enumerate(dense_ids):
        scores[sec_id] = scores.get(sec_id, 0) + 1 / (rank + k)

    # Score sparse results (BM25)
    for rank, sec_id in enumerate(sparse_ids):
        scores[sec_id] = scores.get(sec_id, 0) + 1 / (rank + k)

    # Sort by combined score — highest first
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return sorted_ids


# ── Step 4: The main hybrid search function ───────────────────────────────────

def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Searches IPC sections using both ChromaDB and BM25,
    fuses results with RRF, and returns top_k sections.

    Args:
        query  : the user's legal question
        top_k  : how many final sections to return (default 5)

    Returns:
        List of dicts with section number, title, chapter, and content
    """

    # ── Dense search: ChromaDB finds sections by MEANING ──────────────────────
    # We retrieve top 20 candidates from ChromaDB first
    dense_docs = db.similarity_search(query, k=20)

    # Extract section numbers from results
    dense_ids = [
        str(doc.metadata.get("section"))
        for doc in dense_docs
    ]

    # ── Sparse search: BM25 finds sections by KEYWORDS ────────────────────────
    # Tokenise the query (split into words)
    tokenized_query = query.lower().split()

    # Get BM25 scores for all 575 sections
    bm25_scores = bm25.get_scores(tokenized_query)

    # Pick top 20 section indices by BM25 score
    top20_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:20]

    # Convert indices to section numbers
    sparse_ids = [str(ipc_data[i]["Section"]) for i in top20_bm25_indices]

    # ── RRF Fusion: combine both result lists ──────────────────────────────────
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)

    # ── Build final results: top_k sections ───────────────────────────────────
    # Create a lookup dictionary from section number to IPC data
    ipc_lookup = {str(d["Section"]): d for d in ipc_data}

    results = []
    for sec_id in fused_ids[:top_k]:
        if sec_id in ipc_lookup:
            d = ipc_lookup[sec_id]
            results.append({
                "section"      : d["Section"],
                "section_title": d["section_title"],
                "chapter"      : d["chapter"],
                "chapter_title": d["chapter_title"],
                "content"      : f"Section {d['Section']}: {d['section_title']}\n\n{d['section_desc']}"
            })

    return results


# ── Step 5: Test it directly ───────────────────────────────────────────────────
# Run this file directly to test hybrid search
# python hybrid_retriever.py

if __name__ == "__main__":
    test_query = "What is the punishment for theft?"
    print(f"Query: {test_query}\n")
    print("Running hybrid search (BM25 + ChromaDB + RRF)...\n")

    results = hybrid_search(test_query, top_k=5)

    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. Section {r['section']} — {r['section_title']}")
        print(f"   Chapter {r['chapter']}: {r['chapter_title']}")
        print()
