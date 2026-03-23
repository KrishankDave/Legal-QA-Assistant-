# reranker.py
# This file takes the hybrid search results and re-orders them.
# A cross-encoder reranker reads the QUESTION and each SECTION together
# and gives a proper relevance score — much more accurate than
# similarity search alone.
#
# Think of it like this:
#   Hybrid search = a fast librarian who picks 20 possibly relevant books
#   Reranker      = an expert who reads each book + your question carefully
#                   and picks the best 5

import os
from sentence_transformers import CrossEncoder
from hybrid_retriever import hybrid_search

# ── Step 1: Load the reranker model ───────────────────────────────────────────
# BGE-Reranker-v2-M3 is free, runs locally, no API needed
# It downloads automatically the first time (~570MB)
print("Loading reranker model (downloads once, ~570MB)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("Reranker loaded.\n")


# ── Step 2: The main rerank function ──────────────────────────────────────────

def rerank(query: str, top_k: int = 5) -> list[dict]:
    """
    1. Runs hybrid search to get top 20 candidate sections
    2. Scores each (query, section) pair with the cross-encoder
    3. Returns top_k sections sorted by reranker score

    Args:
        query  : the user's legal question
        top_k  : how many final sections to return (default 5)

    Returns:
        List of dicts — same format as hybrid_search results
        but reordered by true relevance
    """

    # Get 20 candidates from hybrid search
    print(f"Running hybrid search for: '{query}'")
    candidates = hybrid_search(query, top_k=20)

    if not candidates:
        return []

    # ── Score each candidate with the cross-encoder ────────────────────────
    # The cross-encoder reads BOTH the query and the section text TOGETHER
    # This is much more accurate than comparing them separately
    pairs = [
        [query, candidate["content"]]
        for candidate in candidates
    ]

    print(f"Reranking {len(pairs)} candidates...")
    scores = reranker.predict(pairs)

    # ── Attach scores and sort ─────────────────────────────────────────────
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    # Sort by reranker score — highest relevance first
    reranked = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Return only the top_k most relevant sections
    return reranked[:top_k]


# ── Step 3: Test it directly ───────────────────────────────────────────────────
# Run: python reranker.py

if __name__ == "__main__":

    test_query = "What is the punishment for theft?"
    print(f"Query: {test_query}\n")

    results = rerank(test_query, top_k=5)

    print(f"\nTop {len(results)} results after reranking:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. Section {r['section']} — {r['section_title']}")
        print(f"   Chapter: {r['chapter_title']}")
        print(f"   Rerank score: {r['rerank_score']:.4f}")
        print()
