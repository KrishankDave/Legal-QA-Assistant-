# answer.py
# This is the complete Legal QA pipeline — updated version.
# Now uses hybrid search + reranker instead of plain ChromaDB.
#
# Full pipeline:
#   User question
#       → Hybrid search (BM25 + ChromaDB + RRF)
#           → Reranker (picks best 5)
#               → LLaMA 3 generates cited answer

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from reranker import rerank

load_dotenv()

# ── Step 1: Load LLaMA 3 via Ollama ───────────────────────────────────────────
llm = OllamaLLM(model="llama3")

# ── Step 2: Legal prompt template ─────────────────────────────────────────────
# Strict instructions — forces the LLM to:
#   - Only use retrieved IPC sections
#   - Always cite section numbers
#   - Never make anything up
LEGAL_PROMPT = """You are a legal assistant specialising in the Indian Penal Code (IPC).

STRICT RULES:
1. Answer ONLY using the IPC sections provided below.
2. Always cite the exact Section number for every claim. Example: (Section 379 IPC)
3. If the provided sections do not contain enough information, respond with:
   "The provided IPC sections do not contain sufficient information to answer this question."
4. Never guess or use any knowledge outside the provided sections.
5. Keep your answer clear and easy to understand.

IPC SECTIONS RETRIEVED:
{context}

QUESTION: {question}

ANSWER (with IPC section citations):"""


# ── Step 3: Main function ──────────────────────────────────────────────────────

def answer_legal_question(user_question: str) -> dict:
    """
    Full pipeline:
      1. Hybrid search (BM25 + ChromaDB + RRF)
      2. Reranker picks best 5 sections
      3. LLaMA 3 generates a cited answer

    Returns a dict with:
      - answer   : the generated legal answer
      - sections : the top 5 sections used as context
    """

    # ── Hybrid search + rerank ─────────────────────────────────────────────
    print("Searching IPC knowledge base...")
    top_sections = rerank(user_question, top_k=5)

    if not top_sections:
        return {
            "answer"  : "No relevant IPC sections found for your question.",
            "sections": []
        }

    # ── Build context block from top 5 sections ────────────────────────────
    context_parts = []
    for sec in top_sections:
        context_parts.append(
            f"[Section {sec['section']} — {sec['section_title']}]\n{sec['content']}"
        )
    context = "\n\n".join(context_parts)

    # ── Fill prompt and send to LLaMA 3 ───────────────────────────────────
    final_prompt = LEGAL_PROMPT.format(
        context  = context,
        question = user_question
    )

    print("Generating answer with LLaMA 3...\n")
    answer = llm.invoke(final_prompt)

    return {
        "answer"  : answer,
        "sections": top_sections
    }


# ── Step 4: Run the assistant ──────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("     IPC Legal QA Assistant — Advanced RAG Pipeline")
    print("=" * 60)
    print("Hybrid Search + Reranking + LLaMA 3")
    print("Type 'exit' to quit.\n")

    while True:
        user_question = input("Your question: ").strip()

        if user_question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not user_question:
            print("Please type a question.\n")
            continue

        result = answer_legal_question(user_question)

        # Print the sources used
        print("\n" + "─" * 60)
        print("SOURCES USED:")
        print("─" * 60)
        for i, sec in enumerate(result["sections"], 1):
            print(f"  {i}. Section {sec['section']} — {sec['section_title']} "
                  f"(score: {sec['rerank_score']:.4f})")

        # Print the answer
        print("\n" + "─" * 60)
        print("ANSWER:")
        print("─" * 60)
        print(result["answer"])
        print("─" * 60 + "\n")
