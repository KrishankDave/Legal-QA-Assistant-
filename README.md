*Legal QA Assistant*


A question answering system for the Indian Penal Code, built with retrieval-augmented generation.

What this does?



You ask it a legal question. It finds the relevant IPC sections, reads them, and gives you an answer with the exact section number cited. It does not guess. If the answer is not in the IPC, it says so.
That is the core idea — the AI only speaks from what it actually retrieved, not from memory.

Why did I built this?


Ask ChatGPT about a specific IPC section and it will often give you a confident answer that sounds right but is not. It mixes up section numbers, misquotes punishments, and has no way to tell you where it got the information from.
We tested this. When asked about Section 149, ChatGPT said it defines the minimum number of persons for an unlawful assembly as 5. That definition is actually in Section 141. Section 149 says something entirely different. Nobody reading the answer would know it was wrong.
This project fixes that by making the AI retrieve the actual text before answering, and by forcing it to cite where every claim comes from.

How it works?


When you ask a question, the system does five things in sequence:



1) It searches the IPC knowledge base using two methods at the same time — one that understands meaning, one that matches keywords. This handles both casual language and formal legal terminology.
2) It combines the results from both searches into one ranked list.
3) A reranker goes through the top results and picks the five most relevant sections.
4) Those five sections are handed to LLaMA 3 with strict instructions: answer only from this, cite section numbers, and say nothing if the answer is not here.
5) You get a grounded answer with source citations.

The Dataset -


The knowledge base is the complete Indian Penal Code — 575 sections across 23 chapters, stored as structured JSON. 
We also built a set of 100 question-answer pairs from scratch using LLaMA 3, which we use to evaluate and benchmark the system. 
This is the first publicly available QA dataset for the Indian Penal Code.

Evaluation -


We compared four systems on the same 100 questions using RAGAS — an evaluation framework that measures hallucination, relevance, retrieval completeness, and retrieval precision.
The four systems were a plain LLM with no retrieval, keyword search only, basic semantic search, and our full pipeline. 
The results show a clear improvement at each step, with the full pipeline scoring highest on faithfulness — meaning it makes the fewest unsupported claims.

Tech used


Python, LangChain, ChromaDB, rank-bm25, sentence-transformers, Ollama with LLaMA 3, RAGAS, Streamlit (will add i soon).

Disclaimer


This is a research project. It is not legal advice. For any actual legal matter, speak to a qualified lawyer.

