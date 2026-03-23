# evaluate.py
# RAGAS evaluation using LLaMA 3 locally — sequential mode.
# Runs one question at a time to avoid Ollama timeout errors.

import json
import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from rank_bm25 import BM25Okapi
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from reranker import rerank

load_dotenv()

PERSIST_DIRECTORY_PATH = os.getenv("PERSIST_DIRECTORY_PATH")
IPC_COLLECTION_NAME    = os.getenv("IPC_COLLECTION_NAME")
IPC_JSON_PATH          = os.getenv("IPC_JSON_PATH")
base_path              = os.path.dirname(os.path.abspath(__file__))

# ── Configure RAGAS with LLaMA 3 locally ──────────────────────────────────────
print("Configuring RAGAS with LLaMA 3 locally...")

ragas_llm = LangchainLLMWrapper(
    ChatOllama(model="llama3", temperature=0, timeout=180)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="llama3")
)

# Force sequential — 1 worker, no parallelism
run_cfg = RunConfig(
    timeout     = 180,
    max_retries = 5,
    max_wait    = 120,
    max_workers = 1      # ← this is the key fix
)

for metric in [faithfulness, answer_relevancy, context_recall, context_precision]:
    metric.llm        = ragas_llm
    metric.embeddings = ragas_embeddings

print("RAGAS configured. Sequential mode — no timeouts.\n")

# ── Load test set ──────────────────────────────────────────────────────────────
print("Loading IPC-QA test set...")
with open(os.path.join(base_path, "ipc_testset.json"), encoding="utf-8") as f:
    testset = json.load(f)
print(f"Loaded {len(testset)} QA pairs.\n")

# ── Load IPC data ──────────────────────────────────────────────────────────────
with open(os.path.join(base_path, IPC_JSON_PATH), encoding="utf-8") as f:
    ipc_data = json.load(f)

ipc_lookup       = {str(d["Section"]): d for d in ipc_data}
corpus           = [
    f"Section {d['Section']} {d['section_title']} {d['section_desc']}"
    for d in ipc_data
]
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25             = BM25Okapi(tokenized_corpus)

# ── Load ChromaDB ─────────────────────────────────────────────────────────────
db = Chroma(
    collection_name   = IPC_COLLECTION_NAME,
    persist_directory = os.path.join(base_path, PERSIST_DIRECTORY_PATH),
    embedding_function= HuggingFaceEmbeddings()
)

# ── Load LLaMA 3 for answer generation ────────────────────────────────────────
llm = OllamaLLM(model="llama3")

# ── Prompts ────────────────────────────────────────────────────────────────────
LEGAL_PROMPT = """You are a legal assistant for the Indian Penal Code (IPC).
Answer ONLY using the IPC sections provided. Cite section numbers.
If insufficient information, say so. Never guess.

IPC SECTIONS:
{context}

QUESTION: {question}

ANSWER:"""

PLAIN_PROMPT = """You are a legal assistant specialising in the Indian Penal Code (IPC).
Answer the following legal question accurately. Cite the IPC section if known.

QUESTION: {question}

ANSWER:"""

def generate_answer(question, context_text):
    prompt = LEGAL_PROMPT.format(context=context_text, question=question)
    return llm.invoke(prompt)

# ══════════════════════════════════════════════════════════════════════════════
# 4 SYSTEMS
# ══════════════════════════════════════════════════════════════════════════════
def run_plain_llm(question):
    answer = llm.invoke(PLAIN_PROMPT.format(question=question))
    return answer, ["No retrieval — LLM training data only."]

def run_bm25_llm(question, k=5):
    tokens   = question.lower().split()
    scores   = bm25.get_scores(tokens)
    top_idx  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    docs     = [ipc_data[i] for i in top_idx]
    contexts = [f"Section {d['Section']}: {d['section_title']}\n\n{d['section_desc']}" for d in docs]
    answer   = generate_answer(question, "\n\n".join(contexts))
    return answer, contexts

def run_naive_rag(question, k=5):
    docs     = db.similarity_search(question, k=k)
    contexts = [doc.page_content for doc in docs]
    answer   = generate_answer(question, "\n\n".join(contexts))
    return answer, contexts

def run_advanced_rag(question, k=5):
    top_sections = rerank(question, top_k=k)
    contexts     = [sec["content"] for sec in top_sections]
    answer       = generate_answer(question, "\n\n".join(contexts))
    return answer, contexts

# ══════════════════════════════════════════════════════════════════════════════
# RAGAS RUNNER — sequential, one question at a time
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation(system_name, system_fn, testset):
    print(f"\n{'='*55}")
    print(f"  Evaluating: {system_name}")
    print(f"{'='*55}")

    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    for i, item in enumerate(testset, 1):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i}/{len(testset)}] {q[:65]}...")

        try:
            answer, contexts = system_fn(q)
            questions.append(q)
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(gt)
        except Exception as e:
            print(f"  Error on Q{i}: {e}")
            questions.append(q)
            answers.append("Error.")
            contexts_list.append(["Error"])
            ground_truths.append(gt)

        time.sleep(1)  # small pause between questions

    # Build dataset and run RAGAS
    eval_dataset = Dataset.from_dict({
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts_list,
        "ground_truth": ground_truths,
    })

    print(f"\n  Running RAGAS scoring (sequential)...")
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        raise_exceptions = False,
        run_config       = run_cfg,
    )

    scores = {
        "system"           : system_name,
        "faithfulness"     : round(float(result["faithfulness"]), 4),
        "answer_relevancy" : round(float(result["answer_relevancy"]), 4),
        "context_recall"   : round(float(result["context_recall"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }

    print(f"\n  Results for {system_name}:")
    for k, v in scores.items():
        if k != "system":
            print(f"    {k:<22}: {v}")

    return scores

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 55)
    print("   RAGAS Evaluation — IPC Legal QA Assistant")
    print("   LLaMA 3 local — sequential — no API key")
    print("=" * 55)
    print(f"Evaluating {len(testset)} questions across 4 systems.")
    print("Expected time: 60-120 minutes. Do not close terminal.\n")

    all_results = []

    all_results.append(run_evaluation(
        "System 1 — Plain LLM (no retrieval)", run_plain_llm, testset))

    all_results.append(run_evaluation(
        "System 2 — BM25 + LLM", run_bm25_llm, testset))

    all_results.append(run_evaluation(
        "System 3 — Naive RAG (ChromaDB only)", run_naive_rag, testset))

    all_results.append(run_evaluation(
        "System 4 — Advanced RAG (hybrid + reranker)", run_advanced_rag, testset))

    # Save
    with open(os.path.join(base_path, "results_table.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print final table
    print("\n" + "=" * 75)
    print("  FINAL RESULTS TABLE")
    print("=" * 75)
    print(f"{'System':<45} {'Faith':>7} {'Relev':>7} {'Recall':>7} {'Prec':>7}")
    print("-" * 75)
    for r in all_results:
        print(
            f"{r['system']:<45} "
            f"{r['faithfulness']:>7.4f} "
            f"{r['answer_relevancy']:>7.4f} "
            f"{r['context_recall']:>7.4f} "
            f"{r['context_precision']:>7.4f}"
        )
    print("=" * 75)
    print("\nSaved to: results_table.json")
    print("This is your research paper results table!")
