# generate_testset.py
# This file generates 100 question-answer pairs from your IPC JSON.
# These become your evaluation dataset — the IPC-QA benchmark.
#
# For each IPC section we pick, we ask LLaMA 3 to generate:
#   - A realistic question a citizen might ask
#   - The correct answer citing that section
#   - The ground truth section number
#
# Output: ipc_testset.json — 100 QA pairs ready for RAGAS evaluation

import json
import os
import random
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()

IPC_JSON_PATH = os.getenv("IPC_JSON_PATH")
base_path     = os.path.dirname(os.path.abspath(__file__))

# ── Load IPC data ──────────────────────────────────────────────────────────────
with open(os.path.join(base_path, IPC_JSON_PATH), "r", encoding="utf-8") as f:
    ipc_data = json.load(f)

# ── Filter out empty or repealed sections ─────────────────────────────────────
valid_sections = [
    d for d in ipc_data
    if d.get("section_desc", "").strip()
    and "repealed" not in d.get("section_desc", "").lower()
    and len(d.get("section_desc", "").strip()) > 50
]

print(f"Valid sections available: {len(valid_sections)}")

# ── Pick 100 random sections ───────────────────────────────────────────────────
# Set seed for reproducibility — same 100 sections every time you run this
random.seed(42)
selected = random.sample(valid_sections, 100)

print(f"Selected {len(selected)} sections for test set generation.\n")

# ── Load LLaMA 3 ──────────────────────────────────────────────────────────────
llm = OllamaLLM(model="llama3")

# ── Prompt to generate one QA pair ────────────────────────────────────────────
QA_GENERATION_PROMPT = """You are a legal expert on the Indian Penal Code (IPC).

Given the IPC section below, generate ONE realistic question that a common Indian citizen might ask, along with the correct answer based strictly on this section.

IPC Section {section_no} — {section_title}:
{section_desc}

RULES:
1. The question must sound natural — like something a person would actually ask.
2. Do NOT copy the section text directly into the question.
3. The answer must cite Section {section_no} explicitly.
4. Keep the answer under 3 sentences.
5. Respond in this exact JSON format only — no extra text:

{{
  "question": "your question here",
  "answer": "your answer here citing Section {section_no} IPC"
}}"""


# ── Generate QA pairs ──────────────────────────────────────────────────────────
testset = []
failed  = []

for i, section in enumerate(selected, 1):
    sec_no    = section["Section"]
    sec_title = section["section_title"]
    sec_desc  = section["section_desc"]

    print(f"[{i}/100] Generating QA for Section {sec_no} — {sec_title[:50]}...")

    prompt = QA_GENERATION_PROMPT.format(
        section_no  = sec_no,
        section_title = sec_title,
        section_desc  = sec_desc
    )

    try:
        qa = None
        last_error = None

        for attempt in range(3):  # retry up to 3 times
            response = llm.invoke(prompt)
            cleaned = response.strip()

            # Remove markdown code fences like ```json ... ``` or ``` ... ```
            if "```" in cleaned:
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()

            # Find the JSON object even if there's extra text around it
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                cleaned = cleaned[start:end]

            if not cleaned:
                last_error = "Empty response from LLM"
                time.sleep(1)
                continue

            try:
                qa = json.loads(cleaned)
                break  # success — stop retrying
            except Exception as e:
                last_error = e
                time.sleep(1)

        if qa is None:
            raise Exception(last_error)

        testset.append({
            "question"     : qa["question"],
            "answer"       : qa["answer"],
            "ground_truth" : qa["answer"],
            "section_no"   : str(sec_no),
            "section_title": sec_title,
            "context"      : f"Section {sec_no}: {sec_title}\n\n{sec_desc}"
        })

    except Exception as e:
        print(f"  Failed for Section {sec_no}: {e}")
        failed.append(sec_no)

    # Small pause to avoid overwhelming Ollama
    time.sleep(0.5)

# ── Save testset ───────────────────────────────────────────────────────────────
output_path = os.path.join(base_path, "ipc_testset.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(testset, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"Testset generation complete!")
print(f"Successfully generated : {len(testset)} QA pairs")
print(f"Failed                 : {len(failed)} sections {failed if failed else ''}")
print(f"Saved to               : ipc_testset.json")
print(f"{'='*50}")
