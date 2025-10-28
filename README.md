# 📘 AI-Powered PDF Highlighter (LLM + Retrieval Pipeline)

This project builds an **end-to-end AI pipeline** that reads a PDF, finds the most relevant evidence for a user query using an LLM, and **auto-highlights** the matching sentences back inside the original PDF — preserving layout and math symbols.

The pipeline runs in **five modular cells**, each producing clean artifacts and designed for low-latency re-runs.

---

## 🧩 Pipeline Overview

### **Cell 1 — Ingest → Split → Index**
- Loads the PDF with **PyMuPDF**.
- Splits text into **1,500-token overlapping chunks (375 overlap)** using **spaCy** (instead of RecursiveTextSplitter for cleaner linguistic boundaries).
- Attaches **UUID, page, and source** metadata.
- Saves:
  - `sentence_metadata.json`
  - `uuid_lookup.json`
  - Vector index in `vector_index/`
- Builds **FAISS** index with your embedding model and exposes a retriever (`k=3`).

🧠 *Artifacts:* `vector_index/*`, `sentence_metadata.json`, `uuid_lookup.json`

---

### **Cell 2 — Retrieve → Reason → Answer**
- Runs a **query** through the retriever to get top relevant chunks.
- Performs a **two-step reasoning process**:
  1. Extracts **FACTS, GLOSSARY, ENTITIES, JUSTIFICATION**
  2. Formats a **textbook-style final answer**.
- Prints reasoning + answer + source chunks with their UUIDs/pages.

🧠 *Exports:*  
`reasoning_text`, `final_answer_text` (globals)

---

### **Cell 3 — Sentence Segmentation → Evidence Selection (LLM)**
- Uses **syntok** for sentence-level segmentation (faster and more verbatim-stable than spaCy).
- Runs a **repair/merge** pass to fix broken PDF text.
- Assigns sentence IDs (`s1…`), builds `sentence_lookup.json`.
- Performs **batched LLM inference (batch=20)** to select the **minimum set of supporting sentences**.
- Dedupes, restores document order, and saves:
  - `highlight_ids.json`

🧠 *Artifacts:* `sentence_lookup.json`, `highlight_ids.json`

---

### **Cell 4 — Coverage-Aware Filtering + Safe Cleaning**
- Keeps only `highlight_ids` in order.
- Adds optional **LLM framing/context extension**.
- Cleans sentences for exact PDF matching:
  - Removes “Figure…”, “Exercise…” tails.
  - Splits bullet lines.
  - Removes only leading markers.
- Writes:
  - `ai_cleaned_sentences.json` → (sid → [verbatim pieces])

🧠 *Artifacts:* `ai_cleaned_sentences.json`

---

### **Cell 5 — Robust PDF Highlighter**
- Opens the original PDF.
- For each sentence piece:
  1. Searches **block-scoped exact window**
  2. Falls back to **page-wide**
  3. Then **search_for()** with symbol/prime variants
  4. Lastly, a **tolerant token window** (~90% match)
- Draws **micro-highlights** (operators, parentheses, primes).
- Outputs:
  - `highlighted_output.pdf`
  - Console summary of total/missed spans.

🧠 *Artifacts:* `highlighted_output.pdf`

---

## 🧠 Why These Design Choices

| Step | Tool | Reason |
|------|------|--------|
| Chunking | **spaCy** | Linguistically aware splits; avoids mid-sentence breaks unlike RecursiveTextSplitter |
| Sentence segmentation | **syntok** | Lightweight, deterministic, and preserves verbatim text for PDF matching |
| Evidence selection | **Batch LLM inference** | Faster than per-sentence calls; context is shared via the query prefix |
| Cleaning | **Safe text normalization** | Non-destructive cleanup ensures PDF search works perfectly |
| Highlighting | **PyMuPDF + token matching** | Exact glyph-level highlighting with math-safe micro-glow |

---

## ⚙️ Features & Planned Upgrades

- ✅ Caching by document hash (skip redundant FAISS builds)
- ✅ Deterministic highlight order by document sequence
- 🔄 Adjacency-aware augmentation (capture constraints like `(0<i<90°)`)
- 🔄 Role tagging (`core`, `constraint`, `framing`, `symbol_def`)
- 🧮 Return `highlight_pages` and `total_spans` to UI

---

## 🧰 Tech Stack

- **Python 3.10+**
- **spaCy**, **syntok**, **FAISS**, **PyMuPDF**
- **LLM (OpenAI / Local)** for reasoning & selection
- **JSON-based state sharing** between cells

---
## 🗂️ File Mapping

| File | Purpose |
|------|----------|
| `backend.py` | Handles overall orchestration and API/backend integration |
| `file3.py` | 🔹 **Sentence Selection** — runs Cell 3 (syntok segmentation, repair, batched LLM evidence selection) |
| `file4.py` | **Coverage-Aware Filtering + Safe Cleaning** (Cell 4) |
| `file5.py` | **Robust PDF Highlighting** (Cell 5) |
| `cell2_output.txt` | Debug/log file capturing reasoning + answer output from Cell 2 |
| `ai_cleaned_sentences.json` | Cleaned and ready-to-highlight sentences |
| `highlight_ids.json` | Selected sentence IDs returned from Cell 3 |
| `highlighted_output.pdf` | Final AI-highlighted PDF result |

---

Each file corresponds directly to a pipeline cell, so you can run or debug them individually.  
For full reproducibility, start from `backend.py` or run each cell script sequentially.
