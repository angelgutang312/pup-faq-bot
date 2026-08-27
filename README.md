# PUP Student FAQ Assistant — Gen AI Capstone Project

A Retrieval-Augmented Generation (RAG) chatbot that answers PUP student
questions on enrollment, grading, attendance, scholarships, student
organizations, and related policies — grounded in a curated knowledge base
so answers are accurate and citable, instead of hallucinated.

Built for **IS Professional Elective #4 (Generative AI Systems)**,
1st Sem, A.Y. 2026–2027. Checkpoints are aligned to the four Performance
Tasks in the course's Semester Project Guide.

## Status

- [x] **Checkpoint 1 — Foundation & Data Pipeline** (Prelim)
- [x] **Checkpoint 2 — Prompt Architecture & Vector Indexing** (Midterm)
- [ ] Checkpoint 3 — RAG Orchestration & Working Application (Semi-Final)
- [ ] Checkpoint 4 — Customization, Deployment & Final Defense (Final)

## Project structure

```
pup-faq-ragbot/
├── README.md
├── requirements.txt
├── docs/
│   ├── Project_Proposal.docx
│   ├── Reflection_Checkpoint1.md
│   ├── Prompt_Engineering_Document.md      # Checkpoint 2 deliverable #6
│   ├── Distance_Metric_Comparison.md       # Checkpoint 2 deliverable #9
│   ├── Checkpoint2_Demo_Backup_Report.md   # backup record of demo Q&A
│   └── api_demo_run_log.json               # raw run log (prompts + retrieval)
├── data/
│   ├── raw/                     # 25 source FAQ/policy documents (Markdown)
│   └── processed/
│       ├── cleaned_corpus.jsonl
│       ├── preprocessing_report.md
│       ├── embeddings.npz
│       ├── embeddings_report.md
│       └── chroma_db/                      # vector DB persisted here
├── src/
│   ├── preprocessing/clean_text.py
│   ├── embeddings/generate_embeddings.py
│   ├── vectordb/
│   │   ├── setup_chroma.py                 # Checkpoint 2 deliverable #8
│   │   └── distance_metrics_report.py      # Checkpoint 2 deliverable #9
│   └── llm/
│       └── gemini_chat_demo.py             # Checkpoint 2 deliverable #7
└── notebooks/                   # reserved for Checkpoints 3-4
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Checkpoint 1 pipeline

```bash
# 1. Clean and tokenize the raw corpus -> data/processed/cleaned_corpus.jsonl
python src/preprocessing/clean_text.py

# 2. Generate embeddings + run a sanity-check similarity query
python src/embeddings/generate_embeddings.py
```

**Note:** `generate_embeddings.py` tries to load
`sentence-transformers/all-MiniLM-L6-v2` first. If you're running this in
an environment without internet access, it automatically falls back to an
offline TF-IDF vectorizer so the pipeline still runs end-to-end. See
`data/processed/embeddings_report.md` for the full model justification and
sandbox-limitation note.

## Running the Checkpoint 2 pipeline

```bash
# 3. Load embeddings into ChromaDB and run a similarity-search demo
python src/vectordb/setup_chroma.py

# 4. Generate the Cosine vs. Euclidean vs. Dot Product comparison report
python src/vectordb/distance_metrics_report.py

# 5. Run the grounded Q&A demo against Gemini 3.5 Flash-Lite
export GEMINI_API_KEY=your_key_here   # optional — falls back gracefully without it
python src/llm/gemini_chat_demo.py
```

**Note:** `setup_chroma.py` tries real ChromaDB first, falling back to an
offline NumPy-based `SimpleVectorStore` with the same `.add()`/`.query()`
interface if ChromaDB isn't installed. `gemini_chat_demo.py` tries a live
call to `gemini-3.5-flash-lite` via the `google-genai` SDK, falling back
to an extractive (no-LLM) answer if no API key/internet is available. See
`docs/Checkpoint2_Demo_Backup_Report.md` for the manually-verified
expected model outputs, kept as a backup in case the live demo can't
reach the API.

## Dataset

25 topic-scoped documents covering PUP student policy areas (admissions,
enrollment, grading, attendance, scholarships, student council, OJT,
graduation requirements, etc.), consolidated and rewritten in our own
words from the official PUP Student Handbook, freshman enrollment FAQs,
and other public university sources. See `data/raw/`.

## Team

| Team Member | Primary Responsibility | Assigned Component |
|---|---|---|
| TBD | Lead Developer | Data Pipeline & Embeddings |
| TBD | AI Orchestrator | RAG Architecture & Prompting |
| TBD | DevOps/UI | Deployment & Interface |

## Architecture (target, by Final)

```
Raw docs -> Preprocessing -> Embeddings -> Vector DB (Chroma/FAISS)
   -> RAG orchestration (LangChain/LlamaIndex) -> Chat interface
   -> Containerized (Docker) deployment
```
