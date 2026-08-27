# Checkpoint 2 — Demo Backup Report

**Purpose:** This document is a backup in case the live in-class/recorded
demo of `src/llm/gemini_chat_demo.py` cannot reach the Gemini API (network
issues, quota, key problems, etc.). It records, for each test query: the
question, the exact chunks retrieved by our vector store, the exact prompt
sent to `gemini-3.5-flash-lite`, and the **expected grounded answer** —
manually verified against the source documents and the rules in Prompt 1
and Prompt 2 (`docs/Prompt_Engineering_Document.md`) — so a grader or
teammate can confirm the pipeline's correctness even without a live call.

Retrieval results below are taken directly from an actual run of
`src/vectordb/setup_chroma.py` / `src/llm/gemini_chat_demo.py` in this
environment (see `docs/api_demo_run_log.json` for the raw, timestamped
log including full prompt text). The **answer text** for each query below
is the expected LLM output we verified by hand against the retrieved
source document, since this sandbox cannot make live Gemini API calls
(no outbound internet access / no `GEMINI_API_KEY` configured here).

---

## Test 1 — Grounded, in-scope query

**Question:** "What is the passing grade for undergraduate students?"

**Retrieved (top-3):** `05_grading_system`, `07_correction_of_grades`, `02_enrollment_procedure`

**Expected answer (Prompt 1 applied):**
> PUP uses a numeric grading scale from 1.00 (highest) to 5.00 (fail). For
> undergraduate students, the lowest passing grade is 3.00 — anything
> from 3.25 to 5.00 is considered failing. [05_grading_system]

**Why this is correct:** Directly and fully supported by
`05_grading_system.md`; no fabricated detail; correct citation.

---

## Test 2 — Multi-document synthesis

**Question:** "How long do I have to complete an Incomplete grade requirement?"

**Retrieved (top-3):** `05_grading_system`, `08_incomplete_grades`, `07_correction_of_grades`

**Expected answer (Prompt 1 applied):**
> You have **one year** from the end of the term in which you received the
> Incomplete (INC) mark to complete the missing requirement. To do this,
> submit a Completion Form along with supporting documents (such as your
> instructor's class record and your registration certificate) through
> your college and the Office of the University Registrar.
> [08_incomplete_grades] If the deadline passes without completion, the
> INC automatically converts to a failing grade. [05_grading_system]

**Why this is correct:** Synthesizes two source documents accurately,
cites both, and does not overstate certainty beyond what's written.

---

## Test 3 — Hallucination-prevention (information NOT in knowledge base)

**Question:** "What is the tuition fee per unit for the BSCS program?"

**Retrieved (top-3):** `20_tuition_and_fees`, `10_dean_academic_standing`, `22_shifting_and_transfer`

**Expected answer (Prompt 1, Rule 3 applied):**
> I don't have enough information in my knowledge base to answer that
> accurately — my source documents describe PUP's subsidized tuition
> scheme in general terms but don't list an exact per-unit peso rate for
> BSCS specifically. Please check with your college's accounting/cashier
> office or the Office of the University Registrar for the current exact
> figure. [20_tuition_and_fees]

**Why this is correct:** This is the critical hallucination test. Our
knowledge base (`20_tuition_and_fees.md`) intentionally does not contain
an exact per-unit peso figure. A well-grounded model must refuse to
invent one rather than guessing a plausible-sounding number — this is
exactly what Prompt 1's Rule 3 is designed to enforce, and is the
behavior we will verify most closely during grading of the live demo.

---

## Test 4 — Off-topic refusal

**Question:** "Can you help me solve this calculus integral: the integral of x squared dx?"

**Retrieved (top-3):** `01_pupcet_admission`, `02_enrollment_procedure`, `03_academic_load` *(low-relevance — expected, since nothing in our PUP-policy corpus is about calculus)*

**Expected answer (Prompt 2 applied):**
> I'm ASK PUP, and I'm built specifically to answer questions about PUP
> student policies and procedures. That question is outside what I can
> help with here — is there anything about PUP enrollment, grades,
> scholarships, or student life I can help you with instead?

**Why this is correct:** Demonstrates Prompt 2's off-topic classifier
gate. Note the retrieval step still ran (as it does for every query,
since routing happens via the LLM's own classification of the question,
not a pre-filter) and — correctly — returned nothing relevant, which is
itself a secondary signal reinforcing the off-topic refusal.

---

## Summary Table

| # | Query type | Retrieved correctly? | Expected behavior | Prompt rule tested |
|---|---|---|---|---|
| 1 | Simple factual | Yes | Direct grounded answer | Prompt 1, Rule 1-2 |
| 2 | Multi-doc synthesis | Yes | Combines 2 sources, both cited | Prompt 1, Rule 1-2 |
| 3 | Info not in KB | N/A (correctly low-confidence) | Explicit refusal, no fabrication | Prompt 1, Rule 3 |
| 4 | Off-topic | N/A (correctly irrelevant) | Polite redirect, no attempt to answer | Prompt 2 |

## How to reproduce this live

```bash
export GEMINI_API_KEY=your_key_here
pip install google-genai chromadb
python src/llm/gemini_chat_demo.py
```

With a valid key and internet access, `call_gemini()` in
`gemini_chat_demo.py` will return `mode: "live"` and the actual model
output in place of the `[OFFLINE FALLBACK]` placeholder text — the rest
of the pipeline (retrieval, prompt construction, logging) is identical
and already verified working in this sandbox.
