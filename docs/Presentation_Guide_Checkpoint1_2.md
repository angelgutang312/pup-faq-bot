# ASK PUP — Presentation Guide (Checkpoint 1 & 2)
### "Run this → Show this → Say this," explained super simply 🧸

This is your walkthrough script for presenting the project. Part 1 is the
short intro you say at the very start. Part 2 is the live run-through:
for every step, run the command, show the output on screen, then say the
line given (in plain, friendly words — imagine explaining it to a curious
5-year-old).

---

## Part 1 — The Intro (say this first, before touching the terminal)

> "Imagine PUP has a giant pile of question papers — how many units you
> can take, what grade you need to pass, how to get a scholarship. Right
> now, students have to dig through confusing handbooks and websites to
> find answers.
>
> We built a robot helper called **ASK PUP**. It already read 25 real PUP
> policy topics, so when a student asks it a question, it finds the exact
> right document and answers using *only* what's actually written there —
> it never makes things up.
>
> Today we'll show you two things: first, how we prepared the robot's
> 'books' (Checkpoint 1), and second, how the robot actually finds and
> answers questions (Checkpoint 2)."

---

## Part 2 — Live Run-Through

### Step 0 — Show the raw material

**Run this:**
```
ls data/raw
```

**Show this:** the folder of 25 `.md` files (e.g.
`05_grading_system.md`, `09_scholarships.md`). Open one file so people
see real, readable text.

**Say this:**
> "These are our 25 'books' — one topic per file. We rewrote each one
> ourselves in plain language instead of copy-pasting from scanned PDFs
> and old websites, so they're clean and consistent from the start."

---

### Step 1 — Clean the books (Checkpoint 1)

**Run this:**
```
python src/preprocessing/clean_text.py
```

**Show this:** the terminal output —
```
Processed 25 documents.
Cleaned corpus written to: data/processed/cleaned_corpus.jsonl
Before/after report written to: data/processed/preprocessing_report.md
```
Then open `data/processed/preprocessing_report.md` and point at one
before/after example.

**Say this:**
> "This step is like tidying a messy backpack. It strips out formatting
> junk — titles, symbols, extra spaces — and then cuts each document into
> small word pieces called *tokens*, like cutting a sentence into LEGO
> bricks. That's what lets a computer count and compare words later."

---

### Step 2 — Turn words into "number fingerprints"

**Run this:**
```
python src/embeddings/generate_embeddings.py
```

**Show this:** the terminal output —
```
Loaded 25 cleaned documents.
[info] 'sentence-transformers/all-MiniLM-L6-v2' unavailable in this sandbox (no internet). Falling back to offline TF-IDF vectorizer.
Model used for this run: TF-IDF (offline fallback)
Embedding matrix shape: (25, 2048)
Saved embeddings to: data/processed/embeddings.npz

Sample query: '...'
Top-3 most similar documents (cosine similarity):
  1. ...  (score=...)
```

**Say this:**
> "Computers don't understand words the way we do — they understand
> numbers. So every document gets turned into a 'fingerprint' made of
> numbers. Two documents about similar topics get fingerprints that sit
> close together, like two kids who like the same ice cream flavor
> standing near each other.
>
> We wanted to use a smart pre-trained fingerprint-maker
> [`all-MiniLM-L6-v2`], but our sandbox has no internet to download it —
> so the script automatically falls back to a fully offline tool
> (TF-IDF), like using a paper map when the GPS has no signal. The
> pipeline still runs end-to-end either way."

*(This is the end of Checkpoint 1 — 25 clean documents, each with a
number fingerprint.)*

---

### Step 3 — Compare the "rulers" for measuring closeness (Checkpoint 2)

**Run this:**
```
python src/vectordb/distance_metrics_report.py
```

**Show this:** open `docs/Distance_Metric_Comparison.md` and point at one
results table plus the **Recommendation** section at the bottom.

**Say this:**
> "To find the closest fingerprint to a question, we need a ruler. We
> tested three: Cosine Similarity (do two fingerprints point the *same
> direction*?), Dot Product, and Euclidean Distance (straight-line
> distance). All three mostly agreed on the winning document, so we
> picked **Cosine Similarity** as our official ruler — it doesn't get
> confused by documents of different lengths, and it's the ruler our
> vector database likes best by default."

---

### Step 4 — Build the magic shelf (ChromaDB)

**Run this:**
```
python src/vectordb/setup_chroma.py
```

**Show this:** the terminal output —
```
Backend in use: chromadb (persistent, on-disk)
Indexed 25 documents into collection 'pup_faq_docs'.

Query: 'How many units of overload can a graduating student take?'
  1. 01_pupcet_admission  (cosine distance=...)
  ...
```

**Say this:**
> "This is our 'toy bin.' Every document's fingerprint gets stored inside
> ChromaDB, our vector database. When a question comes in, we make a
> fingerprint for the *question* too, and ask the toy bin: 'which
> documents have the closest fingerprints?' It instantly hands back the
> top 3 matches — no need to read all 25 documents every time."

---

### Step 5 — Hire the smart reader (Gemini) and ask real questions

**Run this:**
```
python src/llm/gemini_chat_demo.py
```

**Show this:** the live terminal output — walk through all 4 questions
and their answers, pointing out:
- Q1 (`passing grade`) → a normal, cited, correct answer. **Mode: live**
  means this really came from Gemini, live, right now.
- Q3 (`tuition fee per unit`) → the robot **refuses to guess** a number
  that isn't in our documents.
- Q4 (`calculus integral`) → the robot **stays on topic** and redirects
  instead of solving a random math problem.

**Say this:**
> "Finding the right document isn't enough — students want a short,
> friendly answer, not a whole PDF. So we hand the top 3 documents to an
> AI model, Gemini, along with strict house rules we call *prompts*:
>
> 1. Only answer from the documents we give you — if it's not there, say
>    'I don't know, please ask the registrar,' instead of guessing.
> 2. Stay on topic — if someone asks a math question, politely decline.
> 3. Always cite which document the answer came from.
>
> Watch what happens when we ask about an exact tuition fee that isn't in
> our documents [point at Q3] — the robot admits it doesn't know instead
> of making up a number. That's the single most important behavior we
> tested, because a robot that confidently lies is worse than one that
> says 'I don't know.'"

---

### Step 6 (backup, only if the internet/API fails during the real demo)

**Show this:** `docs/Checkpoint2_Demo_Backup_Report.md`

**Say this:**
> "In case the live call doesn't go through during the actual demo, we
> already wrote down and hand-verified exactly what the robot *should*
> say for each of these same 4 questions — so you can still see the
> expected behavior even offline."

---

## Cheat-sheet — jargon → plain words (keep this open as a reference)

| Fancy word | Say it like a 5-year-old would |
|---|---|
| Corpus | Our pile of 25 little books |
| Tokenization | Cutting sentences into word LEGO bricks |
| Embedding | A number fingerprint that shows what a document is about |
| Vector database (ChromaDB) | A toy bin that stores fingerprints so we can find the closest one fast |
| Cosine similarity | Checking if two fingerprints "point" the same way |
| Retrieval | Asking the toy bin "which documents match this question?" |
| LLM (Gemini) | The smart reader who explains the found document nicely |
| Prompt | The house rules we give the smart reader before it talks |
| Hallucination | The robot making up an answer instead of admitting it doesn't know |
| RAG (Retrieval-Augmented Generation) | "Find the right document first, THEN explain it" — instead of guessing from memory |

---

## Anticipated Q&A (say these if asked)

**Q: Why not just let the AI answer from what it already knows?**
A: Because it might confidently make up a wrong PUP rule. We want it to
only speak from our real 25 documents, and admit when it doesn't know.

**Q: Why did you use a backup (TF-IDF) instead of the "real" fingerprint
maker?**
A: Our workspace had no internet to download the pre-trained model, so we
built a fully offline backup that does the same job, just a bit simpler —
like a paper map instead of GPS with no signal.

**Q: What if the live Gemini call fails during the actual presentation?**
A: We have a hand-verified backup report (`Checkpoint2_Demo_Backup_Report.md`)
showing exactly what the robot should say for the same questions — the
retrieval and prompt logic is identical either way.
