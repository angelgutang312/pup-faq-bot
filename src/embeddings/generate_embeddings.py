"""
Checkpoint 1 — Embedding Generation Demo
PUP Student FAQ RAG Assistant

Converts the cleaned corpus (data/processed/cleaned_corpus.jsonl) into vector
embeddings and runs a small sanity-check similarity query, exactly the kind
of check we'll rely on once we stand up a real vector database in
Checkpoint 2.

Primary model:  sentence-transformers/all-MiniLM-L6-v2  (Hugging Face)
Fallback model: scikit-learn TF-IDF + L2 normalization (offline, numpy-only)

Why all-MiniLM-L6-v2 (see docs/../embeddings_report.md for the full writeup):
  - 384-dimensional sentence embeddings -> small footprint, fast similarity
    search, easy to index in Chroma/FAISS without a GPU.
  - Trained specifically for semantic similarity / retrieval (contrastive
    training on 1B+ sentence pairs), which is exactly what RAG retrieval
    needs -- unlike a general-purpose LLM embedding this is optimized for
    "are these two chunks about the same thing," not language generation.
  - Free, open-source, and runs locally -> no per-call API cost, which
    matters for a student project that will be re-embedding/re-indexing a
    growing document set across every remaining checkpoint.
  - Widely supported by both LangChain and LlamaIndex, which are the
    orchestration frameworks required starting Checkpoint 3.

This sandbox has no outbound network access, so the Hugging Face model
weights cannot be downloaded here. The script tries the real model first;
if that's unavailable it automatically falls back to a TF-IDF vectorizer
(scikit-learn, fully offline) so the pipeline still runs end-to-end and the
similarity-search logic can be demonstrated. Swapping back to
all-MiniLM-L6-v2 for Checkpoint 2 is a one-line change once the model can
be downloaded (or run in an environment with internet access).

Usage:
    python src/embeddings/generate_embeddings.py
"""

import json
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CORPUS_PATH = PROCESSED_DIR / "cleaned_corpus.jsonl"
EMBEDDINGS_OUT = PROCESSED_DIR / "embeddings.npz"

MODEL_NAME_PRIMARY = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus():
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"{CORPUS_PATH} not found. Run src/preprocessing/clean_text.py first."
        )
    records = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def try_sentence_transformers(texts):
    """Attempt to embed with the real production model. Returns None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, None

    try:
        model = SentenceTransformer(MODEL_NAME_PRIMARY)
        vectors = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors), MODEL_NAME_PRIMARY
    except Exception as e:
        print(f"[warn] Could not load '{MODEL_NAME_PRIMARY}' ({e}). Falling back.")
        return None, None


def fallback_tfidf(texts):
    """Offline fallback: TF-IDF + L2 normalization, using scikit-learn (no internet needed)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=2048,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)
    vectors = normalize(matrix, norm="l2").toarray()
    return vectors, "tfidf-fallback (scikit-learn, offline)", vectorizer


def cosine_top_k(query_vec, matrix, k=3):
    sims = matrix @ query_vec  # vectors already L2-normalized
    top_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top_idx]


def embed_query_tfidf(vectorizer, query_text):
    from sklearn.preprocessing import normalize
    vec = vectorizer.transform([query_text])
    return normalize(vec, norm="l2").toarray()[0]


def main():
    records = load_corpus()
    texts = [r["cleaned_text"] for r in records]
    doc_ids = [r["doc_id"] for r in records]

    print(f"Loaded {len(texts)} cleaned documents.")

    vectors, model_used = try_sentence_transformers(texts)
    vectorizer = None
    if vectors is None:
        print(f"[info] '{MODEL_NAME_PRIMARY}' unavailable in this sandbox (no internet). "
              f"Using offline fallback.")
        vectors, model_used, vectorizer = fallback_tfidf(texts)

    print(f"Model used for this run: {model_used}")
    print(f"Embedding matrix shape: {vectors.shape}")

    # Save embeddings + doc ids for later checkpoints (vector DB indexing, etc.)
    np.savez(EMBEDDINGS_OUT, vectors=vectors, doc_ids=np.array(doc_ids))
    print(f"Saved embeddings to: {EMBEDDINGS_OUT}")

    # --- Sanity-check similarity query ---
    sample_query = "What happens if I get an Incomplete grade and don't finish the requirement on time?"
    print(f"\nSample query: {sample_query!r}")

    if vectorizer is not None:
        query_vec = embed_query_tfidf(vectorizer, sample_query)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_used)
        query_vec = model.encode([sample_query], normalize_embeddings=True)[0]

    top_matches = cosine_top_k(query_vec, vectors, k=3)
    print("Top-3 most similar documents (cosine similarity):")
    for rank, (idx, score) in enumerate(top_matches, start=1):
        print(f"  {rank}. {doc_ids[idx]}  (score={score:.4f})")


if __name__ == "__main__":
    main()
