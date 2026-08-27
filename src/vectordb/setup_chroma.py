"""
Checkpoint 2 — Vector Database Setup
PUP Student FAQ RAG Assistant

Loads the Checkpoint 1 embeddings into a vector database (ChromaDB) and
runs a working similarity-search query.

Primary backend: ChromaDB (persistent client, local on-disk store)
Fallback backend: SimpleVectorStore — a small NumPy-only class that
    mirrors Chroma's .add() / .query() interface, used only because this
    sandbox has no internet access to `pip install chromadb`. It persists
    to the same data/processed/chroma_db/ directory so swapping back to
    real Chroma later requires no changes to calling code -- just remove
    the fallback import.

Usage:
    python src/vectordb/setup_chroma.py
"""

import json
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npz"
CORPUS_PATH = PROCESSED_DIR / "cleaned_corpus.jsonl"
CHROMA_DIR = PROCESSED_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "pup_faq_docs"


# --------------------------------------------------------------------------
# Fallback: minimal Chroma-compatible in-memory / on-disk vector store
# --------------------------------------------------------------------------
class SimpleVectorStore:
    """
    Drop-in fallback for chromadb.Collection, offline-only, NumPy-based.
    Supports the subset of the Chroma API this project needs:
        .add(ids, embeddings, documents, metadatas)
        .query(query_embeddings, n_results) -> dict shaped like Chroma's output
    Persists vectors/docs/ids to a .npz + .json pair on disk so re-running
    the script doesn't require re-embedding.
    """

    def __init__(self, name: str, persist_dir: Path):
        self.name = name
        self.persist_dir = persist_dir
        self._vectors = None
        self._ids = []
        self._documents = []
        self._metadatas = []

    def add(self, ids, embeddings, documents, metadatas):
        self._ids = list(ids)
        self._vectors = np.asarray(embeddings)
        self._documents = list(documents)
        self._metadatas = list(metadatas)
        # persist
        np.savez(
            self.persist_dir / f"{self.name}.npz",
            vectors=self._vectors,
            ids=np.array(self._ids),
        )
        with (self.persist_dir / f"{self.name}_meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"documents": self._documents, "metadatas": self._metadatas},
                f,
                ensure_ascii=False,
            )

    def load(self):
        npz = np.load(self.persist_dir / f"{self.name}.npz", allow_pickle=True)
        self._vectors = npz["vectors"]
        self._ids = list(npz["ids"])
        with (self.persist_dir / f"{self.name}_meta.json").open(encoding="utf-8") as f:
            meta = json.load(f)
        self._documents = meta["documents"]
        self._metadatas = meta["metadatas"]

    def query(self, query_embeddings, n_results=3):
        query_vec = np.asarray(query_embeddings[0])
        # cosine similarity (vectors are already L2-normalized from Checkpoint 1)
        sims = self._vectors @ query_vec
        top_idx = np.argsort(-sims)[:n_results]
        return {
            "ids": [[self._ids[i] for i in top_idx]],
            "documents": [[self._documents[i] for i in top_idx]],
            "metadatas": [[self._metadatas[i] for i in top_idx]],
            "distances": [[float(1 - sims[i]) for i in top_idx]],  # cosine distance
        }


def get_collection():
    """Try real ChromaDB first; fall back to SimpleVectorStore if unavailable."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        backend = "chromadb (persistent, on-disk)"
        return collection, backend
    except ImportError:
        print("[info] chromadb not installed in this sandbox (no internet access). "
              "Using offline SimpleVectorStore fallback with the same add()/query() interface.")
        store = SimpleVectorStore(COLLECTION_NAME, CHROMA_DIR)
        return store, "SimpleVectorStore (offline fallback, NumPy + on-disk .npz)"


def load_corpus_and_embeddings():
    if not EMBEDDINGS_PATH.exists() or not CORPUS_PATH.exists():
        raise SystemExit(
            "Missing Checkpoint 1 artifacts. Run:\n"
            "  python src/preprocessing/clean_text.py\n"
            "  python src/embeddings/generate_embeddings.py\nfirst."
        )
    npz = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    vectors = npz["vectors"]
    doc_ids = list(npz["doc_ids"])

    records_by_id = {}
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records_by_id[rec["doc_id"]] = rec

    documents = [records_by_id[d]["cleaned_text"] for d in doc_ids]
    metadatas = [{"doc_id": d, "source_file": records_by_id[d]["source_file"]} for d in doc_ids]
    return doc_ids, vectors, documents, metadatas


def embed_query_like_checkpoint1(query_text: str, vectors_shape_hint):
    """
    Re-uses the same fallback embedding approach from Checkpoint 1
    (TF-IDF, since sentence-transformers requires an internet download that
    isn't available in this sandbox) to embed a new query consistently with
    the stored document vectors.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embeddings"))
    from generate_embeddings import load_corpus, fallback_tfidf, embed_query_tfidf  # noqa

    records = load_corpus()
    texts = [r["cleaned_text"] for r in records]
    _, _, vectorizer = fallback_tfidf(texts)
    return embed_query_tfidf(vectorizer, query_text)


def main():
    doc_ids, vectors, documents, metadatas = load_corpus_and_embeddings()
    collection, backend = get_collection()

    print(f"Backend in use: {backend}")

    collection.add(ids=doc_ids, embeddings=vectors, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(doc_ids)} documents into collection '{COLLECTION_NAME}'.")

    # --- Similarity search demo ---
    sample_queries = [
        "How many units of overload can a graduating student take?",
        "What is the curfew for students staying on campus overnight?",
        "How do I apply for a scholarship at PUP?",
    ]

    for q in sample_queries:
        query_vec = embed_query_like_checkpoint1(q, vectors.shape)
        results = collection.query(query_embeddings=[query_vec], n_results=3)
        print(f"\nQuery: {q!r}")
        for rank, (doc_id, dist) in enumerate(
            zip(results["ids"][0], results["distances"][0]), start=1
        ):
            print(f"  {rank}. {doc_id}  (cosine distance={dist:.4f})")


if __name__ == "__main__":
    main()
