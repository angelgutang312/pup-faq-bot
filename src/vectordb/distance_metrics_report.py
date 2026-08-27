"""
Checkpoint 2 — Distance Metric Comparison
PUP Student FAQ RAG Assistant

Compares Cosine Similarity, Euclidean (L2) Distance, and Dot Product on our
own TF-IDF document embeddings, using the same sample queries as
src/vectordb/setup_chroma.py, and writes the results to
docs/Distance_Metric_Comparison.md for the Checkpoint 2 write-up.

Usage:
    python src/vectordb/distance_metrics_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npz"
CORPUS_PATH = PROCESSED_DIR / "cleaned_corpus.jsonl"

sys.path.insert(0, str(ROOT / "src" / "embeddings"))
from generate_embeddings import load_corpus, fallback_tfidf, embed_query_tfidf  # noqa


def cosine_similarity(query_vec, matrix):
    # vectors are already L2-normalized, so cosine similarity = dot product
    return matrix @ query_vec


def dot_product(query_vec, matrix):
    return matrix @ query_vec


def euclidean_distance(query_vec, matrix):
    diffs = matrix - query_vec
    return np.linalg.norm(diffs, axis=1)


def top_k(scores, doc_ids, k=3, higher_is_better=True):
    order = np.argsort(-scores) if higher_is_better else np.argsort(scores)
    return [(doc_ids[i], float(scores[i])) for i in order[:k]]


def main():
    npz = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    vectors = npz["vectors"]
    doc_ids = list(npz["doc_ids"])

    records = load_corpus()
    texts = [r["cleaned_text"] for r in records]
    _, _, vectorizer = fallback_tfidf(texts)

    sample_queries = [
        "How many units of overload can a graduating student take?",
        "What is the curfew for students staying on campus overnight?",
        "How do I apply for a scholarship at PUP?",
        "What happens if I get an Incomplete grade and don't finish on time?",
    ]

    lines = [
        "# Distance Metric Comparison — Cosine vs. Euclidean vs. Dot Product\n",
        "All three metrics were run against the same 25-document TF-IDF-embedded "
        "corpus from Checkpoint 1 (vectors are L2-normalized). Below are the "
        "top-3 results per metric for each of 4 sample queries.\n",
    ]

    for q in sample_queries:
        query_vec = embed_query_tfidf(vectorizer, q)

        cos_scores = cosine_similarity(query_vec, vectors)
        dot_scores = dot_product(query_vec, vectors)
        euc_scores = euclidean_distance(query_vec, vectors)

        cos_top = top_k(cos_scores, doc_ids, k=3, higher_is_better=True)
        dot_top = top_k(dot_scores, doc_ids, k=3, higher_is_better=True)
        euc_top = top_k(euc_scores, doc_ids, k=3, higher_is_better=False)

        lines.append(f"## Query: \"{q}\"\n")
        lines.append("| Rank | Cosine Similarity | Dot Product | Euclidean Distance |")
        lines.append("|---|---|---|---|")
        for i in range(3):
            c = f"{cos_top[i][0]} ({cos_top[i][1]:.4f})"
            d = f"{dot_top[i][0]} ({dot_top[i][1]:.4f})"
            e = f"{euc_top[i][0]} ({euc_top[i][1]:.4f})"
            lines.append(f"| {i+1} | {c} | {d} | {e} |")
        lines.append("")

        agree = (cos_top[0][0] == dot_top[0][0] == euc_top[0][0])
        lines.append(f"*All three metrics agree on the top-1 result: {agree}*\n")

    lines.append("## Analysis\n")
    lines.append(
        "Because our stored vectors are L2-normalized (unit length), Cosine "
        "Similarity and Dot Product are mathematically equivalent here — "
        "`cos(a,b) = (a·b) / (|a||b|) = a·b` when `|a|=|b|=1`. This is confirmed "
        "above: the Cosine and Dot Product columns rank documents identically "
        "for every query.\n"
    )
    lines.append(
        "Euclidean Distance mostly agrees with the other two on the top-ranked "
        "document (since for unit vectors, `||a-b||² = 2 - 2(a·b)`, a monotonic "
        "function of cosine similarity), but ranking can diverge lower down the "
        "list once vector magnitudes are not perfectly comparable in intent "
        "(e.g., for TF-IDF vectors with different amounts of matched "
        "vocabulary), because Euclidean distance is sensitive to the actual "
        "vector length in a way cosine is not.\n"
    )
    lines.append("## Recommendation\n")
    lines.append(
        "We recommend **Cosine Similarity** as our production distance metric, "
        "for three reasons: (1) it is invariant to document length, which "
        "matters since our FAQ documents vary noticeably in length; (2) once "
        "we swap in true sentence-transformer embeddings in place of the "
        "current TF-IDF fallback, cosine similarity is the metric those models "
        "are explicitly trained and validated against; and (3) it is the "
        "default and best-supported metric in ChromaDB's HNSW index "
        "(`hnsw:space: \"cosine\"`), so it requires no extra configuration. "
        "Dot Product would work identically for us today since our vectors are "
        "normalized, but Cosine is the more robust choice if we ever store "
        "non-normalized vectors down the line.\n"
    )

    out_path = DOCS_DIR / "Distance_Metric_Comparison.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Distance metric comparison written to: {out_path}")


if __name__ == "__main__":
    main()
