"""
Checkpoint 1 — Text Preprocessing Script
PUP Student FAQ RAG Assistant

Reads raw FAQ/policy documents from data/raw/, applies a cleaning and
normalization pipeline, tokenizes the result, and writes:
  - data/processed/cleaned_corpus.jsonl   (one cleaned record per doc)
  - data/processed/preprocessing_report.md (before/after examples for the proposal)

Usage:
    python src/preprocessing/clean_text.py
"""

import json
import re
import unicodedata
from pathlib import Path

try:
    import nltk
    from nltk.tokenize import word_tokenize
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def ensure_nltk_data():
    """Make sure the punkt tokenizer models are available."""
    if not _HAS_NLTK:
        return
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def simple_tokenize(text: str):
    """Fallback whitespace/punctuation tokenizer if NLTK data isn't available."""
    return re.findall(r"[A-Za-z0-9']+|[.,!?;:()\-]", text)


def clean_text(raw_text: str) -> str:
    """
    Cleaning + normalization pipeline:
      1. Strip Markdown heading markers and formatting artifacts
      2. Unicode normalization (NFKC) to fix odd characters from PDFs/scrapes
      3. Collapse whitespace/newlines
      4. Lowercase-preserve version kept separate for embeddings (case matters
         less for embeddings, but we keep original case for display text)
      5. Remove stray bullet/list markers
      6. Strip leading/trailing whitespace
    """
    text = raw_text

    # 1. Remove markdown heading hashes but keep the heading text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # 2. Unicode normalization — collapses odd PDF ligatures/quote variants
    text = unicodedata.normalize("NFKC", text)

    # 3. Normalize smart quotes/dashes to plain ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    # 4. Remove markdown bullet markers ("- ", "* ") at line start
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)

    # 5. Remove markdown bold/italic markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)

    # 6. Collapse multiple blank lines / excess whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def tokenize(text: str):
    if _HAS_NLTK:
        try:
            return word_tokenize(text)
        except LookupError:
            pass
    return simple_tokenize(text)


def process_corpus():
    ensure_nltk_data()

    raw_files = sorted(RAW_DIR.glob("*.md"))
    if not raw_files:
        raise SystemExit(f"No raw documents found in {RAW_DIR}")

    records = []
    report_lines = [
        "# Preprocessing Report — Before / After Examples\n",
        f"Total raw documents processed: **{len(raw_files)}**\n",
    ]

    for i, fpath in enumerate(raw_files, start=1):
        raw = fpath.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        tokens = tokenize(cleaned)

        record = {
            "doc_id": fpath.stem,
            "source_file": fpath.name,
            "raw_char_count": len(raw),
            "cleaned_char_count": len(cleaned),
            "token_count": len(tokens),
            "cleaned_text": cleaned,
            "tokens_preview": tokens[:25],
        }
        records.append(record)

        # Capture a few before/after examples in the report
        if i <= 5:
            report_lines.append(f"## Example {i}: `{fpath.name}`\n")
            report_lines.append("**Before (raw, first 300 chars):**")
            report_lines.append(f"```\n{raw[:300].strip()}\n```")
            report_lines.append("**After (cleaned, first 300 chars):**")
            report_lines.append(f"```\n{cleaned[:300].strip()}\n```")
            report_lines.append(
                f"- Raw length: {len(raw)} chars → Cleaned length: {len(cleaned)} chars\n"
                f"- Token count: {len(tokens)}\n"
                f"- First 15 tokens: {tokens[:15]}\n"
            )

    # Write cleaned corpus as JSONL (one document per line)
    out_path = PROCESSED_DIR / "cleaned_corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write the before/after report
    report_path = PROCESSED_DIR / "preprocessing_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Processed {len(records)} documents.")
    print(f"Cleaned corpus written to: {out_path}")
    print(f"Before/after report written to: {report_path}")


if __name__ == "__main__":
    process_corpus()
