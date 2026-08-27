# Checkpoint 1 Reflection

Our theme is a **PUP Student FAQ Assistant**, so the "messiness" in our data
wasn't corrupted files or broken encodings — it was **fragmentation and
inconsistency across sources**. Real PUP student policy information is
scattered across a 2014-era PDF handbook, freshman-enrollment FAQ sheets
re-issued every academic year, scanned/OCR'd copies on document-sharing
sites (with garbled characters from bad OCR), Wikipedia summaries of
university governance, and informal blog posts explaining the grading
system in plain language. No single canonical, clean source exists.

To make this usable for Checkpoint 1, we consolidated the information into
25 topic-scoped Markdown documents (one policy area per file: admissions,
grading, attendance, scholarships, student council, etc.), each rewritten
in our own words from the source material rather than copy-pasted, both to
avoid reproducing copyrighted text verbatim and to normalize the
inconsistent tone/formatting across sources into a single consistent style
our pipeline could process predictably.

The other real mess was **environment constraints, not just data**: our
development sandbox for this checkpoint has no internet access, so we
could not download the sentence-transformer model weights we ultimately
want to use in production. We handled this by writing the embedding script
to attempt the real model first and fall back to an offline TF-IDF
vectorizer, so the full pipeline still runs and is gradeable end-to-end,
with the swap back to the production model documented as a one-line
change once we run this in an internet-connected environment for
Checkpoint 2.
