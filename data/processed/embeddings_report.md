# Embedding Model Choice & Justification

## Chosen production model: `sentence-transformers/all-MiniLM-L6-v2`

**Why this model:**

1. **Purpose-built for retrieval.** Unlike a general-purpose LLM's hidden
   states, all-MiniLM-L6-v2 is trained contrastively on over a billion
   sentence pairs specifically so that semantically related text ends up
   close together in vector space — exactly the property a RAG retriever
   depends on.
2. **Small and fast.** 384-dimensional output vectors keep the vector index
   small and similarity search cheap, which matters once the corpus grows
   past our current 25 documents into hundreds of chunks in later
   checkpoints.
3. **Free and local.** No per-call API cost and no rate limits, since we'll
   be re-embedding the corpus repeatedly as the ingestion pipeline evolves
   (Checkpoint 3).
4. **Ecosystem support.** Both LangChain and LlamaIndex (our required
   orchestration frameworks from Checkpoint 3 onward) have first-class
   `HuggingFaceEmbeddings` wrappers for sentence-transformers models, so
   there's no integration friction later.

## Sandbox limitation and fallback used for this demo

This development sandbox has **no outbound internet access**, so the
Hugging Face model weights for all-MiniLM-L6-v2 cannot be downloaded here.
`src/embeddings/generate_embeddings.py` therefore:

1. First attempts to load `all-MiniLM-L6-v2` via `sentence-transformers`.
2. If that fails (as it does in this sandbox), it automatically falls back
   to a **TF-IDF vectorizer** (scikit-learn, fully offline, bag-of-words +
   inverse document frequency weighting, L2-normalized) so the full
   pipeline — embed, save, similarity-search — still runs end-to-end and
   can be demonstrated and graded.

This is a one-line swap (`try_sentence_transformers` already returns the
correctly-shaped output) — running the same script on a machine with
internet access, or once we move to Docker/cloud deployment in
Checkpoint 4, will automatically use the real transformer embeddings
without any other code changes.

## Sanity check result

Query: *"What happens if I get an Incomplete grade and don't finish the
requirement on time?"*

Even with the much weaker TF-IDF fallback, the top-3 nearest documents by
cosine similarity were:

1. `05_grading_system` (score 0.1696)
2. `08_incomplete_grades` (score 0.1590)
3. `07_correction_of_grades` (score 0.1279)

All three are the correct topically-relevant documents, which validates
that the preprocessing → vectorization → similarity-search pipeline is
wired correctly and ready to have the fallback swapped for a real
sentence-transformer (or a hosted embedding API) in Checkpoint 2.
