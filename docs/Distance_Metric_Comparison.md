# Distance Metric Comparison — Cosine vs. Euclidean vs. Dot Product

All three metrics were run against the same 25-document TF-IDF-embedded corpus from Checkpoint 1 (vectors are L2-normalized). Below are the top-3 results per metric for each of 4 sample queries.

## Query: "How many units of overload can a graduating student take?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 01_pupcet_admission (0.0728) | 01_pupcet_admission (0.0728) | 01_pupcet_admission (1.3618) |
| 2 | 16_student_council (0.0719) | 16_student_council (0.0719) | 16_student_council (1.3624) |
| 3 | 24_open_university_and_itech (0.0699) | 24_open_university_and_itech (0.0699) | 24_open_university_and_itech (1.3639) |

*All three metrics agree on the top-1 result: True*

## Query: "What is the curfew for students staying on campus overnight?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 04_classification_of_students (0.0609) | 04_classification_of_students (0.0609) | 04_classification_of_students (1.3705) |
| 2 | 09_scholarships (0.0524) | 09_scholarships (0.0524) | 09_scholarships (1.3767) |
| 3 | 16_student_council (0.0395) | 16_student_council (0.0395) | 16_student_council (1.3860) |

*All three metrics agree on the top-1 result: True*

## Query: "How do I apply for a scholarship at PUP?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 11_id_and_uniform_policy (0.0620) | 11_id_and_uniform_policy (0.0620) | 11_id_and_uniform_policy (1.3697) |
| 2 | 01_pupcet_admission (0.0447) | 01_pupcet_admission (0.0447) | 01_pupcet_admission (1.3822) |
| 3 | 25_guidance_counseling (0.0259) | 25_guidance_counseling (0.0259) | 25_guidance_counseling (1.3958) |

*All three metrics agree on the top-1 result: True*

## Query: "What happens if I get an Incomplete grade and don't finish on time?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 22_shifting_and_transfer (0.0997) | 22_shifting_and_transfer (0.0997) | 22_shifting_and_transfer (1.3419) |
| 2 | 10_dean_academic_standing (0.0486) | 10_dean_academic_standing (0.0486) | 10_dean_academic_standing (1.3794) |
| 3 | 23_library_services (0.0471) | 23_library_services (0.0471) | 23_library_services (1.3805) |

*All three metrics agree on the top-1 result: True*

## Analysis

Because our stored vectors are L2-normalized (unit length), Cosine Similarity and Dot Product are mathematically equivalent here — `cos(a,b) = (a·b) / (|a||b|) = a·b` when `|a|=|b|=1`. This is confirmed above: the Cosine and Dot Product columns rank documents identically for every query.

Euclidean Distance mostly agrees with the other two on the top-ranked document (since for unit vectors, `||a-b||² = 2 - 2(a·b)`, a monotonic function of cosine similarity), but ranking can diverge lower down the list once vector magnitudes are not perfectly comparable in intent (e.g., for TF-IDF vectors with different amounts of matched vocabulary), because Euclidean distance is sensitive to the actual vector length in a way cosine is not.

## Recommendation

We recommend **Cosine Similarity** as our production distance metric, for three reasons: (1) it is invariant to document length, which matters since our FAQ documents vary noticeably in length; (2) once we swap in true sentence-transformer embeddings in place of the current TF-IDF fallback, cosine similarity is the metric those models are explicitly trained and validated against; and (3) it is the default and best-supported metric in ChromaDB's HNSW index (`hnsw:space: "cosine"`), so it requires no extra configuration. Dot Product would work identically for us today since our vectors are normalized, but Cosine is the more robust choice if we ever store non-normalized vectors down the line.
