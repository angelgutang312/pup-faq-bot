# Distance Metric Comparison — Cosine vs. Euclidean vs. Dot Product

All three metrics were run against the same 25-document TF-IDF-embedded corpus from Checkpoint 1 (vectors are L2-normalized). Below are the top-3 results per metric for each of 4 sample queries.

## Query: "How many units of overload can a graduating student take?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 03_academic_load (0.1960) | 03_academic_load (0.1960) | 03_academic_load (1.2681) |
| 2 | 16_student_council (0.0657) | 16_student_council (0.0657) | 16_student_council (1.3670) |
| 3 | 01_pupcet_admission (0.0589) | 01_pupcet_admission (0.0589) | 01_pupcet_admission (1.3719) |

*All three metrics agree on the top-1 result: True*

## Query: "What is the curfew for students staying on campus overnight?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 14_curfew_dormitory_rules (0.4127) | 14_curfew_dormitory_rules (0.4127) | 14_curfew_dormitory_rules (1.0838) |
| 2 | 23_library_services (0.0503) | 23_library_services (0.0503) | 23_library_services (1.3782) |
| 3 | 01_pupcet_admission (0.0424) | 01_pupcet_admission (0.0424) | 01_pupcet_admission (1.3839) |

*All three metrics agree on the top-1 result: True*

## Query: "How do I apply for a scholarship at PUP?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 09_scholarships (0.2294) | 09_scholarships (0.2294) | 09_scholarships (1.2414) |
| 2 | 05_grading_system (0.0536) | 05_grading_system (0.0536) | 05_grading_system (1.3758) |
| 3 | 01_pupcet_admission (0.0311) | 01_pupcet_admission (0.0311) | 01_pupcet_admission (1.3921) |

*All three metrics agree on the top-1 result: True*

## Query: "What happens if I get an Incomplete grade and don't finish on time?"

| Rank | Cosine Similarity | Dot Product | Euclidean Distance |
|---|---|---|---|
| 1 | 07_correction_of_grades (0.1519) | 07_correction_of_grades (0.1519) | 07_correction_of_grades (1.3024) |
| 2 | 08_incomplete_grades (0.1339) | 08_incomplete_grades (0.1339) | 08_incomplete_grades (1.3161) |
| 3 | 05_grading_system (0.1320) | 05_grading_system (0.1320) | 05_grading_system (1.3176) |

*All three metrics agree on the top-1 result: True*

## Analysis

Because our stored vectors are L2-normalized (unit length), Cosine Similarity and Dot Product are mathematically equivalent here — `cos(a,b) = (a·b) / (|a||b|) = a·b` when `|a|=|b|=1`. This is confirmed above: the Cosine and Dot Product columns rank documents identically for every query.

Euclidean Distance mostly agrees with the other two on the top-ranked document (since for unit vectors, `||a-b||² = 2 - 2(a·b)`, a monotonic function of cosine similarity), but ranking can diverge lower down the list once vector magnitudes are not perfectly comparable in intent (e.g., for TF-IDF vectors with different amounts of matched vocabulary), because Euclidean distance is sensitive to the actual vector length in a way cosine is not.

## Recommendation

We recommend **Cosine Similarity** as our production distance metric, for three reasons: (1) it is invariant to document length, which matters since our FAQ documents vary noticeably in length; (2) once we swap in true sentence-transformer embeddings in place of the current TF-IDF fallback, cosine similarity is the metric those models are explicitly trained and validated against; and (3) it is the default and best-supported metric in ChromaDB's HNSW index (`hnsw:space: "cosine"`), so it requires no extra configuration. Dot Product would work identically for us today since our vectors are normalized, but Cosine is the more robust choice if we ever store non-normalized vectors down the line.
