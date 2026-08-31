# Lightweight QA Inputs

This directory is intentionally included in the public repository.

| File | Relation | Rows |
|---|---|---:|
| `movies.jsonl` | movie → director | 10,964 |
| `songs.jsonl` | song → performer | 2,157 |
| `basketball.jsonl` | basketball player → birthplace | 13,309 |

Each line contains a question, one or more reference answers, the question
entity, and its Wikidata sitelink-based popularity.

Model responses, token probabilities, Wikipedia co-occurrence indices, and
baseline generations are released in a minimized, analysis-ready form under
[`model_outputs/`](model_outputs/). See
[`docs/DATA_AND_ARTIFACTS.md`](../docs/DATA_AND_ARTIFACTS.md) for the compact
schema, excluded large artifacts, and regeneration instructions.
