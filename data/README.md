# Lightweight QA Inputs

This directory is intentionally included in the public repository.

| File | Relation | Rows |
|---|---|---:|
| `movies.jsonl` | movie → director | 10,964 |
| `songs.jsonl` | song → performer | 2,157 |
| `basketball.jsonl` | basketball player → birthplace | 13,309 |

## QA input schema

Each line is one JSON object:

| Field | JSON type | Description |
|---|---|---|
| `question` | string | Natural-language factual question. |
| `reference` | array of strings | One or more accepted answer aliases. |
| `popularity` | integer | Wikidata sitelink count for the question entity. |
| `question_entity` | string | Subject entity mentioned by the question. |

Example:

```json
{
  "question": "Who is the director of the movie The Intouchables",
  "reference": ["Olivier Nakache", "Éric Toledano"],
  "popularity": 60,
  "question_entity": "The Intouchables"
}
```

Read a QA file with:

```python
import json

with open("data/movies.jsonl", encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
```

Model responses, aggregated token confidence, Wikipedia co-occurrence indices,
and baseline scores are released in a minimized, analysis-ready form under
[`model_outputs/`](model_outputs/).
See [`model_outputs/README.md`](model_outputs/README.md) for the complete
compact data dictionary, missing-value semantics, example row, and loading
code. See [`docs/DATA_AND_ARTIFACTS.md`](../docs/DATA_AND_ARTIFACTS.md) for the
compact schema, excluded large artifacts, and regeneration instructions.
