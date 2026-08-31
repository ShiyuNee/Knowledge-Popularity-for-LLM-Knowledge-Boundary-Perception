# Compact Model Outputs

This directory is the default public reproduction dataset. It contains one
gzip-compressed JSONL file per model; each file combines the movies, songs, and
basketball examples.

| File | Rows | Approximate size |
|---|---:|---:|
| `llama3-8b.jsonl.gz` | 26,430 | 1.6 MB |
| `qwen2-7b.jsonl.gz` | 26,430 | 1.6 MB |
| `gpt-3.5-turbo.jsonl.gz` | 26,430 | 1.6 MB |
| `qwen2.5-7b.jsonl.gz` | 26,430 | 1.5 MB |
| `qwen2.5-14b.jsonl.gz` | 26,430 | 1.5 MB |
| `qwen2.5-32b.jsonl.gz` | 26,430 | 1.5 MB |

## Row schema

Each line is one JSON object. `null` means that the corresponding upstream
value was unavailable or could not be parsed; it does not mean zero.

### Identity and QA fields

| Field | JSON type | Allowed values | Description |
|---|---|---|---|
| `dataset` | string | `movies`, `songs`, `basketball` | Source dataset. |
| `index` | integer | `0` to dataset size minus one | Zero-based row index within the dataset. The same `(dataset, index)` identifies the same question in every model archive. |
| `question` | string | non-empty | Factual QA prompt. |
| `reference` | array of strings | one or more answers | Accepted ground-truth answer aliases. |
| `response` | string | model-generated text | The archived model answer. |
| `correct` | integer | `0` or `1` | Whether the normalized tokens of any reference answer occur as a contiguous sequence in the response. Matching is case-insensitive and normalizes punctuation, articles, whitespace, and Unicode representation. |
| `confidence` | number or `null` | `[0, 1]` | Arithmetic mean of the generated answer-token probabilities. `null` indicates missing token probabilities. |

`correct` is a normalized containment score, not strict string equality. For
example, an answer may contain additional text and still receive `1` when it
contains a complete reference answer after normalization. Unicode text is
normalized, but diacritics are not discarded; for example, `Eric` does not
match `Éric` in the archived scoring rule.

### Popularity and occurrence fields

| Field | JSON type | Allowed values | Description |
|---|---|---|---|
| `qpop` | integer | `>= 0` | Number of Wikidata sitelinks for the question entity. |
| `gt_pop` | integer | `>= 0` | Number of Wikidata sitelinks for the first ground-truth answer entity. |
| `gene_pop` | integer | `>= 0` | Number of Wikidata sitelinks for the cleaned generated-answer entity. |
| `gt_pop_found` | boolean | `true` or `false` | Whether the ground-truth entity was found in the popularity resource. |
| `gene_pop_found` | boolean | `true` or `false` | Whether the generated entity was found in the popularity resource. |
| `gt_coo` | integer or `null` | `>= 0` | Number of Wikipedia documents containing both the question entity and ground-truth answer entity. |
| `gene_coo` | integer or `null` | `>= 0` | Number of Wikipedia documents containing both the question entity and generated-answer entity. |
| `q_occ` | integer or `null` | `>= 0` | Number of Wikipedia documents containing the question entity. |
| `gt_occ` | integer or `null` | `>= 0` | Number of Wikipedia documents containing the ground-truth answer entity. |
| `gene_occ` | integer or `null` | `>= 0` | Number of Wikipedia documents containing the generated-answer entity. |

When `gt_pop_found` or `gene_pop_found` is `false`, the corresponding compact
popularity value is stored as `0`. Always use the `*_found` flag when you need
to distinguish a missing entity from a genuine zero-valued count.

### Confidence-baseline and estimated-popularity fields

| Field | JSON type | Allowed values | Description |
|---|---|---|---|
| `sc_confidence` | number or `null` | `[0, 1]` | Self-consistency confidence derived from repeated generations. |
| `vc_confidence` | number or `null` | `[0, 1]` | Verbalized-confidence score. |
| `llm_qpop` | integer or `null` | `1` to `10` | Zero-shot LLM estimate of question-entity popularity. |
| `llm_gene_pop` | integer or `null` | `1` to `10` | Zero-shot LLM estimate of generated-answer popularity. |
| `llm_coo` | integer or `null` | `1` to `10` | Zero-shot LLM estimate of question/answer relation popularity. |

## Example row

```json
{
  "dataset": "movies",
  "index": 0,
  "question": "Who is the director of the movie The Intouchables",
  "reference": ["Olivier Nakache", "Éric Toledano"],
  "response": "Eric Toledano",
  "correct": 0,
  "confidence": 0.9888770878314972,
  "qpop": 60,
  "gt_pop": 15,
  "gt_pop_found": true,
  "gene_pop": 0,
  "gene_pop_found": false,
  "gt_coo": 9,
  "gene_coo": 2,
  "q_occ": 63,
  "gt_occ": 32,
  "gene_occ": 2,
  "sc_confidence": 0.2,
  "vc_confidence": 1.0,
  "llm_qpop": 8,
  "llm_gene_pop": 4,
  "llm_coo": 8
}
```

## Reading the archives

```python
import gzip
import json

path = "data/model_outputs/llama3-8b.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as stream:
    first_row = json.loads(next(stream))

print(first_row["dataset"], first_row["confidence"])
```

## Why entity popularity and co-occurrence use different sources

The primary entity-popularity variables in the paper are `qpop`, `gt_pop`,
and `gene_pop`. They are computed from the number of Wikidata sitelinks for an
entity. This entity-linked statistic is substantially cleaner than counting
raw entity strings in Wikipedia text.

The Wikipedia single-entity counts `q_occ`, `gt_occ`, and `gene_occ` are more
sensitive to ambiguous names, aliases, and common-word matches. Consequently,
the paper does **not** use these three fields as its primary entity-popularity
measure. They are retained because the experiment code uses them to:

1. filter extremely frequent, noisy entity matches; and
2. normalize ground-truth co-occurrence when computing relation specificity
   (for example, `gt_coo / q_occ`).

Pairwise fields `gt_coo` and `gene_coo` must instead come from Wikipedia.
Wikidata can provide structured entity relations and sitelink counts, but it
does not record how frequently two entities are mentioned together in a
natural-language corpus. Wikipedia document-level co-occurrence approximates
this exposure signal.

In short:

| Quantity | Source | Role in the paper |
|---|---|---|
| `qpop`, `gt_pop`, `gene_pop` | Wikidata sitelinks | Primary single-entity popularity |
| `q_occ`, `gt_occ`, `gene_occ` | Wikipedia documents | Noise filtering and normalization |
| `gt_coo`, `gene_coo` | Wikipedia documents | Pairwise knowledge/exposure signal |

The archives intentionally omit prompts, token IDs, per-token entropy,
top-logprob dictionaries, repeated sampled answers, and judge traces. Those
fields are not needed to reproduce the paper's reported statistics.

To create the legacy directory layout expected by the original analysis code:

```bash
python scripts/materialize_compact_data.py
```

To inspect a file from the shell without extracting it:

```bash
gzip -cd data/model_outputs/llama3-8b.jsonl.gz | head -n 1
```

The files are generated deterministically by:

```bash
python scripts/build_compact_release_data.py
```
