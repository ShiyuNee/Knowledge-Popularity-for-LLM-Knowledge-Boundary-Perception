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

Each row contains:

| Field | Meaning |
|---|---|
| `dataset`, `index` | Dataset identity and stable original row index |
| `question`, `reference`, `response` | QA input, ground truth, and model answer |
| `correct`, `confidence` | Exact-match correctness and mean answer-token probability |
| `qpop` | Wikidata sitelink popularity of the question entity |
| `gt_pop`, `gene_pop` | Wikidata popularity of the ground-truth and generated entities |
| `gt_pop_found`, `gene_pop_found` | Whether each entity was found in the popularity resource |
| `gt_coo` | Number of Wikipedia documents containing both the question entity and ground-truth answer entity |
| `gene_coo` | Number of Wikipedia documents containing both the question entity and generated answer entity |
| `q_occ` | Number of Wikipedia documents containing the question entity |
| `gt_occ` | Number of Wikipedia documents containing the ground-truth answer entity |
| `gene_occ` | Number of Wikipedia documents containing the generated answer entity |
| `sc_confidence`, `vc_confidence` | Self-consistency and verbalized-confidence scores |
| `llm_qpop`, `llm_gene_pop`, `llm_coo` | Zero-shot LLM-estimated popularity scores |

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

To inspect a file without extracting it:

```bash
gzip -cd data/model_outputs/llama3-8b.jsonl.gz | head -n 1
```

The files are generated deterministically by:

```bash
python scripts/build_compact_release_data.py
```
