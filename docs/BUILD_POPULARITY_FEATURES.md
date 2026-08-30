# Building External Popularity Features

This guide rebuilds the external features used by the paper from model
responses, Wikidata, and English Wikipedia. It is optional for numerical
reproduction because the compact model-output archives already contain the
final response-level values.

## Definitions and source snapshot

| Output | Definition | Source |
|---|---|---|
| Entity popularity | Number of Wikidata sitelinks for an entity | Wikidata Query Service |
| Single occurrence | Number of English Wikipedia documents containing an entity string | Wikipedia `20231101.en` |
| Pairwise co-occurrence | Number of English Wikipedia documents containing both entity strings | Wikipedia `20231101.en` |

The paper cites the
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
dataset. The original preprocessing layout—41 English parquet shards—matches
the `20231101.en` configuration. Pin that configuration for comparable counts:

```bash
python -m pip install -r requirements-data.txt
hf download wikimedia/wikipedia \
  --repo-type dataset \
  --include "20231101.en/*.parquet" \
  --local-dir external_data/wikipedia
```

This downloads several GB and is intentionally not part of the Git
repository. The expected parquet columns are `id` and `text`. If a differently
named but equivalent snapshot is used, pass `--id-column` and `--text-column`
to the indexing command and report the snapshot in the reproduction record.

## 1. Extract entities and required pairs

The command accepts both the compact `.jsonl.gz` files and raw response JSONL
files produced by the inference pipeline:

```bash
python code/prepare_before_analysis/build_external_popularity.py prepare \
  --inputs data/model_outputs/*.jsonl.gz \
  --entities-output external_data/entities.jsonl \
  --pairs-output external_data/entity_pairs.jsonl
```

`entities.jsonl` stores a case-folded entity key, all observed surface forms,
and whether it appeared as a question, ground-truth, or generated entity.
`entity_pairs.jsonl` stores only the question–answer pairs whose co-occurrence
is needed, avoiding an unnecessary all-pairs computation.

## 2. Obtain Wikidata sitelink counts

Set a descriptive user agent, including a contact address when possible, and
run the resumable Wikidata query:

```bash
export WIKIDATA_USER_AGENT="knowledge-popularity-reproduction/1.0 contact@example.org"
python code/prepare_before_analysis/build_external_popularity.py wikidata \
  --entities external_data/entities.jsonl \
  --output external_data/wikidata_sitelinks.jsonl \
  --resume
```

For every entity, the output records `wikidata_id`, `sitelinks`, `found`, and
the matched surface form. Exact English labels are queried and, when multiple
items share a label, the item with the largest sitelink count is selected.
Requests are rate-limited, retried, flushed after each entity, and can be
resumed after interruption.

The response-level paper fields `qpop`, `gt_pop`, and `gene_pop` are obtained
by looking up the corresponding normalized entity key in this file. Missing
entities are represented as zero in downstream experiments while the `found`
flag remains available to distinguish missing from genuinely zero-popularity
entities.

## 3. Build the Wikipedia entity-document index

```bash
python code/prepare_before_analysis/build_external_popularity.py wikipedia-index \
  --entities external_data/entities.jsonl \
  --wikipedia external_data/wikipedia/20231101.en \
  --output external_data/entity_documents.parquet
```

The implementation reproduces the paper's original matching rule:
case-insensitive surface-string matching over each document. Each entity is
counted at most once per document. This deliberately differs from Wikidata
entity linking and is noisier for ambiguous aliases and common words.

The index is the large derived intermediate artifact. Its size depends on the
number of entities and can reach several GB, so it should be kept outside Git
or distributed in the optional full artifact archive.

## 4. Compute occurrence and co-occurrence counts

```bash
python code/prepare_before_analysis/build_external_popularity.py aggregate \
  --index external_data/entity_documents.parquet \
  --pairs external_data/entity_pairs.jsonl \
  --single-output external_data/single_occurrence.json \
  --cooccurrence-output external_data/cooccurrence.json
```

For an entity `e`, `single_occurrence[e]` is the number of unique document IDs
in its index row. For a pair `(q, a)`, `cooccurrence[q][a]` is the size of the
intersection between their document-ID sets.

These resources produce the response-level fields as follows:

| Response-level field | Lookup |
|---|---|
| `q_occ` | `single_occurrence[question_entity]` |
| `gt_occ` | `single_occurrence[ground_truth_entity]` |
| `gene_occ` | `single_occurrence[generated_entity]` |
| `gt_coo` | `cooccurrence[question_entity][ground_truth_entity]` |
| `gene_coo` | `cooccurrence[question_entity][generated_entity]` |

## 5. Attach the fields to response rows

Run this once per model-response file:

```bash
python code/prepare_before_analysis/build_external_popularity.py annotate \
  --input path/to/model_responses.jsonl \
  --wikidata external_data/wikidata_sitelinks.jsonl \
  --single external_data/single_occurrence.json \
  --cooccurrence external_data/cooccurrence.json \
  --output external_data/model_responses_with_external_popularity.jsonl
```

This adds `qpop`, `gt_pop`, `gene_pop`, their lookup flags, `q_occ`, `gt_occ`,
`gene_occ`, `gt_coo`, and `gene_coo`. It uses the first accepted reference for
the ground-truth lookup, matching the released analysis pipeline. Missing
lookups are encoded as zero while their `*_found` field is false.

## Methodological note

The paper uses Wikidata sitelinks—not Wikipedia string occurrences—as its
primary single-entity popularity feature. Wikipedia occurrence counts contain
more alias and ambiguity noise and are retained for high-frequency filtering
and relation-specificity normalization. Wikipedia is nevertheless necessary
for pairwise co-occurrence because Wikidata does not measure how often two
mentions appear together in natural-language documents.
