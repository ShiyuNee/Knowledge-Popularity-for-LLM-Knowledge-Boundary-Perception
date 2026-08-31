# Data and Artifact Availability

This repository intentionally keeps the Git history small. It includes source
code, lightweight QA inputs, compact model outputs, and result summaries.
It does **not** include model checkpoints, raw Wikipedia dumps, model-generation
traces, or multi-gigabyte intermediate artifacts.

## What is included

| Path | Contents | Approximate size |
|---|---|---:|
| `data/movies.jsonl` | Movie-to-director QA inputs | 1.7 MB |
| `data/songs.jsonl` | Song-to-performer QA inputs | 0.3 MB |
| `data/basketball.jsonl` | Player-to-birthplace QA inputs | 2.1 MB |
| `data/model_outputs/*.jsonl.gz` | Six response-level reproduction files with joined popularity features | ~9 MB total |
| `code/analysis_correlation/*_results.json` | Compact experiment summaries | < 2 MB total |

The QA inputs are derived from the entity-centric Wikidata datasets used in
[Attention Satisfies](https://arxiv.org/abs/2309.15098) (Yuksekgonul et al.,
2023). Users remain responsible for complying with the licenses and terms of
the upstream datasets, Wikidata, Wikipedia, and model providers.

## Compact public data versus raw artifacts

The public `data/model_outputs/` files are sufficient to reproduce the paper.
They contain the model response, correctness, aggregated confidence, question
popularity, ground-truth/generated-entity popularity, both co-occurrence
values, occurrence counts used for filtering, and final SC/VC/LLM-pop scores.
The complete field-level data dictionary, types, missing-value semantics, and
loading example are documented in
[`data/model_outputs/README.md`](../data/model_outputs/README.md).

They omit bulky information that is not consumed by the reported analysis:
full prompts, token IDs, per-token entropy, top-logprob dictionaries, repeated
sampled answers, judge reasoning, and complete Wikipedia indices.

The compact data preserve two intentionally different notions of frequency:

- `qpop`, `gt_pop`, and `gene_pop` are Wikidata sitelink counts and are the
  paper's primary entity-popularity variables. Entity resolution makes them
  cleaner than raw Wikipedia string-occurrence counts.
- `q_occ`, `gt_occ`, and `gene_occ` are Wikipedia document counts. Because
  ambiguous aliases and common words introduce more noise, these fields are
  used only for high-frequency filtering and relation-specificity
  normalization.
- `gt_coo` and `gene_coo` are Wikipedia document-level co-occurrence counts.
  Wikidata cannot measure textual co-exposure, so the corpus statistic is
  required for this pairwise signal.

Run:

```bash
python scripts/materialize_compact_data.py
python scripts/verify_artifacts.py --level full
```

before the analysis. The following raw resources are therefore optional.

## What is not included

| Local path or resource | Typical local size | Why it is excluded |
|---|---:|---|
| `res/` | ~137 MB | Per-example model answers, token probabilities, and external popularity indices |
| `llm_pop_generation/` | ~1.6 GB | Per-example LLM-estimated popularity outputs for multiple shots and models |
| `baselines/self_consistency/` | ~650 MB | Ten-sample generations and judge outputs |
| `baselines/verbalized_confidence/` | ~30 MB | Per-example verbalized-confidence outputs |
| Wikipedia dump and entity-document index | many GB | Upstream corpus and derived document-level index |
| Llama/Qwen model weights | many GB | Distributed by the model providers under their own licenses |
| OpenAI API responses | variable | Generated through a paid third-party API |

These raw artifacts are not part of the public release and are not required to
reproduce the paper's reported results. The compact files under
`data/model_outputs/` contain the response-level fields consumed by the public
analysis pipeline. Model checkpoints and the raw Wikipedia dump must be
obtained directly from their original providers.

## Regenerating the artifacts

Run all commands from the repository root unless a section says otherwise.

### 1. Install the analysis environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Wikipedia/Wikidata preprocessing:

```bash
pip install -r requirements-data.txt
```

For open-source model inference, install a CUDA-compatible version of
`vllm` separately, following the vLLM and PyTorch instructions for your system.

### 2. Obtain model weights

Download the following models from their official provider pages, subject to
their licenses:

- [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [`Qwen/Qwen2-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [`Qwen/Qwen2.5-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

Pass the local model directory to the shell wrappers. Model weights must remain
outside this repository.

### 3. Generate QA responses

```bash
cd qa_generation
bash run_vllm.sh /absolute/path/to/models greedy
```

The expected output is:

```text
res/{dataset}/{dataset}_{model}_temperature1.jsonl
```

For ChatGPT experiments, set `OPENAI_API_KEY` and run the API-based scripts.
API outputs may vary as hosted models change.

### 4. Build popularity statistics

Entity popularity is the Wikidata sitelink count. Relation popularity is the
number of Wikipedia documents containing both entities.

1. Obtain the paper's 41-shard English `20231101.en` configuration from the
   official
   [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
   distribution.
2. Follow [`BUILD_POPULARITY_FEATURES.md`](BUILD_POPULARITY_FEATURES.md) to
   extract the required entities and pairs.
3. Use the parameterized public CLI to collect Wikidata sitelinks, build the
   entity-to-document index, and compute occurrence and co-occurrence counts.

The analysis expects:

```text
res/gt_gene_entity_popularity_*.jsonl
res/cooccurrence_*.json
res/single_occurrence_*.json
```

Record the Wikipedia snapshot date in any reproduction report. Counts can
change between snapshots.

### 5. Regenerate confidence baselines

Self-consistency uses **10 sampled generations per question** in the released
experiment configuration:

```bash
cd baselines
bash run_self_consistency.sh /absolute/path/to/models ..
bash run_verbalized_confidence.sh /absolute/path/to/models
```

ChatGPT sampling can be run separately:

```bash
bash run_chatgpt_sampling.sh 10 gpt-3.5-turbo-1106 5
```

### 6. Regenerate LLM-estimated popularity

```bash
cd pop_generation
python build_clean_data.py
bash run_pop_vllm.sh /absolute/path/to/models ../res ../llm_pop_generation
```

### 7. Run the analyses

Core RQ1/RQ2 statistics:

```bash
python code/analysis_correlation/verify_per_model.py
```

Calibration and transfer experiments:

```bash
python code/analysis_correlation/calibration_experiment.py
python code/analysis_correlation/transfer_all_methods_v3.py
python code/analysis_correlation/setting3_natural_train_test.py
python code/analysis_correlation/transfer_all_methods_setting3.py
```

The compact JSON result files produced in `code/analysis_correlation/` can be
compared with the summaries committed to the public repository.

## Expected directory layout after acquisition

```text
.
├── data/                         # included
├── data/model_outputs/           # included compact reproduction files
├── res/                          # materialized, acquired, or regenerated; ignored
│   ├── movies/
│   ├── songs/
│   └── basketball/
├── llm_pop_generation/           # acquired or regenerated; ignored
│   ├── movies/
│   ├── songs/
│   └── basketball/
└── baselines/
    ├── self_consistency/          # acquired or regenerated; ignored
    └── verbalized_confidence/     # acquired or regenerated; ignored
```

## Integrity and release checks

Before publishing a commit, run:

```bash
python scripts/check_public_release.py --working-tree
```

The check fails when a tracked file is larger than 20 MB, a generated artifact
directory is tracked, or a likely credential file is present.
