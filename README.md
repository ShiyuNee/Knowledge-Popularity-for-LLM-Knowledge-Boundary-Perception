# Popular but Wrong: Understanding and Mitigating LLM Overconfidence through Knowledge Popularity

Official code, compact model outputs, and reproduction instructions for
[**Popular but Wrong: Understanding and Mitigating LLM Overconfidence through
Knowledge Popularity**](https://arxiv.org/abs/2505.17537).

This repository is designed as a reproduction guide. It supports two distinct
goals:

1. **Reproduce the paper's numerical results** from the released response
   snapshots. This is the recommended route and does not require a GPU,
   model weights, paid APIs, or a Wikipedia dump.
2. **Regenerate the pipeline from source data** by rerunning model inference,
   confidence baselines, Wikidata sitelink collection, Wikipedia
   co-occurrence counting, and the final analyses.

The paper studies whether entity popularity and question–answer co-occurrence
provide useful signals for detecting when an LLM is confidently wrong.

## Recommended reproduction route

### 1. Create the environment

Python 3.10 is recommended. The analysis stages run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare the released model outputs

The six compressed files in `data/model_outputs/` already contain every
response-level value consumed by the reported analyses. Materialize the file
layout expected by the original analysis scripts and validate it:

```bash
python scripts/materialize_compact_data.py
python scripts/verify_artifacts.py --level full
```

A successful validation ends with:

```text
Artifact verification passed for level=full.
```

### 3. Run the paper analyses

```bash
python scripts/reproduce.py --stage all
```

The complete CPU run can take tens of minutes. Every command writes an
independent log under `reproduction_logs/`. To run or resume one part only:

```bash
python scripts/reproduce.py --stage core
python scripts/reproduce.py --stage calibration
python scripts/reproduce.py --stage distribution
python scripts/reproduce.py --stage transfer
python scripts/reproduce.py --stage figures
python scripts/reproduce.py --stage check
```

Use `--dry-run` to inspect the commands without executing them.

### 4. Validate the reproduced values

```bash
python scripts/check_expected_results.py
```

The checker validates the balanced calibration results, balanced-to-natural
and natural-to-natural settings, final transfer matrices, and the 1,000-sample
bootstrap configuration. A successful run ends with:

```text
All reproduced values are within the documented tolerances.
```

## Which command reproduces which result?

| Scientific question | Stage | Main output |
|---|---|---|
| How are popularity, confidence, and correctness related? | `core` | terminal/log output and `bootstrap_results.json` |
| Does popularity improve knowledge-boundary detection? | `calibration` | `calibration_results.json` |
| Does the method survive a balanced-to-natural prior shift? | `distribution` | `setting2_results.json` |
| How does it perform under a natural train/test distribution? | `distribution` | `setting3_results.json` |
| Does the predictor transfer across datasets and models? | `transfer` | `transfer_all_methods_v3_results.json` and `transfer_all_methods_setting3_results.json` |
| Can the paper figures be regenerated? | `figures` | `code/analysis_correlation/paper_figures/` |

Internal names such as `setting2` and `setting3` are retained for compatibility
with the released scripts. Their semantic definitions are documented in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Released reproduction data

The repository contains three factual QA datasets:

| Dataset | Relation | Questions |
|---|---|---:|
| Movies | movie → director | 10,964 |
| Songs | song → performer | 2,157 |
| Basketball | player → birthplace | 13,309 |

It also contains one gzip-compressed JSONL file for each evaluated model:

- Llama-3-8B-Instruct
- Qwen2-7B-Instruct
- GPT-3.5-Turbo
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Qwen2.5-32B-Instruct

Each model file contains all 26,430 questions. Important fields are:

| Field group | Fields | Meaning |
|---|---|---|
| QA result | `question`, `reference`, `response`, `correct` | Input, accepted answer, generated answer, and correctness |
| Token confidence | `confidence` | Mean probability of generated-answer tokens |
| Wikidata popularity | `qpop`, `gt_pop`, `gene_pop` | Sitelink counts for question, ground-truth, and generated entities |
| Wikipedia occurrence | `q_occ`, `gt_occ`, `gene_occ` | Documents containing each entity string |
| Wikipedia co-occurrence | `gt_coo`, `gene_coo` | Documents containing both the question and answer entity |
| Confidence baselines | `sc_confidence`, `vc_confidence` | Self-consistency and verbalized confidence |
| LLM popularity estimates | `llm_qpop`, `llm_gene_pop`, `llm_coo` | Zero-shot model estimates on a 1–10 scale |

See [`data/model_outputs/README.md`](data/model_outputs/README.md) for the
complete schema.

### Why Wikidata and Wikipedia are both used

The primary single-entity popularity variables—`qpop`, `gt_pop`, and
`gene_pop`—are Wikidata sitelink counts. They are cleaner than raw Wikipedia
string counts for ambiguous aliases and common words.

The Wikipedia counts `q_occ`, `gt_occ`, and `gene_occ` contain more string-match
noise. They are retained for high-frequency filtering and
relation-specificity normalization, not as the paper's primary entity
popularity measure.

Pairwise `gt_coo` and `gene_coo` must come from Wikipedia because Wikidata
cannot measure how often two entities co-occur in natural-language documents.

## Regenerating the pipeline from scratch

Exact regeneration is more expensive and is not required for numerical
verification of the paper.

### Model responses

Inference code for open-weight models is under `qa_generation/`. Install a
CUDA-compatible PyTorch and vLLM build, then follow
[`docs/REPRODUCE.md`](docs/REPRODUCE.md#5-regenerate-model-outputs-from-scratch).
Hosted-model outputs may change when a provider updates a model or serving
stack; use the released response snapshots when validating paper numbers.

### Wikidata sitelinks and Wikipedia co-occurrence

The parameterized preprocessing CLI can:

1. extract all question, ground-truth, and generated entities;
2. query Wikidata sitelink counts with retries and resume support;
3. scan the English Wikipedia `20231101.en` parquet snapshot;
4. build an entity-to-document index;
5. calculate single-entity occurrence and pairwise co-occurrence; and
6. attach the resulting fields to response rows.

Install the additional dependencies and follow the dedicated guide:

```bash
pip install -r requirements-data.txt
```

See [`docs/BUILD_POPULARITY_FEATURES.md`](docs/BUILD_POPULARITY_FEATURES.md)
for download links, commands, schemas, and matching rules.

### Confidence and LLM-popularity baselines

- `baselines/` contains self-consistency and verbalized-confidence generation.
- `pop_generation/` contains zero-shot LLM popularity estimation.

The expected filenames and commands are documented in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Expected numerical behavior

Deterministic statistical stages and the final distribution/transfer summaries
should match the checked-in values. Neural calibration results can exhibit
small library- or optimization-level variation.

Use [`docs/RESULTS.md`](docs/RESULTS.md) for the complete numerical checkpoints
and evaluation-setting map.

## Large artifacts

Multi-GB prompts, per-token traces, repeated sampling generations, judge
reasoning, Wikipedia dumps, and entity-document indices are intentionally not
stored in Git. They are not consumed by the recommended numerical reproduction
route.

See [`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md) for the exact
artifact policy and the GitHub issue form for requesting redistributable raw
snapshots. Model weights and Wikipedia dumps must be obtained from their
original providers.

## Repository layout

```text
.
├── data/                         # QA datasets and compact response snapshots
├── qa_generation/                # Open-weight and hosted-model inference
├── baselines/                    # Self-consistency and verbalized confidence
├── pop_generation/               # LLM-estimated popularity generation
├── code/
│   ├── analysis_correlation/     # Statistics, calibration, transfer, figures
│   └── prepare_before_analysis/  # Wikidata/Wikipedia feature construction
├── scripts/                      # Materialization, orchestration, validation
├── tests/                        # External-feature pipeline tests
└── docs/                         # Detailed reproduction guides
```

## Detailed documentation

- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): complete step-by-step commands
- [`docs/RESULTS.md`](docs/RESULTS.md): expected values and output mapping
- [`docs/BUILD_POPULARITY_FEATURES.md`](docs/BUILD_POPULARITY_FEATURES.md): external feature construction
- [`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md): included and excluded artifacts
- [`data/README.md`](data/README.md): source QA data
- [`data/model_outputs/README.md`](data/model_outputs/README.md): compact response schema

## Citation

If this repository is useful, please cite the paper:

```bibtex
@article{ni2025popular,
  title={Popular but Wrong: Understanding and Mitigating LLM Overconfidence through Knowledge Popularity},
  author={Ni, Shiyu and Bi, Keping and Guo, Jiafeng and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2505.17537},
  year={2025}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). Model weights,
hosted-model outputs, Wikipedia/Wikidata resources, upstream datasets, and
derived artifacts remain subject to their respective licenses and terms.
