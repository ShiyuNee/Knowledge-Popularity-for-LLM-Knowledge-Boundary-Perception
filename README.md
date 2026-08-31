# Popular but Wrong: Understanding and Mitigating LLM Overconfidence through Knowledge Popularity

Official code, compact model outputs, and reproduction instructions for
EMNLP2026: [**Popular but Wrong: Understanding and Mitigating LLM Overconfidence through
Knowledge Popularity**](https://arxiv.org/abs/2505.17537).

The recommended workflow uses the released response snapshots. It reproduces
the paper's main statistics on CPU without model weights, paid APIs, a GPU, or
a Wikipedia dump.

## Choose a reproduction path

| Goal | Requires training? | Hardware | Start here |
|---|---:|---|---|
| Verify the data and reproduce the paper's main statistical findings | No | CPU | [Quick start](#quick-start-no-training) |
| Rerun calibration, distribution-shift, and transfer experiments | Yes | CPU + PyTorch | [Full analysis](#full-analysis-including-training) |
| Regenerate model responses and popularity features | Yes | GPU and external data/services | [From-scratch regeneration](#from-scratch-regeneration) |

## Quick start: no training

Python 3.10 is recommended. From a fresh clone:

```bash
git clone https://github.com/ShiyuNee/Knowledge-Popularity-for-LLM-Knowledge-Boundary-Perception.git
cd Knowledge-Popularity-for-LLM-Knowledge-Boundary-Perception

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

Materialize the released compact snapshots and validate every required path
and row count:

```bash
python scripts/materialize_compact_data.py
python scripts/verify_artifacts.py --level full
```

A successful validation ends with:

```text
Artifact verification passed for level=full.
```

Run the non-training analyses used for the paper's main findings:

```bash
python scripts/reproduce.py --stage core
```

This computes the per-model popularity/correctness/confidence statistics and
1,000-sample bootstrap confidence intervals. Results are printed to the
terminal, written to `reproduction_logs/`, and summarized in
`code/analysis_correlation/bootstrap_results.json`.

Finally, verify that the checked-in calibration and transfer summaries are
within the paper's documented tolerances:

```bash
python scripts/check_expected_results.py
```

This last command validates released result files; it does not retrain the
calibration models. A successful run ends with:

```text
All reproduced values are within the documented tolerances.
```

The materialization step creates compatibility directories such as `res/`,
`llm_pop_generation/`, and generated baseline files inside the clone. Reuse
the same clone for later stages; rerunning materialization is safe.

## Full analysis: including training

Install the full analysis dependencies:

```bash
pip install -r requirements.txt
```

Then run only the stage you need:

```bash
python scripts/reproduce.py --stage calibration   # Table 5 calibration results
python scripts/reproduce.py --stage distribution  # balanced/natural settings
python scripts/reproduce.py --stage transfer      # cross-dataset/model transfer
python scripts/reproduce.py --stage figures       # paper figures
python scripts/reproduce.py --stage check         # numerical checkpoints
```

To run every stage, including training:

```bash
python scripts/reproduce.py --stage all
```

The complete CPU run can take tens of minutes or longer depending on hardware.
Each command writes an independent log under `reproduction_logs/`, so failed or
interrupted stages can be resumed individually. Use `--dry-run` to inspect the
commands first.

### Expected numerical variation

Deterministic statistics should match the released values. MLP-based results
may vary slightly with the PyTorch version, hardware, and random initialization.
For reference, the released snapshot and arXiv v2 differ only slightly:

| Method | arXiv v2 | Released snapshot |
|---|---:|---:|
| Confidence + generated-answer co-occurrence | 82.62 | 82.78 |
| Confidence + all external popularity features | 83.72 | 83.84 |
| Confidence + LLM-estimated co-occurrence | 78.15 | 78.01 |
| Confidence + all LLM-estimated popularity features | 79.02 | 79.08 |

These differences are below 0.2 percentage points and do not change the
paper's conclusions or method ranking. The automated checker uses explicit
tolerances for stochastic results.

## Command-to-result map

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

The repository directly includes two analysis-ready data layers:

1. three source factual-QA files under `data/`; and
2. six compact model-output archives under `data/model_outputs/`.

No additional download is needed for the recommended reproduction workflow.

### Source QA datasets

| File | Relation | Rows |
|---|---|---:|
| `data/movies.jsonl` | movie → director | 10,964 |
| `data/songs.jsonl` | song → performer | 2,157 |
| `data/basketball.jsonl` | basketball player → birthplace | 13,309 |

Each line is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `question` | string | Natural-language factual question. |
| `reference` | array of strings | One or more accepted answer aliases. |
| `popularity` | integer | Wikidata sitelink count for the question entity. |
| `question_entity` | string | Subject entity mentioned in the question. |

### Compact model-output data

The repository contains one gzip-compressed JSONL file for each evaluated
model. Every archive has 26,430 rows and combines all three QA datasets.

| File | Model | Rows |
|---|---|---:|
| `data/model_outputs/llama3-8b.jsonl.gz` | Llama-3-8B-Instruct | 26,430 |
| `data/model_outputs/qwen2-7b.jsonl.gz` | Qwen2-7B-Instruct | 26,430 |
| `data/model_outputs/gpt-3.5-turbo.jsonl.gz` | GPT-3.5-Turbo | 26,430 |
| `data/model_outputs/qwen2.5-7b.jsonl.gz` | Qwen2.5-7B-Instruct | 26,430 |
| `data/model_outputs/qwen2.5-14b.jsonl.gz` | Qwen2.5-14B-Instruct | 26,430 |
| `data/model_outputs/qwen2.5-32b.jsonl.gz` | Qwen2.5-32B-Instruct | 26,430 |

Each line is one JSON object with the following complete schema:

| Field | Type / range | Description |
|---|---|---|
| `dataset` | string | Source dataset: `movies`, `songs`, or `basketball`. |
| `index` | integer | Zero-based row index within the source dataset. `(dataset, index)` identifies the same question across model archives. |
| `question` | string | Factual QA prompt. |
| `reference` | array of strings | Accepted ground-truth answer aliases. |
| `response` | string | Archived model-generated answer. |
| `correct` | integer, `0` or `1` | Whether a normalized reference answer occurs as a contiguous token sequence in the response. |
| `confidence` | number in `[0, 1]`, or `null` | Mean probability of the generated answer tokens. |
| `qpop` | integer, `>= 0` | Wikidata sitelink count for the question entity. |
| `gt_pop` | integer, `>= 0` | Wikidata sitelink count for the first ground-truth answer entity. |
| `gene_pop` | integer, `>= 0` | Wikidata sitelink count for the cleaned generated-answer entity. |
| `gt_pop_found` | boolean | Whether the ground-truth entity was found in the popularity resource. |
| `gene_pop_found` | boolean | Whether the generated entity was found in the popularity resource. |
| `q_occ` | integer `>= 0`, or `null` | Number of Wikipedia documents containing the question entity. |
| `gt_occ` | integer `>= 0`, or `null` | Number of Wikipedia documents containing the ground-truth entity. |
| `gene_occ` | integer `>= 0`, or `null` | Number of Wikipedia documents containing the generated entity. |
| `gt_coo` | integer `>= 0`, or `null` | Wikipedia documents containing both the question and ground-truth entities. |
| `gene_coo` | integer `>= 0`, or `null` | Wikipedia documents containing both the question and generated entities. |
| `sc_confidence` | number in `[0, 1]`, or `null` | Self-consistency confidence from repeated generations. |
| `vc_confidence` | number in `[0, 1]`, or `null` | Verbalized-confidence score. |
| `llm_qpop` | integer from `1` to `10`, or `null` | Zero-shot LLM estimate of question-entity popularity. |
| `llm_gene_pop` | integer from `1` to `10`, or `null` | Zero-shot LLM estimate of generated-answer popularity. |
| `llm_coo` | integer from `1` to `10`, or `null` | Zero-shot LLM estimate of question/answer relation popularity. |

`null` means that an upstream value was unavailable or could not be parsed; it
does not mean zero. When `gt_pop_found` or `gene_pop_found` is `false`, the
corresponding compact popularity value is stored as `0`, so use the flag to
distinguish a missing entity from a genuine zero count.

The `correct` field uses normalized containment rather than strict string
equality. Matching is case-insensitive and normalizes punctuation, articles,
whitespace, and Unicode representation, while preserving diacritics.

See [`data/model_outputs/README.md`](data/model_outputs/README.md) for the
same schema with an example row and loading code, and [`data/README.md`](data/README.md)
for the source QA format.

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

## From-scratch regeneration

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

## Troubleshooting and detailed documentation

- If validation reports missing files, run
  `python scripts/materialize_compact_data.py` before any analysis stage.
- The no-training path needs only `requirements-core.txt`; PyTorch is required
  only for calibration, distribution, and transfer stages.
- Hosted-model outputs may change when providers update a model or serving
  stack. Use the released snapshots when comparing against paper values.
- Exact commands, schemas, expected outputs, and data provenance are documented
  below.

## Large artifacts

Multi-GB prompts, per-token traces, repeated sampling generations, judge
reasoning, Wikipedia dumps, and entity-document indices are intentionally not
stored in Git. They are not consumed by the recommended numerical reproduction
route.

See [`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md) for the exact
artifact policy and regeneration instructions. Model weights and Wikipedia
dumps must be obtained from their original providers.

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
@article{ni2025knowledge,
  title={How knowledge popularity influences and enhances llm knowledge boundary perception},
  author={Ni, Shiyu and Bi, Keping and Guo, Jiafeng and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2505.17537},
  year={2025}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). Model weights,
hosted-model outputs, Wikipedia/Wikidata resources, upstream datasets, and
derived artifacts remain subject to their respective licenses and terms.
