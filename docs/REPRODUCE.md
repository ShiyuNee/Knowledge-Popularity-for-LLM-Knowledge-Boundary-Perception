# Reproducing the Paper

This guide reproduces the results of [**Popular but Wrong: Understanding and
Mitigating LLM Overconfidence through Knowledge
Popularity**](https://arxiv.org/abs/2505.17537) from the public repository.

There are two supported routes:

1. **Compact reproduction (recommended):** use the six model-level archives
   committed under `data/model_outputs/`. They contain all response-level
   values required by the paper analyses.
2. **From-scratch regeneration:** rerun model inference and rebuild popularity
   features before running the same analysis pipeline.

The compact route recovers the paper numbers. Hosted APIs, model-serving
libraries, and Wikipedia/Wikidata snapshots can change over time, so newly
generated outputs may differ slightly.

## 1. Environment

The analysis pipeline runs on CPU. GPU inference is only required when regenerating outputs for open-weight models.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

The compact no-training path needs only NumPy and SciPy. Before running the
calibration, distribution, transfer, or figure stages, install the full
analysis dependencies:

```bash
pip install -r requirements.txt
```

For dataset construction and Wikipedia processing, also install:

```bash
pip install -r requirements-data.txt
```

For open-weight model inference, install a CUDA-compatible PyTorch build and vLLM separately because the correct versions depend on the host CUDA driver:

```bash
pip install vllm
```

The experiments were developed with Python 3.10. Later Python 3 versions should also work.

## 2. Included public data

The repository includes the three evaluation datasets:

| Dataset | File | Rows |
|---|---|---:|
| Movies | `data/movies.jsonl` | 10,964 |
| Songs | `data/songs.jsonl` | 2,157 |
| Basketball | `data/basketball.jsonl` | 13,309 |

Each JSONL row contains the question, the reference answer, entity metadata, and popularity-related fields that can be redistributed directly. See [`data/README.md`](../data/README.md) for the schema and provenance.

## 3. Prepare the included compact experiment data

The six files under `data/model_outputs/` each combine all three datasets for
one model. Together they are approximately 9 MB compressed. Each response is
already joined with question/ground-truth/generated-entity popularity,
co-occurrence, occurrence-filter values, self-consistency, verbalized
confidence, and LLM-estimated popularity.

Materialize the compatibility layout used by the original analysis scripts:

```bash
python scripts/materialize_compact_data.py
```

This creates:

```text
res/
├── movies/*.jsonl
├── songs/*.jsonl
├── basketball/*.jsonl
├── gt_gene_entity_popularity_*.jsonl
├── cooccurrence_*.json
└── single_occurrence_*.json

baselines/
├── self_consistency/consis_judge_res/*.jsonl
└── verbalized_confidence/*_vc.jsonl

llm_pop_generation/
├── movies/*.jsonl
├── songs/*.jsonl
└── basketball/*.jsonl
```

Validate the materialized data before running any experiment:

```bash
python scripts/verify_artifacts.py --level full
```

The verifier checks all required paths and exact row counts. A successful run
ends with `Artifact verification passed`.

## 4. Reproduce the reported results

The wrapper below runs the original experiment scripts in a documented order and records their stdout/stderr under `reproduction_logs/`.

### 4.1 Primary correlations and discrimination results

```bash
python scripts/reproduce.py --stage core
```

This stage verifies the core artifacts, computes per-model metrics, and runs bootstrap confidence intervals. It covers the main relationship between knowledge popularity, verbalized confidence, and answer correctness.

### 4.2 Calibrated knowledge-boundary prediction

```bash
python scripts/reproduce.py --stage calibration
```

The primary summary is written to `code/analysis_correlation/calibration_results.json`. The main comparison is confidence-only calibration versus calibration augmented with external and LLM-estimated popularity.

### 4.3 Distribution-shift experiments

The paper evaluates three data regimes:

| Semantic name | Training distribution | Test distribution | Internal script name |
|---|---|---|---|
| Balanced evaluation | Balanced correct/incorrect | Balanced correct/incorrect | primary calibration |
| Prior-shift stress test | Balanced correct/incorrect | Natural model distribution | `setting2` |
| Natural deployment distribution | Natural model distribution | Natural model distribution | `setting3` |

Run the two distribution-shift regimes:

```bash
python scripts/reproduce.py --stage distribution
```

Outputs are written to `code/analysis_correlation/setting2_results.json` and `code/analysis_correlation/setting3_results.json`. The legacy filenames are retained only to avoid breaking the original scripts.

### 4.4 Cross-model transfer

```bash
python scripts/reproduce.py --stage transfer
```

This trains a boundary predictor on outputs from one model and evaluates it on another. Summaries are written to `code/analysis_correlation/transfer_all_methods_v3_results.json` and `code/analysis_correlation/transfer_all_methods_setting3_results.json`.

### 4.5 Figures

```bash
python scripts/reproduce.py --stage figures
```

The main generated figures are saved under `code/analysis_correlation/paper_figures/`; additional diagnostic plots are saved beside their plotting scripts.

### 4.6 Numerical checkpoints

Finally, compare the generated summaries with the checked-in expected values:

```bash
python scripts/reproduce.py --stage check
```

or run the checker directly:

```bash
python scripts/check_expected_results.py
```

Expected headline values and their output-file mapping are documented in [`RESULTS.md`](RESULTS.md).

### 4.7 Run the complete analysis pipeline

Once the artifact verifier passes, all analysis stages can be launched with:

```bash
python scripts/reproduce.py --stage all
```

`--stage all` runs compact-data materialization automatically when the legacy
layout is absent.

The complete CPU run can take tens of minutes. Each command streams progress
to the terminal and writes an independent log under `reproduction_logs/`, so a
failed or interrupted stage can be rerun on its own.

Use `--dry-run` to inspect every command without executing it:

```bash
python scripts/reproduce.py --stage all --dry-run
```

## 5. Regenerate model outputs from scratch

This route is substantially more expensive and is not required to verify the paper's analysis.

### 5.1 Open-weight QA inference

Start a vLLM server and run greedy decoding:

```bash
cd qa_generation
bash run_vllm.sh /absolute/path/to/model greedy
cd ..
```

Run this once for each open-weight model. The scripts use `QA_MODEL_PATH` and related environment variables; inspect `qa_generation/run_vllm.sh` before launching on a cluster. Normalize generated files into:

```text
res/<dataset>/<dataset>_<model>_temperature1.jsonl
```

### 5.2 Hosted-model QA inference

Set the provider credential in the environment, never in a source file:

```bash
export OPENAI_API_KEY="..."
```

Then use the generation scripts in `qa_generation/`. Hosted outputs are not guaranteed to match the archived snapshot because provider-side model versions can change.

### 5.3 External popularity features

The external popularity features use:

- Wikidata sitelink counts for global entity popularity;
- Wikipedia entity occurrence counts;
- Wikipedia question-entity/entity co-occurrence counts.

Download the Wikipedia snapshot and run the parameterized entity-extraction,
Wikidata sitelink, Wikipedia indexing, and pairwise-intersection commands in
[`BUILD_POPULARITY_FEATURES.md`](BUILD_POPULARITY_FEATURES.md). The final
merged files must match the three `res/` resource families listed in Section
3. Run:

```bash
python scripts/verify_artifacts.py --level calibration
```

before calibration.

### 5.4 Self-consistency and verbalized confidence

Generate the two baseline families with:

```bash
cd baselines
bash run_self_consistency.sh /absolute/path/to/models
bash run_verbalized_confidence.sh /absolute/path/to/models
cd ..
```

The final files must appear in:

```text
baselines/self_consistency/consis_judge_res/
baselines/verbalized_confidence/
```

### 5.5 LLM-estimated popularity

Use the generation code under `pop_generation/` to create question popularity, answer-entity popularity, and co-occurrence estimates for every dataset/model pair:

```bash
cd pop_generation
python build_clean_data.py
bash run_pop_vllm.sh /absolute/path/to/models ../res ../llm_pop_generation
cd ..
```

The resulting artifacts are stored under `llm_pop_generation/`. The full snapshot contains 54 zero-shot JSONL files:

```text
llm_pop_generation/<dataset>/<dataset>_<model>_<feature>_0.jsonl
```

where `<feature>` is `question_pop`, `gene_pop`, or `coo_pop`.

After all five steps, run:

```bash
python scripts/verify_artifacts.py --level full
python scripts/reproduce.py --stage all
```

## 6. Troubleshooting

- **Missing artifact:** use the exact relative path printed by `verify_artifacts.py`; do not rename individual files after extracting the archive.
- **Different hosted-model numbers:** use the archived artifact snapshot for exact reproduction.
- **CUDA/vLLM errors:** install versions compatible with the machine's NVIDIA driver; analysis-only reproduction does not require vLLM.
- **A long stage fails:** inspect the corresponding file under `reproduction_logs/`, fix the dependency or input, and rerun that stage.
- **Only want to inspect commands:** add `--dry-run`.

The bootstrap stage uses 1,000 resamples and seed 42 by default and writes
`code/analysis_correlation/bootstrap_results.json`. For a quick pipeline smoke
test only, run the underlying script with a smaller value:

```bash
python code/analysis_correlation/bootstrap_ci.py --n-bootstrap 10
```
