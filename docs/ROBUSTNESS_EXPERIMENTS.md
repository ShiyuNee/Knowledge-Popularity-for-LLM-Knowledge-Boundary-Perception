# Additional Calibration, Distribution, and Transfer Experiments

This document maps the additional analyses developed during review to the
public, reproducibility-first workflow. These experiments are organized by
scientific purpose rather than by review chronology.

## Coverage matrix

| Question | Methods/settings | Public command | Output |
|---|---|---|---|
| Does popularity help beyond MLP capacity? | Confidence threshold, 1D MLP, 2D popularity MLP | `python scripts/reproduce.py --stage calibration` | `code/analysis_correlation/calibration_results.json` |
| How do standard calibration baselines compare? | Temperature, Platt, isotonic, MLP | `--stage calibration` | same file |
| Does ECE hide discrimination quality? | ECE, Brier, AUROC, log loss, separation | `--stage calibration` | same file |
| What happens under class-prior shift? | Balanced train, natural test | `python scripts/reproduce.py --stage distribution` | `setting2_results.json` |
| What happens in a deployment-like distribution? | Natural train, natural test | `--stage distribution` | `setting3_results.json` |
| Does the predictor transfer? | In-domain, cross-dataset/relation, cross-model | `python scripts/reproduce.py --stage transfer` | `transfer_all_methods_v3_results.json`, `transfer_all_methods_setting3_results.json` |
| Can the original early transfer table be regenerated? | Historical 2D MLP and cross-relation baselines | `python scripts/reproduce.py --stage historical` | `transfer_results.json`, `transfer_cross_relation_all_methods_results.json` |

## Evaluation settings

| Setting | Training distribution | Test distribution | Purpose |
|---|---|---|---|
| Setting 1 | Balanced | Balanced | Controlled discrimination comparison |
| Setting 2 | Balanced | Natural | Class-prior/distribution-shift stress test |
| Setting 3 | Natural | Natural | Deployment-like calibration setting |

All final experiments use three datasets, six models, a 50:50 stratified split,
and seed 42. Transfer summaries use macro averages over 18 in-domain, 36
cross-dataset, or 90 cross-model source-target pairs.

## Training methods

- Temperature scaling: grid search over the documented temperature grid.
- Platt scaling: logistic regression with `random_state=42`.
- Isotonic regression: monotonic non-parametric calibration.
- MLP (confidence only): 128/64/32 hidden units with BatchNorm and dropout.
- MLP (confidence + generated-answer co-occurrence): the same MLP with the
  Wikipedia `gene_coo` feature added.

The MLP uses Adam, learning rate `5e-4`, weight decay `1e-4`, batch size 256,
up to 100 epochs, and validation-loss early stopping with patience 10.

## Machine-checkable headline results

Run:

```bash
python scripts/check_expected_results.py
```

The checker validates:

- the balanced calibration table;
- Setting 2 and Setting 3 alignment, Brier, AUROC, and log loss;
- MLP-2D in-domain, cross-dataset, and cross-model results for all three
  settings;
- the 1,000-resample bootstrap configuration.

The expected transfer macro values are documented in [`RESULTS.md`](RESULTS.md).

## Reported snapshot versus deterministic rerun

The originally reported balanced-calibration snapshot was created before the
global PyTorch seed was explicitly set. It is retained at:

```text
code/analysis_correlation/reported_results/calibration_results_reported.json
```

For the headline popularity-augmented alignment, the reported value is 82.62%
and the deterministic seed-42 rerun is 82.78%, a difference of 0.16 percentage
points. The public checker therefore uses a documented tolerance for this
stochastic MLP experiment. One-dimensional monotonic baselines and the final
consolidated Setting 1/2/3 transfer scripts are deterministic.

The early `transfer_experiment.py` also originally omitted an explicit PyTorch
seed. Its unseeded historical snapshot is retained at:

```text
code/analysis_correlation/reported_results/transfer_results_historical_unseeded.json
```

The script is now seeded and reproducible, but its new deterministic output is
not expected to be byte-identical to that archived historical snapshot. The
paper's final transfer conclusions should be validated with the consolidated
v3 and Setting 3 scripts, which supersede the early table.

## Runtime expectations

All analysis and training stages run on CPU. Runtime depends strongly on CPU
speed and PyTorch threading. The bootstrap, calibration MLP grid, and full
transfer matrices are the expensive stages; a complete run can take tens of
minutes. Run stages separately to resume or inspect logs:

```bash
python scripts/reproduce.py --stage core
python scripts/reproduce.py --stage calibration
python scripts/reproduce.py --stage distribution
python scripts/reproduce.py --stage transfer
python scripts/reproduce.py --stage figures
python scripts/reproduce.py --stage check
```

Every command writes a separate log under `reproduction_logs/`.
