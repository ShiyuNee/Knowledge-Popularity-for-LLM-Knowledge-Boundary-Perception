#!/usr/bin/env python3
"""Compare reproduced compact results with the paper's numerical checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_EXPECTED = {
    "conf_only": 77.08,
    "conf_temp": 77.08,
    "conf_platt": 77.08,
    "conf_isotonic": 77.08,
    "conf_mlp": 77.08,
    "conf+gene_coo_mlp": 82.62,
    "conf+all_ext_mlp": 83.72,
    "conf+llm_coo_pop_mlp": 78.15,
    "conf+llm_all_mlp": 79.02,
}
SETTING3_EXPECTED = {
    "iso": {
        "align": 0.8623,
        "brier": 0.1005,
        "auc": 0.8347,
        "logloss": 0.3351,
    },
    "mlp2": {
        "align": 0.8971,
        "brier": 0.0811,
        "auc": 0.8759,
        "logloss": 0.2807,
    },
}
SETTING2_EXPECTED = {
    "iso": {"align": 0.7658, "brier": 0.1577, "auc": 0.8331, "logloss": 0.5032},
    "mlp2": {"align": 0.8135, "brier": 0.1365, "auc": 0.8741, "logloss": 0.4282},
}
TRANSFER_MLP2_EXPECTED = {
    "setting1": {
        "in_domain": (0.8245, 0.1228, 0.8768, 0.3882),
        "cross_dataset": (0.7460, 0.1803, 0.8448, 0.5559),
        "cross_model": (0.7910, 0.1465, 0.8640, 0.4542),
    },
    "setting2": {
        "in_domain": (0.8174, 0.1366, 0.8754, 0.4279),
        "cross_dataset": (0.8019, 0.1469, 0.8514, 0.4587),
        "cross_model": (0.7885, 0.1422, 0.8640, 0.4387),
    },
    "setting3": {
        "in_domain": (0.8971, 0.0811, 0.8759, 0.2807),
        "cross_dataset": (0.8114, 0.1385, 0.8452, 0.4317),
        "cross_model": (0.8600, 0.1043, 0.8608, 0.3494),
    },
}


def calibration_macro(data: dict, experiment: str) -> float:
    values = []
    for dataset_results in data.values():
        for model_results in dataset_results.values():
            value = model_results.get(experiment, {}).get("alignment")
            if value is not None:
                values.append(float(value))
    if len(values) != 18:
        raise ValueError(f"{experiment}: expected 18 values, found {len(values)}")
    return sum(values) / len(values)


def transfer_macro(data: dict, setting: str, category: str, method: str) -> dict:
    values = [
        value
        for key, value in data[setting][category].items()
        if key.endswith(f"__{method}")
    ]
    expected_count = {"in_domain": 18, "cross_dataset": 36, "cross_model": 90}[category]
    if len(values) != expected_count:
        raise ValueError(
            f"{setting}.{category}.{method}: expected {expected_count} values, "
            f"found {len(values)}"
        )
    return {
        metric: sum(float(value[metric]) for value in values) / len(values)
        for metric in ("align", "brier", "auc", "logloss")
    }


def main() -> int:
    calibration_path = (
        ROOT / "code" / "analysis_correlation" / "calibration_results.json"
    )
    setting3_path = ROOT / "code" / "analysis_correlation" / "setting3_results.json"
    setting2_path = ROOT / "code" / "analysis_correlation" / "setting2_results.json"
    transfer_v3_path = ROOT / "code" / "analysis_correlation" / "transfer_all_methods_v3_results.json"
    transfer_s3_path = ROOT / "code" / "analysis_correlation" / "transfer_all_methods_setting3_results.json"
    required = (calibration_path, setting2_path, setting3_path, transfer_v3_path, transfer_s3_path)
    if not all(path.is_file() for path in required):
        print("Missing reproduced result JSON. Run the calibration and distribution stages.")
        return 1

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    setting2 = json.loads(setting2_path.read_text(encoding="utf-8"))
    setting3 = json.loads(setting3_path.read_text(encoding="utf-8"))
    transfer_v3 = json.loads(transfer_v3_path.read_text(encoding="utf-8"))
    transfer_s3 = json.loads(transfer_s3_path.read_text(encoding="utf-8"))
    failures = []

    print("Calibration Alignment checkpoints (18-pair macro average):")
    for experiment, expected in CALIBRATION_EXPECTED.items():
        actual = calibration_macro(calibration, experiment)
        tolerance = 0.05 if experiment in {
            "conf_only",
            "conf_temp",
            "conf_platt",
            "conf_isotonic",
        } else 1.0
        print(f"  {experiment:<28} actual={actual:6.2f} expected≈{expected:6.2f}")
        if abs(actual - expected) > tolerance:
            failures.append(
                f"{experiment}: {actual:.2f} differs from {expected:.2f} "
                f"by more than {tolerance:.2f}"
            )

    print("\nNatural-train/natural-test checkpoints:")
    aggregate = setting3["aggregate"]
    for method, metrics in SETTING3_EXPECTED.items():
        for metric, expected in metrics.items():
            actual = float(aggregate[method][metric])
            tolerance = 0.015 if method.startswith("mlp") else 0.005
            print(
                f"  {method}.{metric:<10} actual={actual:.4f} expected≈{expected:.4f}"
            )
            if abs(actual - expected) > tolerance:
                failures.append(
                    f"{method}.{metric}: {actual:.4f} differs from "
                    f"{expected:.4f} by more than {tolerance:.4f}"
                )

    print("\nBalanced-train/natural-test checkpoints:")
    for method, metrics in SETTING2_EXPECTED.items():
        for metric, expected in metrics.items():
            actual = float(setting2[method][metric])
            print(f"  {method}.{metric:<10} actual={actual:.4f} expected≈{expected:.4f}")
            if abs(actual - expected) > 0.005:
                failures.append(f"setting2 {method}.{metric}: {actual:.4f} vs {expected:.4f}")

    print("\nTransfer checkpoints (MLP with confidence + generated-answer co-occurrence):")
    transfer_by_setting = {
        "setting1": transfer_v3,
        "setting2": transfer_v3,
        "setting3": transfer_s3,
    }
    metrics_order = ("align", "brier", "auc", "logloss")
    for setting, categories in TRANSFER_MLP2_EXPECTED.items():
        for category, expected_values in categories.items():
            actual_values = transfer_macro(
                transfer_by_setting[setting], setting, category, "mlp2"
            )
            summary = " ".join(
                f"{metric}={actual_values[metric]:.4f}"
                for metric in metrics_order
            )
            print(f"  {setting}.{category:<13} {summary}")
            for metric, expected in zip(metrics_order, expected_values):
                actual = actual_values[metric]
                if abs(actual - expected) > 0.002:
                    failures.append(
                        f"{setting}.{category}.{metric}: {actual:.4f} vs {expected:.4f}"
                    )

    bootstrap_path = ROOT / "code" / "analysis_correlation" / "bootstrap_results.json"
    if bootstrap_path.is_file():
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        if bootstrap.get("n_bootstrap") != 1000 or bootstrap.get("seed") != 42:
            failures.append("bootstrap_results.json was not generated with 1000 resamples and seed 42")
        else:
            print("\nBootstrap checkpoint: 1000 resamples, seed=42.")

    if failures:
        print("\nResult check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll reproduced values are within the documented tolerances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
