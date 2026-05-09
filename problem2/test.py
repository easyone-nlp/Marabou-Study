#!/usr/bin/env python3
"""Run Marabou on the external TinyBarsNet model.

The query checks local L-infinity robustness around one correctly classified
test image. For each competing class j, Marabou searches for an input x' within
epsilon of x such that output[j] >= output[predicted]. SAT means an adversarial
counterexample was found for that target; UNSAT means that target cannot beat
the predicted class inside the perturbation box.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

candidate_roots = []
if os.environ.get("MARABOU_ROOT"):
    candidate_roots.append(Path(os.environ["MARABOU_ROOT"]).expanduser())
candidate_roots.append(Path(__file__).resolve().parents[1])

for marabou_root in candidate_roots:
    if (marabou_root / "maraboupy" / "Marabou.py").exists():
        sys.path.insert(0, str(marabou_root))
        break

from maraboupy import Marabou


CLASS_NAMES = ["vertical_bar", "horizontal_bar", "diagonal_bar"]


def solve_target(nnet_path: Path, sample: np.ndarray, true_class: int, target_class: int, epsilon: float, timeout: int) -> dict:
    network = Marabou.read_nnet(str(nnet_path))
    input_vars = network.inputVars[0][0].flatten()
    output_vars = network.outputVars[0][0].flatten()

    for variable, value in zip(input_vars, sample):
        network.setLowerBound(int(variable), max(0.0, float(value) - epsilon))
        network.setUpperBound(int(variable), min(1.0, float(value) + epsilon))

    # Counterexample condition: target_class score is at least true_class score.
    network.addInequality(
        [int(output_vars[true_class]), int(output_vars[target_class])],
        [1.0, -1.0],
        0.0,
    )

    options = Marabou.createOptions(timeoutInSeconds=timeout, verbosity=0)
    start = time.perf_counter()
    exit_code, values, stats = network.solve(options=options)
    elapsed = time.perf_counter() - start

    result = "SAT" if exit_code == "sat" else "UNSAT" if exit_code == "unsat" else str(exit_code).upper()
    counterexample = None
    if result == "SAT":
        adversarial_input = np.array([values[int(v)] for v in input_vars], dtype=float)
        outputs = network.evaluateWithoutMarabou(adversarial_input)[0].flatten()
        counterexample = {
            "target_class": target_class,
            "target_name": CLASS_NAMES[target_class],
            "linf_distance": float(np.max(np.abs(adversarial_input - sample))),
            "predicted_from_counterexample": int(np.argmax(outputs)),
            "outputs": outputs.astype(float).tolist(),
            "input_first_16_values": adversarial_input[:16].tolist(),
        }

    return {
        "target_class": target_class,
        "target_name": CLASS_NAMES[target_class],
        "result": result,
        "runtime_seconds": elapsed,
        "counterexample": counterexample,
    }


def run_verification(artifact_dir: Path, epsilon: float, timeout: int) -> dict:
    nnet_path = artifact_dir / "tiny_bars.nnet"
    sample_path = artifact_dir / "sample.npy"
    metadata_path = artifact_dir / "metadata.json"

    if not nnet_path.exists() or not sample_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Missing artifacts. Run `python train_model.py` in problem2 first."
        )

    sample = np.load(sample_path).astype(float)
    metadata = json.loads(metadata_path.read_text())
    true_class = int(metadata["sample_prediction"])

    per_target = []
    for target_class in range(len(CLASS_NAMES)):
        if target_class == true_class:
            continue
        per_target.append(
            solve_target(nnet_path, sample, true_class, target_class, epsilon, timeout)
        )

    overall = "SAT" if any(item["result"] == "SAT" for item in per_target) else "UNSAT"
    return {
        "model": str(nnet_path),
        "epsilon": epsilon,
        "timeout_per_target_seconds": timeout,
        "sample_label": int(metadata["sample_label"]),
        "sample_prediction": true_class,
        "sample_prediction_name": CLASS_NAMES[true_class],
        "interpretation": (
            "SAT means at least one adversarial counterexample exists in the epsilon box; "
            "UNSAT means no checked target class can beat the predicted class in that box."
        ),
        "overall_result": overall,
        "per_target": per_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts", type=Path)
    parser.add_argument("--epsilon", default=0.02, type=float)
    parser.add_argument("--timeout", default=30, type=int)
    args = parser.parse_args()

    result = run_verification(args.artifact_dir, args.epsilon, args.timeout)
    output_path = args.artifact_dir / f"verification_eps_{args.epsilon:g}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
