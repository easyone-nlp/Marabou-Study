# Marabou Study

Assignment 3 repository for exploring Marabou and running a small external
verification example.

## Contents

- `problem1_resources.md`: exploration report for the official Marabou
  `resources/` directory.
- `problem2/`: external model experiment.
  - `train_model.py`: loads CIFAR-10 from a local torchvision cache,
    preprocesses images to 8x8 grayscale, trains a small ReLU MLP, and exports
    it to Marabou `.nnet`.
  - `verify_marabou.py`: implements the Marabou L-infinity robustness query.
  - `artifacts/tiny_cifar_mlp.nnet`: exported model used by the verification script.
  - `artifacts/sample_preview.png`: selected CIFAR-10 sample after preprocessing.
  - `artifacts/verification_eps_0.02.json`: UNSAT robustness result.
  - `artifacts/verification_eps_0.2.json`: UNSAT robustness result for a larger
    perturbation box.
- `test.py`: repository-level entry point that calls the Problem 2 verifier.
- `requirements.txt`: Python dependencies for the assignment scripts.

## Environment

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Marabou itself can be installed through pip:

```bash
python -m pip install maraboupy==2.0.0
```

On the local assignment machine, the prebuilt wheel imported correctly but
exited during `solve()`. The working run therefore used a source build of
Marabou and set `MARABOU_ROOT` to that clone:

```bash
export MARABOU_ROOT=/path/to/Marabou
```

`problem2/verify_marabou.py` checks `MARABOU_ROOT` first, then falls back to the
installed `maraboupy` package.

## Run Problem 2

From the repository root:

```bash
python test.py --epsilon 0.02 --timeout 30
python test.py --epsilon 0.2 --timeout 30
```

The committed `problem2/artifacts/` files are enough to run verification.
Retraining with `python problem2/train_model.py` requires a local CIFAR-10 cache
or a `--data-root` pointing to `cifar-10-batches-py`.

Observed results:

- `epsilon=0.02`: `UNSAT`, no checked target class can beat the predicted class
  by the required margin inside the perturbation box.
- `epsilon=0.2`: `UNSAT`, the selected high-margin CIFAR-10 sample remains
  verified against all target classes for this tested radius.

## Notes

The official Marabou repository, build directory, downloaded dependencies, and
large generated build artifacts are intentionally not committed here. This
repository contains the assignment code, small model artifact, verification
results, and documentation needed to reproduce the experiment.
