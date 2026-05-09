# Marabou Study

Assignment 3 repository for exploring Marabou and running a small external
verification example.

## Contents

- `problem1_resources.md`: exploration report for the official Marabou
  `resources/` directory.
- `problem2/`: external model experiment.
  - `train_model.py`: generates a synthetic 8x8 bars dataset, trains a small
    ReLU MLP, and exports it to Marabou `.nnet`.
  - `test.py`: runs Marabou verification for an L-infinity robustness query.
  - `artifacts/tiny_bars.nnet`: exported model used by `test.py`.
  - `artifacts/verification_eps_0.02.json`: UNSAT robustness result.
  - `artifacts/verification_eps_0.3.json`: SAT counterexample result.
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

`problem2/test.py` checks `MARABOU_ROOT` first, then falls back to the installed
`maraboupy` package.

## Run Problem 2

From the repository root:

```bash
cd problem2
python train_model.py
python test.py --epsilon 0.02 --timeout 30
python test.py --epsilon 0.3 --timeout 30
```

Observed results:

- `epsilon=0.02`: `UNSAT`, no checked target class can beat the predicted class
  inside the perturbation box.
- `epsilon=0.3`: `SAT`, Marabou finds counterexamples in the larger
  perturbation box.

## Notes

The official Marabou repository, build directory, downloaded dependencies, and
large generated build artifacts are intentionally not committed here. This
repository contains the assignment code, small model artifact, verification
results, and documentation needed to reproduce the experiment.
