# Assignment 3 Problem 2

This directory contains a small EMNIST Digits model and a Marabou verification script.

## Model and Dataset

- Dataset: EMNIST Digits loaded through `torchvision.datasets.EMNIST`
- Preprocessing: grayscale 28x28 image -> average pool to 14x14 -> 196
  inputs in `[0, 1]`
- Classes: digits `0` through `9`
- Model: fully connected ReLU network, `196 -> 32 -> 10`
- Marabou model format: `.nnet`

The EMNIST files are external to Marabou and are not part of Marabou's
`resources` directory. The script uses `download=False` by default, so the
dataset must already exist in the local torchvision cache, be supplied with
`--data-root`, or be fetched once with `--download`.

## Environment

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

On the assignment machine used for this run, the prebuilt `maraboupy==2.0.0`
wheel could be imported, but the solver exited during `solve()`. To make the
run reproducible in that environment, Marabou was built from the cloned source
tree. The `verify_marabou.py` script checks `MARABOU_ROOT`; if it points to a
Marabou source checkout with a built `maraboupy/MarabouCore...so`, that local
build is used before falling back to the installed `maraboupy` package.

Generic source-build flow:

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd Marabou
python -m pip install cmake maraboupy==2.0.0
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j 4
export MARABOU_ROOT=/path/to/Marabou
```

Note: OpenBLAS CPU autodetection failed under the QEMU virtual CPU, so OpenBLAS
was manually built once with `TARGET=NEHALEM` before rerunning CMake.

## Run

Train the model and export it to `.nnet`:

```bash
python problem2/train_model.py
```

This retraining step requires EMNIST to already be available locally because
the script uses `download=False` by default. Run once with `--download` if the
cache is missing. The committed artifacts are enough to run `test.py` without
retraining.

Run Marabou verification:

```bash
python test.py --epsilon 0.02 --timeout 30
```

The repository-level `test.py` is the entry point. It calls
`problem2/verify_marabou.py`, which contains the actual Marabou query.

Observed results:

- `epsilon=0.02`: `UNSAT` for all nine non-predicted target classes. The selected
  EMNIST digit `0` sample is verified against all other digits within this
  perturbation box.
- `epsilon=0.2`: `SAT` for target digit `8`. Marabou finds an input in the
  larger perturbation box where digit `8` exceeds the original digit `0` score
  by the configured margin.

## Query

The script chooses one correctly classified test image. For each target class
different from the predicted class, it asks Marabou whether there exists an
input `x'` such that:

- `||x' - x||_inf <= epsilon`
- all pixels remain in `[0, 1]`
- the target class score is at least `margin` larger than the predicted class
  score

Interpretation:

- `UNSAT`: no adversarial example was found for the checked target, so the
  predicted class is verified against that target within the epsilon box.
- `SAT`: Marabou found a counterexample in the epsilon box.

The default margin is `0.05`. This avoids treating near-ties between class
outputs as meaningful counterexamples.

The JSON result is saved to `artifacts/verification_eps_<epsilon>.json`.
