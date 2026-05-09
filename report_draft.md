# Assignment 3 Report Draft

## Model and Dataset

For Problem 2, I used a small external dataset generated locally in
`problem2/train_model.py`. The dataset contains synthetic 8x8 grayscale images
with three classes: vertical bar, horizontal bar, and diagonal bar. This dataset
is not part of the official Marabou `resources` directory and does not require
network downloads.

The model is a fully connected ReLU network with architecture `64 -> 8 -> 3`.
It is intentionally small because Marabou can time out on large models. After
training, the model reached 100% accuracy on the generated train and test split.
The trained model was exported to Marabou's `.nnet` format as
`problem2/artifacts/tiny_bars.nnet`.

## Verification Query

I selected one correctly classified test sample predicted as class 0
(`vertical_bar`). The verification query checks local robustness under an
L-infinity perturbation:

`||x' - x||_inf <= epsilon`

All perturbed pixel values are constrained to stay in `[0, 1]`. For each target
class different from the predicted class, Marabou checks whether there exists a
counterexample where the target output is larger than the original predicted
output by the configured margin. If such an input exists, the result is `SAT`;
otherwise the target is verified as impossible within the perturbation box and
the result is `UNSAT`.

## Results

For `epsilon=0.02`, Marabou returned `UNSAT` for both target classes. This means
that, for the selected sample and this small perturbation radius, no checked
target class can beat the original predicted class by the required margin. The
selected input is therefore verified robust within that perturbation box.

For `epsilon=0.3`, Marabou returned `SAT` for both target classes. This larger
perturbation radius allows Marabou to find counterexamples where a target class
exceeds the original predicted class output by the required margin. This shows
that the verification result depends strongly on the perturbation radius.

The saved JSON outputs are:

- `problem2/artifacts/verification_eps_0.02.json`
- `problem2/artifacts/verification_eps_0.3.json`

## Discussion

Marabou is useful because it gives formal SAT/UNSAT answers for a precisely
defined neural-network property. The result is stronger than empirical testing:
`UNSAT` means Marabou proved that no counterexample exists within the specified
input region.

The main limitation I observed is environment and scalability. Large models are
not practical for this assignment, so a small ReLU network was used. I also had
to build Marabou from source in the local environment because the prebuilt
`maraboupy` wheel imported correctly but exited during solving. OpenBLAS CPU
autodetection failed under the QEMU virtual CPU, so OpenBLAS was manually built
with `TARGET=NEHALEM` before building Marabou.
