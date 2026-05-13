# Assignment 3 Report Draft

## Model and Dataset

For Problem 2, I used CIFAR-10 as the external dataset. CIFAR-10 is not part of
the official Marabou `resources` directory. The script loads it with
`torchvision.datasets.CIFAR10` using `download=False`, then preprocesses each
RGB 32x32 image into an 8x8 grayscale input by average pooling.

The model is a fully connected ReLU network with architecture `64 -> 32 -> 10`.
It is intentionally small because Marabou can time out on large models. The
trained model reached 33.12% accuracy on the balanced training subset and 28.1%
accuracy on the balanced test subset. The accuracy is modest because the
network sees only a heavily compressed grayscale version of CIFAR-10, but it is
small enough for fast formal verification. The trained model was exported to
Marabou's `.nnet` format as `problem2/artifacts/tiny_cifar_mlp.nnet`.

## Verification Query

I selected one correctly classified CIFAR-10 test sample predicted as class 7
(`horse`). The verification query checks local robustness under an L-infinity
perturbation:

`||x' - x||_inf <= epsilon`

All perturbed pixel values are constrained to stay in `[0, 1]`. For each target
class different from the predicted class, Marabou checks whether there exists a
counterexample where the target output is larger than the original predicted
output by the configured margin. If such an input exists, the result is `SAT`;
otherwise the target is verified as impossible within the perturbation box and
the result is `UNSAT`.

## Results

For `epsilon=0.02`, Marabou returned `UNSAT` for all nine non-predicted target
classes. This means that, for the selected sample and this small perturbation
radius, no checked target class can beat the original predicted class by the
required margin. The selected input is therefore verified robust within that
perturbation box.

For `epsilon=0.2`, Marabou also returned `UNSAT` for all nine target classes.
This does not mean the model is generally robust on CIFAR-10; it only proves
the stated property for the selected preprocessed input, the selected model, and
the specified perturbation box.

The saved JSON outputs are:

- `problem2/artifacts/verification_eps_0.02.json`
- `problem2/artifacts/verification_eps_0.2.json`

## Discussion

Marabou is useful because it gives formal SAT/UNSAT answers for a precisely
defined neural-network property. The result is stronger than empirical testing:
`UNSAT` means Marabou proved that no counterexample exists within the specified
input region.

The main limitation I observed is environment and scalability. Large models are
not practical for this assignment, so I used a very small ReLU network and
compressed CIFAR-10 inputs to 64 features. I also had to build Marabou from
source in the local environment because the prebuilt
`maraboupy` wheel imported correctly but exited during solving. OpenBLAS CPU
autodetection failed under the QEMU virtual CPU, so OpenBLAS was manually built
with `TARGET=NEHALEM` before building Marabou.
