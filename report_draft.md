# Assignment 3 Report Draft

## Model and Dataset

For Problem 2, I used EMNIST Digits as the external dataset. EMNIST is not part
of the official Marabou `resources` directory. The script loads it with
`torchvision.datasets.EMNIST`, then preprocesses each 28x28 grayscale image
into a 14x14 input by average pooling.

The model is a fully connected ReLU network with architecture `196 -> 32 -> 10`.
It is intentionally small because Marabou can time out on large models. The
trained model reached 99.2% accuracy on the balanced training subset and 95.65%
accuracy on the balanced test subset. The trained model was exported to
Marabou's `.nnet` format as `problem2/artifacts/tiny_emnist_mlp.nnet`.

## Verification Query

I selected one correctly classified EMNIST test sample predicted as digit `0`.
The verification query checks local robustness under an L-infinity perturbation:

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

For `epsilon=0.2`, Marabou returned `SAT` for target digit `8`. The
counterexample stays within the larger perturbation box, but changes the output
ordering so that digit `8` exceeds the original digit `0` score by the required
margin. This shows that the verified property depends strongly on the chosen
perturbation radius.

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
compressed EMNIST inputs to 196 features. I also had to build Marabou from
source in the local environment because the prebuilt
`maraboupy` wheel imported correctly but exited during solving. OpenBLAS CPU
autodetection failed under the QEMU virtual CPU, so OpenBLAS was manually built
with `TARGET=NEHALEM` before building Marabou.
