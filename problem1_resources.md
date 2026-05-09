# Problem 1: Explore the Marabou Resources Directory

Working directory: `/home/dilab/Desktop/Jiwon/ResNet-18/Marabou`

Repository checked: `https://github.com/NeuralNetworkVerification/Marabou`

Local commit used for this summary: `52f74072`

## Overview

The Marabou `resources` directory contains sample neural network models, benchmark properties, input-query files, and helper scripts for running verification examples. In the local checkout, `resources` contains 875 files.

Main file types:

| File type | Count | Purpose |
| --- | ---: | --- |
| `.nnet` | 634 | Neural-network benchmarks, especially ACAS Xu, MNIST, CollisionAvoidance, and TwinStreams |
| `.onnx` | 119 | ONNX neural-network models and operator/layer tests |
| `.txt` | 87 | Marabou property files, mostly input/output constraints |
| `.vnnlib` | 15 | VNN-COMP style property/query specifications |
| `.h5` | 4 | Keras models |
| `.ipq` | 2 | Serialized Marabou input-query files |
| `.mps` | 2 | Linear-programming examples |
| `.py` | 3 | Helper scripts for running or generating queries |

## Model Types Provided

Marabou provides examples in several model formats.

### `.nnet`

Directory: `resources/nnet`

The top-level `resources/README.md` describes the `.nnet` benchmark categories:

| Directory | Model category |
| --- | --- |
| `resources/nnet/acasxu` | ACAS Xu aircraft collision-avoidance networks from the Reluplex benchmarks |
| `resources/nnet/coav` | CollisionAvoidance networks from the Planet paper |
| `resources/nnet/twin` | TwinStreams benchmarks from the Planet repository |
| `resources/nnet/mnist` | MNIST fully connected benchmark networks |

Example files:

- `resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet`
- `resources/nnet/mnist/mnist10x20.nnet`
- `resources/nnet/fc_2-2-3.nnet`

### `.onnx`

Directory: `resources/onnx`

The ONNX resources include converted benchmark networks and smaller operator/layer tests.

Important groups:

- `resources/onnx/acasxu`: ONNX versions of ACAS Xu models. The README says these were taken from the VNN-COMP 2021 ACAS Xu benchmark repository and renamed to match the `.nnet` model names.
- `resources/onnx/cifar10`: CIFAR-10 models evaluated in an OpenReview paper. The README notes that `_simp` models were simplified with `onnx-simplifier`.
- `resources/onnx/layer-zoo`: small ONNX models for individual supported operations such as `relu`, `gemm`, `conv`, `maxpool`, `batchnorm`, `flatten`, `reshape`, `sigmoid`, `tanh`, and `leakyRelu`.
- `resources/onnx/vnnlib`: small ONNX models paired with VNNLIB property files for query testing.

Example files:

- `resources/onnx/acasxu/ACASXU_experimental_v2a_1_1.onnx`
- `resources/onnx/cifar10/cifar_base_kw.onnx`
- `resources/onnx/mnist2x10.onnx`
- `resources/onnx/model-german-traffic-sign-fast.onnx`
- `resources/onnx/traffic-classifier64.onnx`

### `.h5`

Directory: `resources/keras`

These are Keras model files. Examples include:

- `resources/keras/cnn_max_mnist2.h5`
- `resources/keras/cnn_max_mnist3.h5`
- `resources/keras/robust_mnist_sigmoid_linear.h5`
- `resources/keras/fc_2-2sigmoids-3.h5`

### Other formats

- `.ipq`: Marabou input-query files in `resources/target`, for example `mnist-bnn_index2_eps0.001_target9_unsat.ipq`.
- `.mps`: linear-programming examples in `resources/mps`, including feasible and infeasible LP instances.
- BNN query examples: `resources/bnn_queries/smallBNN_original` and `resources/bnn_queries/smallBNN_parsed`.

The helper script `resources/runMarabou.py` accepts `.nnet`, `.pb`, and `.onnx` network files. This means the resources directory mostly demonstrates `.nnet` and `.onnx`, while the script also supports TensorFlow `.pb` models if supplied.

## Datasets and Input Specifications

The resources directory includes benchmark-specific properties and scripts rather than full training datasets.

### ACAS Xu properties

Directory: `resources/properties`

Files:

- `acas_property_1.txt`
- `acas_property_2.txt`
- `acas_property_3.txt`
- `acas_property_4.txt`

These define input bounds and output constraints over variables such as `x0`, `x1`, and `y0`. For example, `acas_property_1.txt` constrains five ACAS inputs and checks an output lower bound.

### CollisionAvoidance and TwinStreams property

File:

- `resources/properties/builtin_property.txt`

The top-level resources README says this is used for CollisionAvoidance and TwinStreams benchmarks because the property is built into the final network layer. The file checks `y0 <= 0`.

### MNIST targeted-attack properties

Directory:

- `resources/properties/mnist`

This directory contains many files named like:

- `image1_target2_epsilon0.005.txt`
- `image2_target9_epsilon0.05.txt`
- `image3_target4_epsilon0.1.txt`

The names encode the image index, target class, and allowed perturbation radius. The files define bounds for flattened MNIST input variables, such as `x0 >= -0.005` and `x0 <= 0.005`.

### Dataset-based query generation

The script `resources/runMarabou.py` can generate L-infinity robustness queries for:

- `mnist`
- `cifar10`

The script loads MNIST through `tensorflow.keras.datasets.mnist` and CIFAR-10 through `torchvision.datasets.CIFAR10`. It uses:

- `--dataset` to choose `mnist` or `cifar10`
- `--epsilon` for the L-infinity perturbation bound
- `--target-label` for targeted adversarial checks
- `--index` for the selected test-set point

## Example Verification Queries

The directory includes several query styles.

### Network plus property file

Examples:

- ACAS Xu `.nnet` model with an ACAS property:
  - `resources/nnet/acasxu/ACASXU_experimental_v2a_1_1.nnet`
  - `resources/properties/acas_property_1.txt`
- MNIST `.nnet` model with a targeted-attack property:
  - `resources/nnet/mnist/mnist10x20.nnet`
  - `resources/properties/mnist/image1_target2_epsilon0.005.txt`

### VNNLIB queries

Directory:

- `resources/onnx/vnnlib`

Examples:

- `acasxu_prop1.vnnlib`
- `test_tiny_vnncomp.vnnlib`
- `test_small_vnncomp.vnnlib`
- `test_nano_vnncomp.vnnlib`
- `test_add_var.vnnlib`
- `test_mul_var_const.vnnlib`
- `test_sub_var.vnnlib`
- `test_and.vnnlib`

These are SMT-LIB-like VNNLIB property specifications. For example, `test_tiny_vnncomp.vnnlib` declares real-valued input/output variables and asserts a constraint over them.

### Saved input-query files

Directory:

- `resources/target`

Example:

- `mnist-bnn_index2_eps0.001_target9_unsat.ipq`

The README describes this folder as containing challenging queries.

### BNN query files

Directory:

- `resources/bnn_queries`

Examples:

- `smallBNN_original`
- `smallBNN_parsed`

These appear to be binary neural-network query examples in original and parsed forms.

### LP examples

Directory:

- `resources/mps`

Examples:

- `lp_feasible_1.mps`
- `lp_infeasible_1.mps`

These are not neural-network models, but they are useful solver input examples for feasible and infeasible linear programs.

## Notes for Selecting a Model for Problem 2

For a follow-up verification task, the easiest choices are likely:

1. Use an existing `.nnet` ACAS Xu or MNIST model with a matching `.txt` property, because these are already paired with Marabou-style property files.
2. Use an ONNX model with a VNNLIB property from `resources/onnx/vnnlib`, because this follows common VNN-COMP conventions.
3. Use `resources/runMarabou.py` to generate an MNIST or CIFAR-10 L-infinity robustness query from a model, dataset name, epsilon, target label, and test index.

The supported model formats visible from `resources/runMarabou.py` are `.nnet`, `.onnx`, and `.pb`; the resources directory itself mainly provides `.nnet`, `.onnx`, and `.h5` examples.
