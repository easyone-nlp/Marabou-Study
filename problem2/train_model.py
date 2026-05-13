#!/usr/bin/env python3
"""Train a small CIFAR-10 model and export it to Marabou's .nnet format.

Problem 2 asks for a model and dataset that are not already included in the
Marabou resources directory. This script uses the external CIFAR-10 dataset from
the local torchvision cache, preprocesses each image to 8x8 grayscale, trains a
small fully connected ReLU network, and writes the network in Marabou's .nnet
format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10


INPUT_SIZE = 64
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
SEED = 7
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class TinyCifarMLP(nn.Module):
    """Small fully connected ReLU network: 64 -> 32 -> 10."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def preprocess_cifar_images(images: np.ndarray) -> np.ndarray:
    """Convert CIFAR RGB images to flattened 8x8 grayscale inputs in [0, 1]."""

    rgb = images.astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    tensor = torch.from_numpy(gray).unsqueeze(1)
    pooled = F.avg_pool2d(tensor, kernel_size=4, stride=4)
    return pooled.squeeze(1).reshape(len(images), -1).numpy().astype(np.float32)


def balanced_subset(x: np.ndarray, y: np.ndarray, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for label in range(OUTPUT_SIZE):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        chosen.append(indices[:per_class])
    selected = np.concatenate(chosen)
    rng.shuffle(selected)
    return x[selected], y[selected]


def load_cifar10_dataset(data_root: Path, train_per_class: int, test_per_class: int) -> tuple:
    try:
        train_set = CIFAR10(root=str(data_root), train=True, download=False)
        test_set = CIFAR10(root=str(data_root), train=False, download=False)
    except RuntimeError as exc:
        raise FileNotFoundError(
            "CIFAR-10 was not found in the requested data root. Place the "
            "cifar-10-batches-py directory there or pass --data-root."
        ) from exc

    x_train = preprocess_cifar_images(train_set.data)
    y_train = np.array(train_set.targets, dtype=np.int64)
    x_test = preprocess_cifar_images(test_set.data)
    y_test = np.array(test_set.targets, dtype=np.int64)

    return (
        *balanced_subset(x_train, y_train, train_per_class, SEED),
        *balanced_subset(x_test, y_test, test_per_class, SEED + 1),
    )


def train_model(
    output_dir: Path,
    data_root: Path,
    epochs: int,
    train_per_class: int,
    test_per_class: int,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train, y_train, x_test, y_test = load_cifar10_dataset(
        data_root,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=128,
        shuffle=True,
    )

    model = TinyCifarMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(torch.from_numpy(x_train))
        test_logits = model(torch.from_numpy(x_test))
        train_pred = train_logits.argmax(dim=1).numpy()
        test_pred = test_logits.argmax(dim=1).numpy()

    train_accuracy = float((train_pred == y_train).mean())
    test_accuracy = float((test_pred == y_test).mean())

    correct_indices = np.flatnonzero(test_pred == y_test)
    if len(correct_indices) == 0:
        raise RuntimeError("No correctly classified CIFAR-10 test sample was found.")

    logits_np = test_logits.detach().numpy().astype(float)
    sorted_logits = np.sort(logits_np[correct_indices], axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    sample_index = int(correct_indices[int(np.argmax(margins))])
    sample = x_test[sample_index]
    sample_label = int(y_test[sample_index])
    sample_prediction = int(test_pred[sample_index])
    sample_logits = logits_np[sample_index]

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "tiny_cifar_mlp.pt")
    write_nnet(model, output_dir / "tiny_cifar_mlp.nnet")
    np.save(output_dir / "sample.npy", sample)
    save_preview(sample, output_dir / "sample_preview.png")

    metadata = {
        "dataset": "CIFAR-10 test/train data loaded with torchvision.datasets.CIFAR10",
        "dataset_origin": "CIFAR-10 external image classification benchmark",
        "preprocessing": "RGB 32x32 -> grayscale -> average pool to 8x8 -> flatten to 64 values in [0, 1]",
        "classes": CLASS_NAMES,
        "architecture": "64 -> 32 ReLU -> 10",
        "seed": SEED,
        "train_samples_per_class": train_per_class,
        "test_samples_per_class": test_per_class,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "sample_index_in_balanced_test_subset": sample_index,
        "sample_label": sample_label,
        "sample_label_name": CLASS_NAMES[sample_label],
        "sample_prediction": sample_prediction,
        "sample_prediction_name": CLASS_NAMES[sample_prediction],
        "sample_logits": sample_logits.tolist(),
        "sample_logit_margin": float(np.max(sample_logits) - np.partition(sample_logits, -2)[-2]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def save_preview(sample: np.ndarray, path: Path) -> None:
    image = np.clip(sample.reshape(8, 8) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").resize((160, 160), Image.Resampling.NEAREST).save(path)


def write_nnet(model: TinyCifarMLP, path: Path) -> None:
    """Write the trained 64 -> 32 -> 10 ReLU MLP in Marabou .nnet format."""

    first = model.net[0]
    second = model.net[2]
    assert isinstance(first, nn.Linear)
    assert isinstance(second, nn.Linear)

    weights = [
        first.weight.detach().cpu().numpy(),
        second.weight.detach().cpu().numpy(),
    ]
    biases = [
        first.bias.detach().cpu().numpy(),
        second.bias.detach().cpu().numpy(),
    ]

    layer_sizes = [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]
    with path.open("w") as f:
        f.write("// TinyCifarMLP trained on external CIFAR-10 data for Assignment 3 Problem 2\n")
        f.write(f"2,{INPUT_SIZE},{OUTPUT_SIZE},{max(layer_sizes)},\n")
        f.write(",".join(str(size) for size in layer_sizes) + ",\n")
        f.write("0,\n")
        f.write(",".join("0.0" for _ in range(INPUT_SIZE)) + ",\n")
        f.write(",".join("1.0" for _ in range(INPUT_SIZE)) + ",\n")
        f.write(",".join("0.0" for _ in range(INPUT_SIZE + 1)) + ",\n")
        f.write(",".join("1.0" for _ in range(INPUT_SIZE + 1)) + ",\n")

        for weight_matrix, bias_vector in zip(weights, biases):
            for row in weight_matrix:
                f.write(",".join(f"{float(value):.9g}" for value in row) + ",\n")
            for value in bias_vector:
                f.write(f"{float(value):.9g},\n")


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=default_output_dir(), type=Path)
    parser.add_argument("--data-root", default=default_data_root(), type=Path)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--train-per-class", default=500, type=int)
    parser.add_argument("--test-per-class", default=100, type=int)
    args = parser.parse_args()

    metadata = train_model(
        args.output_dir,
        data_root=args.data_root,
        epochs=args.epochs,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
