#!/usr/bin/env python3
"""Train a small external model and export it to Marabou's .nnet format.

The dataset is generated locally so the assignment remains reproducible without
network downloads. Each 8x8 sample belongs to one of three simple image classes:
vertical bar, horizontal bar, or diagonal bar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_SIZE = 64
HIDDEN_SIZE = 8
OUTPUT_SIZE = 3
SEED = 7


class TinyBarsNet(nn.Module):
    """Small fully connected ReLU network: 64 -> 8 -> 3."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_bars_dataset(samples_per_class: int = 240, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic 8x8 image dataset with mild random noise."""

    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label in range(OUTPUT_SIZE):
        for _ in range(samples_per_class):
            image = rng.uniform(0.0, 0.06, size=(8, 8)).astype(np.float32)

            if label == 0:
                col = int(rng.choice([2, 3, 4, 5]))
                image[:, col] = rng.uniform(0.82, 1.0, size=8)
            elif label == 1:
                row = int(rng.choice([2, 3, 4, 5]))
                image[row, :] = rng.uniform(0.82, 1.0, size=8)
            else:
                offset = int(rng.choice([-1, 0, 1]))
                for r in range(8):
                    c = r + offset
                    if 0 <= c < 8:
                        image[r, c] = float(rng.uniform(0.82, 1.0))

            images.append(np.clip(image, 0.0, 1.0).reshape(-1))
            labels.append(label)

    x = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    permutation = rng.permutation(len(y))
    return x[permutation], y[permutation]


def split_dataset(x: np.ndarray, y: np.ndarray, train_fraction: float = 0.8) -> tuple:
    n_train = int(len(y) * train_fraction)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def train_model(output_dir: Path, epochs: int = 120) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x, y = make_bars_dataset()
    x_train, y_train, x_test, y_test = split_dataset(x, y)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
    )

    model = TinyBarsNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
        train_pred = model(torch.from_numpy(x_train)).argmax(dim=1).numpy()
        test_logits = model(torch.from_numpy(x_test))
        test_pred = test_logits.argmax(dim=1).numpy()

    train_accuracy = float((train_pred == y_train).mean())
    test_accuracy = float((test_pred == y_test).mean())

    correct_indices = np.flatnonzero(test_pred == y_test)
    if len(correct_indices) == 0:
        raise RuntimeError("No correctly classified test sample was found.")

    sample_index = int(correct_indices[0])
    sample = x_test[sample_index]
    sample_label = int(y_test[sample_index])
    sample_prediction = int(test_pred[sample_index])
    sample_logits = test_logits[sample_index].detach().numpy().astype(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "tiny_bars.pt")
    write_nnet(model, output_dir / "tiny_bars.nnet")
    np.save(output_dir / "sample.npy", sample)

    metadata = {
        "dataset": "Synthetic 8x8 BarsDataset generated in train_model.py",
        "classes": ["vertical_bar", "horizontal_bar", "diagonal_bar"],
        "architecture": "64 -> 8 ReLU -> 3",
        "seed": SEED,
        "samples_per_class": 240,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "sample_index_in_test_split": sample_index,
        "sample_label": sample_label,
        "sample_prediction": sample_prediction,
        "sample_logits": sample_logits.tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def write_nnet(model: TinyBarsNet, path: Path) -> None:
    """Write the trained 64 -> 8 -> 3 ReLU MLP in Marabou .nnet format."""

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
        f.write("// TinyBarsNet trained for Assignment 3 Problem 2\n")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts", type=Path)
    parser.add_argument("--epochs", default=120, type=int)
    args = parser.parse_args()

    metadata = train_model(args.output_dir, epochs=args.epochs)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
