"""Example with nested groups.

Structure:
    app (root group)
    ├── train (command)
    ├── eval (command)
    └── data (nested group)
        ├── download (command)
        └── preprocess (command)

Usage:
    python nested_groups.py --help
    python nested_groups.py -p
    python nested_groups.py train --help
    python nested_groups.py train -p
    python nested_groups.py train -pg
    python nested_groups.py data --help
    python nested_groups.py data download -p
    python nested_groups.py data -p
"""

from dataclasses import dataclass

from nanocli import group, option


# Root commands
@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = option(100, help="Number of training epochs")
    lr: float = option(0.001, help="Learning rate")
    batch_size: int = option(32, help="Batch size")


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    checkpoint: str = option("model.pt", help="Path to checkpoint")
    batch_size: int = option(64, help="Batch size for evaluation")


# Nested group commands
@dataclass
class DownloadConfig:
    """Download configuration."""

    dataset: str = option("imagenet", help="Dataset name")
    output_dir: str = option("./data", help="Output directory")


@dataclass
class PreprocessConfig:
    """Preprocess configuration."""

    input_dir: str = option("./data", help="Input directory")
    output_dir: str = option("./processed", help="Output directory")
    resize: int = option(224, help="Resize images to this size")


# Create root group
app = group()


@app.command()
def train(cfg: TrainConfig) -> None:
    """Train the model."""
    print("Training model:")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  LR: {cfg.lr}")
    print(f"  Batch size: {cfg.batch_size}")


@app.command()
def eval(cfg: EvalConfig) -> None:
    """Evaluate the model."""
    print("Evaluating model:")
    print(f"  Checkpoint: {cfg.checkpoint}")
    print(f"  Batch size: {cfg.batch_size}")


# Create nested group
data = app.group("data", help="Data management commands")


@data.command()
def download(cfg: DownloadConfig) -> None:
    """Download dataset."""
    print("Downloading dataset:")
    print(f"  Dataset: {cfg.dataset}")
    print(f"  Output: {cfg.output_dir}")


@data.command()
def preprocess(cfg: PreprocessConfig) -> None:
    """Preprocess dataset."""
    print("Preprocessing dataset:")
    print(f"  Input: {cfg.input_dir}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  Resize: {cfg.resize}")


if __name__ == "__main__":
    app()
