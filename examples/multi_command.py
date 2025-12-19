"""Example using group() for multi-command CLI.

Usage:
    python multi_command.py --help
    python multi_command.py train --help
    python multi_command.py train epochs=10
    python multi_command.py eval checkpoint=model.pt
"""

from dataclasses import dataclass

from nanocli import group, option


@dataclass
class DataConfig:
    """Data configuration."""

    type: str = option("resnet", help="Data type (resnet, vit)")
    path: str = option("data", help="Path to data")


@dataclass
class TrainConfig:
    """Training configuration."""

    data: DataConfig = option(default_factory=DataConfig)
    epochs: int = option(100, help="Number of training epochs")
    lr: float = option(0.001, help="Learning rate")
    batch_size: int = option(32, help="Batch size")


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    checkpoint: str = option("model.pt", help="Path to checkpoint")
    batch_size: int = option(64, help="Batch size for evaluation")


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


if __name__ == "__main__":
    app()
