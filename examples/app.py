"""Example using NanoCLI with option() for help text.

Usage:
    python app.py --help
    python app.py --cfg config.yml
    python app.py epochs=50 lr=0.001
    python app.py model.type=vit model.layers=100
    python app.py --print
"""

from dataclasses import dataclass, field

from nanocli import option, run


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    type: str = option("resnet", help="Model architecture (resnet, vit)")
    layers: int = option(50, help="Number of layers")
    hidden_dim: int = option(512, help="Hidden dimension")


@dataclass
class TrainConfig:
    """Training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = option(100, help="Number of training epochs")
    lr: float = option(0.001, help="Learning rate")
    batch_size: int = option(32, help="Batch size")
    device: str = option("cuda", help="Device to use")


def train(cfg: TrainConfig) -> None:
    """Train the model with given config."""
    print(f"Training {cfg.model.type} model")
    print(f"  Layers: {cfg.model.layers}")
    print(f"  Hidden dim: {cfg.model.hidden_dim}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  LR: {cfg.lr}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Device: {cfg.device}")


if __name__ == "__main__":
    run(train)
