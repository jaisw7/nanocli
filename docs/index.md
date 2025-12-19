# NanoCLI

A cli framework with the simplicity of argparse, the colors of rich, the config handling of Hydra, and without the complexity.

## Installation

```bash
pip install nanocli
```

## Quick Example

```python
from dataclasses import dataclass
from nanocli import group

@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.001

app = group()

@app.command()
def train(cfg: TrainConfig):
    print(f"Training for {cfg.epochs} epochs")

if __name__ == "__main__":
    app()
```

## Key Features

- **Type-safe configs** from dataclasses
- **Nested groups** as a unified tree structure
- **YAML configs** with dotted-path overrides
- **Rich help output** with colors
- **Zero boilerplate** - your function signature IS the CLI
