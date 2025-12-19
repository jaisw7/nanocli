# NanoCLI

[![PyPI version](https://badge.fury.io/py/nanocli.svg)](https://pypi.org/project/nanocli/)

A cli framework with the simplicity of argparse, the colors of rich, the config handling of Hydra, and without the complexity.

![NanoCLI Demo](assets/demo.gif)

## Installation

```bash
pip install nanocli
```

## Core Model

```text
CLI = YAML Tree
├── commands (leaf nodes)
└── groups (subtrees)
```

## Quick Start

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

## Usage

```bash
# Run command
python app.py train
python app.py train epochs=200

# Hydra-style overrides at root level
python app.py train.epochs=200 -p

# Print config: -p (local), -g (global from root)
python app.py train -p
python app.py data download -g

# Load YAML config
python app.py -c config.yml train

# Help
python app.py -h
python app.py train -h
```

## Nested Groups

```python
app = group()

@app.command()
def train(cfg: TrainConfig):
    ...

data = app.group("data", help="Data commands")

@data.command()
def download(cfg: DownloadConfig):
    ...
```

```bash
python app.py data download
python app.py data download dataset=coco -p
python app.py data download dataset=coco -g  # prints full tree
```

## Flags

| Flag      | Meaning                         |
|-----------|---------------------------------|
| `-p`      | Print config from current node  |
| `-g`      | Print config from root (global) |
| `-h`      | Show help                       |
| `-c PATH` | Load base config from YAML      |

## Development

```bash
make dev          # Install with dev deps
make test         # Run tests
make pre-commit   # Run all checks
make lint-fix     # Fix lint issues
make type-check   # Type check
```

## License

MIT
