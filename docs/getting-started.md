# Getting Started

## Installation

```bash
pip install nanocli
# or
uv pip install nanocli
```

## Your First CLI

Create a simple CLI with typed configuration:

```python
from dataclasses import dataclass
from nanocli import group

@dataclass
class Config:
    name: str = "world"
    count: int = 1

app = group()

@app.command()
def hello(cfg: Config):
    for _ in range(cfg.count):
        print(f"Hello, {cfg.name}!")

if __name__ == "__main__":
    app()
```

## Running Your CLI

```bash
# Show help
python app.py -h

# Run with defaults
python app.py hello

# Override values
python app.py hello name=Alice count=3

# Print config
python app.py hello -p

# Load from YAML
python app.py -c config.yml hello
```

## Nested Groups

Organize commands into groups:

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
python app.py data download -g  # print global config
```

## Flags Reference

| Flag      | Meaning                         |
|-----------|---------------------------------|
| `-p`      | Print config from current node  |
| `-g`      | Print config from root (global) |
| `-h`      | Show help                       |
| `-c PATH` | Load base config from YAML      |
