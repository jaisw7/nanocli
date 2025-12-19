"""NanoCLI - CLI framework derived from config schema.

Core model:
    Config := Map<String, Value>
    Value  := Scalar | Config | List<Value>

The CLI structure is derived from the config schema.
"""

__version__ = "0.1.0"

from nanocli.config import (
    ConfigError,
    compile_config,
    load_yaml,
    option,
    parse_overrides,
    save_yaml,
    to_yaml,
)
from nanocli.core import NanoCLI, group, run

__all__ = [
    # Core
    "NanoCLI",
    "group",
    "run",
    # Config
    "option",
    "compile_config",
    "load_yaml",
    "save_yaml",
    "to_yaml",
    "parse_overrides",
    "ConfigError",
]
