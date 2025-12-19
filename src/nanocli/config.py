"""Config layer: Recursive config tree with OmegaConf.

This module provides configuration handling for NanoCLI:

- `option()` - Dataclass field wrapper with help text
- `compile_config()` - Pure function to compile configs
- `load_yaml()` / `to_yaml()` - YAML I/O
- `parse_overrides()` - Parse CLI overrides
"""

from dataclasses import MISSING, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


class ConfigError(Exception):
    """Configuration-related errors.

    Raised when config files are missing, overrides are invalid, etc.
    """


def option(
    default: Any = MISSING,
    *,
    help: str = "",
    **kwargs: Any,
) -> Any:
    """Dataclass field wrapper with help text for CLI.

    Use this instead of `field()` to add help text that appears in CLI help.

    Args:
        default: Default value for the field.
        help: Help text shown in CLI.
        **kwargs: Additional arguments passed to `dataclasses.field()`.

    Returns:
        A dataclass field with metadata.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     epochs: int = option(100, help="Number of epochs")
        ...     lr: float = option(0.001, help="Learning rate")
        >>> cfg = Config()
        >>> cfg.epochs
        100
    """
    metadata = kwargs.pop("metadata", {})
    metadata["help"] = help
    return field(default=default, metadata=metadata, **kwargs)


def load_yaml(path: str | Path) -> DictConfig:
    """Load a YAML file into a DictConfig.

    Args:
        path: Path to the YAML file.

    Returns:
        DictConfig containing the parsed YAML.

    Raises:
        ConfigError: If the file does not exist.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
        ...     _ = f.write("name: test\\ncount: 42")
        ...     path = f.name
        >>> cfg = load_yaml(path)
        >>> cfg.name
        'test'
        >>> Path(path).unlink()
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    return OmegaConf.load(path)  # type: ignore[return-value]


def parse_overrides(overrides: list[str]) -> DictConfig:
    """Parse CLI overrides into a config tree.

    Supports three types of overrides:
    - `key=value` - Scalar override
    - `key.path=value` - Nested override
    - `key=@file.yml` - Subtree replacement from file

    Args:
        overrides: List of override strings.

    Returns:
        DictConfig with parsed overrides.

    Raises:
        ConfigError: If an override doesn't contain '='.

    Examples:
        >>> cfg = parse_overrides(["name=test", "count=42"])
        >>> cfg.name
        'test'
        >>> cfg.count
        42
        >>> cfg = parse_overrides(["model.layers=24"])
        >>> cfg.model.layers
        24
    """
    result: dict[str, Any] = {}

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Invalid override: '{override}'. Expected 'key=value' format.")

        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Handle @file syntax for subtree replacement
        if value.startswith("@"):
            file_path = value[1:]
            parsed = OmegaConf.to_container(load_yaml(file_path))
        else:
            parsed = _parse_value(value)

        # Build nested dict from dot notation
        _set_nested(result, key.split("."), parsed)

    return OmegaConf.create(result)


def _parse_value(value: str) -> Any:
    """Parse a string value into Python type.

    Args:
        value: String to parse.

    Returns:
        Parsed Python value (bool, None, int, float, list, or str).

    Examples:
        >>> _parse_value("true")
        True
        >>> _parse_value("42")
        42
        >>> _parse_value("3.14")
        3.14
        >>> _parse_value("[1, 2, 3]")
        [1, 2, 3]
    """
    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # None
    if value.lower() in ("none", "null"):
        return None

    # List
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(v.strip()) for v in inner.split(",")]

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Quoted string
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    return value


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys.

    Args:
        d: Dictionary to modify.
        keys: List of keys forming the path.
        value: Value to set.

    Examples:
        >>> d = {}
        >>> _set_nested(d, ["a", "b", "c"], 42)
        >>> d
        {'a': {'b': {'c': 42}}}
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def compile_config(
    base: DictConfig | None = None,
    overrides: list[str] | None = None,
    schema: type[T] | None = None,
) -> DictConfig | T:
    """Compile a config from base + overrides.

    This is the core function: pure tree rewrite.
    Priority: schema defaults < base < overrides

    Args:
        base: Base config tree (from YAML).
        overrides: List of override strings (`key=value`, `key=@file`).
        schema: Optional dataclass for type validation.

    Returns:
        Compiled config. Typed if schema provided, else DictConfig.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     name: str = "default"
        ...     count: int = 1
        >>> cfg = compile_config(schema=Config)
        >>> cfg.name
        'default'
        >>> cfg = compile_config(schema=Config, overrides=["name=custom"])
        >>> cfg.name
        'custom'
    """
    # Build config: schema defaults -> base -> overrides
    if schema is not None:
        cfg = OmegaConf.structured(schema)
        if base is not None:
            cfg = OmegaConf.merge(cfg, base)
    else:
        cfg = base if base is not None else OmegaConf.create({})

    # Apply overrides (tree rewrite)
    if overrides:
        override_cfg = parse_overrides(overrides)
        try:
            cfg = OmegaConf.merge(cfg, override_cfg)
        except Exception as e:
            # Extract the key from OmegaConf error message
            error_msg = str(e)
            if "Key" in error_msg and "not in" in error_msg:
                # Parse: Key 'typer' not in 'ModelConfig'
                import re

                match = re.search(r"Key '(\w+)' not in '(\w+)'", error_msg)
                if match:
                    key, cls = match.groups()
                    raise ConfigError(
                        f"Invalid config key '{key}' in '{cls}'. Check for typos in your overrides."
                    ) from None
            # Re-raise with friendlier message
            raise ConfigError(f"Config error: {error_msg}") from None

    # Convert to typed object if schema provided
    if schema is not None:
        return OmegaConf.to_object(cfg)  # type: ignore[return-value]

    return cfg  # type: ignore[no-any-return]


def to_yaml(config: Any) -> str:
    """Convert config to YAML string.

    Args:
        config: Config object (dataclass, dict, or DictConfig).

    Returns:
        YAML string representation.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     name: str = "test"
        >>> yaml_str = to_yaml(Config())
        >>> "name: test" in yaml_str
        True
    """
    if is_dataclass(config) and not isinstance(config, type):
        cfg = OmegaConf.structured(config)
    elif isinstance(config, DictConfig):
        cfg = config
    else:
        cfg = OmegaConf.create(config)

    return OmegaConf.to_yaml(cfg)


def save_yaml(config: Any, path: str | Path) -> None:
    """Save config to YAML file.

    Args:
        config: Config object to save.
        path: Path to write the YAML file.

    Examples:
        >>> import tempfile
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     name: str = "test"
        >>> with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
        ...     save_yaml(Config(), f.name)
        ...     content = open(f.name).read()
        >>> "name: test" in content
        True
    """
    Path(path).write_text(to_yaml(config))


# --- Schema introspection for CLI help ---


def get_flattened_config_options(
    schema: type,
    prefix: str = "",
) -> list[tuple[str, str, Any, str]]:
    """Recursively flatten nested dataclasses into dotted paths.

    Args:
        schema: Dataclass type to introspect.
        prefix: Current path prefix (for recursion).

    Returns:
        List of tuples: (dotted_name, type_name, default, help_text).

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     name: str = option("default", help="Name field")
        >>> opts = get_flattened_config_options(Config)
        >>> opts[0][0]
        'name'
    """
    if not is_dataclass(schema):
        return []

    result = []

    for f in fields(schema):
        name = f.name
        full_name = f"{prefix}.{name}" if prefix else name
        help_text = f.metadata.get("help", "")
        field_type = f.type

        if is_dataclass(field_type):
            # Get help from nested dataclass docstring if not provided
            if not help_text and field_type.__doc__:
                help_text = field_type.__doc__.strip().split("\n")[0]
            # Recurse into nested dataclass
            result.extend(get_flattened_config_options(field_type, prefix=full_name))  # type: ignore[arg-type]
        else:
            default = f.default if f.default is not MISSING else None
            type_name = getattr(field_type, "__name__", str(field_type))
            result.append((full_name, type_name, default, help_text))

    return result


def get_schema_structure(schema: type) -> tuple[dict[str, type], list[tuple[str, str, Any, str]]]:
    """Inspect schema to get flattened config options.

    Args:
        schema: Dataclass type to introspect.

    Returns:
        Tuple of (subcommands, config_options):
        - subcommands: Always empty dict (for backward compatibility)
        - config_options: List of (dotted_name, type_name, default, help)
    """
    return {}, get_flattened_config_options(schema)
