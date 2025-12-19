"""Tests for the config layer."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from omegaconf import DictConfig

from nanocli.config import (
    ConfigError,
    compile_config,
    load_yaml,
    parse_overrides,
    to_yaml,
)


# Test schemas
@dataclass
class NestedConfig:
    layers: int = 50
    hidden_dim: int = 512


@dataclass
class SimpleConfig:
    name: str = "default"
    count: int = 1
    rate: float = 0.001


@dataclass
class ComplexConfig:
    model: NestedConfig = field(default_factory=NestedConfig)
    epochs: int = 100
    lr: float = 0.001


class TestLoadYaml:
    """Tests for load_yaml."""

    def test_load_existing_file(self, tmp_path: Path):
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text("name: test\ncount: 5")

        cfg = load_yaml(yaml_file)

        assert cfg.name == "test"
        assert cfg.count == 5

    def test_load_missing_file(self):
        with pytest.raises(ConfigError, match="not found"):
            load_yaml("/nonexistent/file.yml")


class TestParseOverrides:
    """Tests for parse_overrides."""

    def test_simple_override(self):
        cfg = parse_overrides(["name=test"])
        assert cfg.name == "test"

    def test_nested_override(self):
        cfg = parse_overrides(["model.layers=24"])
        assert cfg.model.layers == 24

    def test_multiple_overrides(self):
        cfg = parse_overrides(["name=test", "count=10"])
        assert cfg.name == "test"
        assert cfg.count == 10

    def test_integer_value(self):
        cfg = parse_overrides(["count=42"])
        assert cfg.count == 42
        assert isinstance(cfg.count, int)

    def test_float_value(self):
        cfg = parse_overrides(["rate=0.001"])
        assert cfg.rate == 0.001

    def test_boolean_true(self):
        cfg = parse_overrides(["enabled=true"])
        assert cfg.enabled is True

    def test_boolean_false(self):
        cfg = parse_overrides(["enabled=false"])
        assert cfg.enabled is False

    def test_none_value(self):
        cfg = parse_overrides(["value=none"])
        assert cfg.value is None

    def test_list_value(self):
        cfg = parse_overrides(["items=[1, 2, 3]"])
        assert cfg["items"] == [1, 2, 3]

    def test_empty_list(self):
        cfg = parse_overrides(["items=[]"])
        assert cfg["items"] == []

    def test_invalid_syntax(self):
        with pytest.raises(ConfigError, match="Invalid override"):
            parse_overrides(["no_equals"])


class TestCompileConfig:
    """Tests for compile_config - the core function."""

    def test_compile_empty(self):
        cfg = compile_config()
        assert isinstance(cfg, DictConfig)

    def test_compile_with_schema_defaults(self):
        cfg = compile_config(schema=SimpleConfig)
        assert cfg.name == "default"
        assert cfg.count == 1

    def test_compile_with_overrides(self):
        cfg = compile_config(
            schema=SimpleConfig,
            overrides=["name=custom", "count=10"],
        )
        assert cfg.name == "custom"
        assert cfg.count == 10

    def test_compile_nested_overrides(self):
        cfg = compile_config(
            schema=ComplexConfig,
            overrides=["model.layers=24", "epochs=50"],
        )
        assert cfg.model.layers == 24
        assert cfg.epochs == 50

    def test_compile_base_plus_overrides(self, tmp_path: Path):
        yaml_file = tmp_path / "base.yml"
        yaml_file.write_text("name: from_file\ncount: 5\nrate: 0.001")

        base = load_yaml(yaml_file)
        cfg = compile_config(
            base=base,
            overrides=["count=10"],
            schema=SimpleConfig,
        )

        assert cfg.name == "from_file"  # From base
        assert cfg.count == 10  # Overridden


class TestToYaml:
    """Tests for to_yaml synthesis."""

    def test_dataclass_to_yaml(self):
        cfg = SimpleConfig(name="test", count=5)
        yaml_str = to_yaml(cfg)

        assert "name: test" in yaml_str
        assert "count: 5" in yaml_str

    def test_nested_to_yaml(self):
        cfg = ComplexConfig()
        yaml_str = to_yaml(cfg)

        assert "model:" in yaml_str
        assert "layers:" in yaml_str
        assert "epochs:" in yaml_str

    def test_round_trip(self, tmp_path: Path):
        # Create config with overrides
        cfg1 = compile_config(
            schema=ComplexConfig,
            overrides=["model.layers=24", "epochs=50"],
        )

        # Write to YAML
        yaml_file = tmp_path / "roundtrip.yml"
        yaml_file.write_text(to_yaml(cfg1))

        # Load back - use DictConfig since we're not re-merging with schema
        cfg2 = load_yaml(yaml_file)

        assert cfg1.model.layers == cfg2.model.layers
        assert cfg1.epochs == cfg2.epochs
