"""Comprehensive tests for NanoCLI core framework."""

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nanocli import NanoCLI, group, run
from nanocli.core import parse_args


@dataclass
class SimpleConfig:
    name: str = "world"
    count: int = 1


@dataclass
class NestedConfig:
    value: int = 10


@dataclass
class ParentConfig:
    child: NestedConfig = field(default_factory=NestedConfig)
    name: str = "parent"


class TestParseArgs:
    """Tests for parse_args function."""

    def test_empty_args(self):
        path, overrides, flags = parse_args([])
        assert path == []
        assert overrides == []
        assert flags["print"] is False
        assert flags["help"] is False
        assert flags["cfg"] is None

    def test_single_command(self):
        path, overrides, flags = parse_args(["train"])
        assert path == ["train"]
        assert overrides == []

    def test_nested_command(self):
        path, overrides, flags = parse_args(["data", "download"])
        assert path == ["data", "download"]

    def test_overrides(self):
        path, overrides, flags = parse_args(["train", "epochs=100", "lr=0.01"])
        assert path == ["train"]
        assert overrides == ["epochs=100", "lr=0.01"]

    def test_print_flag(self):
        path, overrides, flags = parse_args(["-p"])
        assert flags["print"] is True

    def test_print_flag_long(self):
        path, overrides, flags = parse_args(["--print"])
        assert flags["print"] is True

    def test_help_flag(self):
        path, overrides, flags = parse_args(["-h"])
        assert flags["help"] is True

    def test_help_flag_long(self):
        path, overrides, flags = parse_args(["--help"])
        assert flags["help"] is True

    def test_global_flag(self):
        path, overrides, flags = parse_args(["-g"])
        assert flags["print_global"] is True

    def test_global_flag_long(self):
        path, overrides, flags = parse_args(["--global"])
        assert flags["print_global"] is True

    def test_cfg_flag(self):
        path, overrides, flags = parse_args(["-c", "config.yml"])
        assert flags["cfg"] == "config.yml"

    def test_cfg_flag_long(self):
        path, overrides, flags = parse_args(["--cfg", "config.yml"])
        assert flags["cfg"] == "config.yml"

    def test_cfg_flag_no_value(self):
        path, overrides, flags = parse_args(["-c"])
        assert flags["cfg"] is None

    def test_dotted_override(self):
        path, overrides, flags = parse_args(["train.epochs=100", "-p"])
        assert path == []
        assert overrides == ["train.epochs=100"]
        assert flags["print"] is True

    def test_mixed_args(self):
        path, overrides, flags = parse_args(
            ["data", "download", "name=test", "-p", "-c", "cfg.yml"]
        )
        assert path == ["data", "download"]
        assert overrides == ["name=test"]
        assert flags["print"] is True
        assert flags["cfg"] == "cfg.yml"


class TestNanoCLI:
    """Tests for NanoCLI class."""

    def test_create_app(self):
        app = NanoCLI(name="testapp", help="Test")
        assert app._name == "testapp"
        assert app._help == "Test"

    def test_create_app_defaults(self):
        app = NanoCLI()
        assert app._name is None
        assert app._help is None
        assert app._parent is None

    def test_command_registration(self):
        app = NanoCLI()

        @app.command()
        def hello(cfg: SimpleConfig):
            pass

        assert "hello" in app._commands

    def test_command_with_name(self):
        app = NanoCLI()

        @app.command(name="greet")
        def hello(cfg: SimpleConfig):
            pass

        assert "greet" in app._commands

    def test_group_registration(self):
        app = NanoCLI()
        data = app.group("data", help="Data commands")

        assert "data" in app._groups
        assert data._name == "data"
        assert data._parent is app

    def test_get_root(self):
        app = NanoCLI(name="root")
        child = app.group("child")
        grandchild = child.group("grandchild")

        assert grandchild._get_root() is app
        assert child._get_root() is app
        assert app._get_root() is app

    def test_get_path(self):
        app = NanoCLI(name="root")
        child = app.group("child")
        grandchild = child.group("grandchild")

        assert grandchild._get_path() == "child.grandchild"
        assert child._get_path() == "child"
        assert app._get_path() == ""

    def test_command_execution(self):
        app = NanoCLI()

        @app.command()
        def hello(cfg: SimpleConfig):
            print(f"Hello, {cfg.name}!")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["hello"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Hello, world!" in output

    def test_command_with_overrides(self):
        app = NanoCLI()

        @app.command()
        def greet(cfg: SimpleConfig):
            print(f"Greeting: {cfg.name}")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["greet", "name=Alice"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Alice" in output

    def test_command_without_config(self):
        app = NanoCLI()

        @app.command()
        def simple():
            print("No config needed")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["simple"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "No config needed" in output

    def test_nested_group_command(self):
        app = NanoCLI()
        data = app.group("data", help="Data commands")

        @data.command()
        def download(cfg: SimpleConfig):
            print(f"Downloading {cfg.name}")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["data", "download"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Downloading world" in output

    def test_print_flag(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["train", "-p"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "name: world" in output
        assert "count: 1" in output

    def test_help_flag_group(self):
        app = NanoCLI(name="myapp", help="My application")

        @app.command()
        def train(cfg: SimpleConfig):
            """Train a model."""
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["-h"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "myapp" in output or "Usage" in output

    def test_help_flag_command(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            """Train a model."""
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["train", "-h"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "train" in output or "Usage" in output

    def test_unknown_command(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        with pytest.raises(SystemExit):
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                app(["unknown"])
            finally:
                sys.stdout = old_stdout

    def test_no_command_shows_message(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app([])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "No command specified" in output or "-h" in output

    def test_print_group_config(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["-p"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "train:" in output

    def test_print_global_from_nested(self):
        app = NanoCLI()
        data = app.group("data")

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        @data.command()
        def download(cfg: SimpleConfig):
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["data", "download", "-g"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "train:" in output
        assert "data:" in output

    def test_collect_configs(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        @app.command()
        def eval(cfg: SimpleConfig):
            pass

        configs = app._collect_configs(None, [])
        assert "train" in configs
        assert "eval" in configs

    def test_collect_nested_configs(self):
        app = NanoCLI()
        data = app.group("data")

        @data.command()
        def download(cfg: SimpleConfig):
            pass

        configs = app._collect_configs(None, [])
        assert "data" in configs
        assert "download" in configs["data"]

    def test_config_with_yaml(self, tmp_path: Path):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            print(f"Training {cfg.name}")

        # Global tree structure - train subtree will be extracted
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text("train:\n  name: from_yaml\n  count: 42")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["train", "-c", str(yaml_file)])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "from_yaml" in output

    def test_nested_config_overrides(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: ParentConfig):
            print(f"Value: {cfg.child.value}")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["train", "child.value=99"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "99" in output


class TestGroupFunction:
    """Tests for group() convenience function."""

    def test_create_group(self):
        app = group()
        assert isinstance(app, NanoCLI)

    def test_group_with_name(self):
        app = group(name="myapp", help="My app")
        assert app._name == "myapp"
        assert app._help == "My app"


class TestRunFunction:
    """Tests for run() convenience function."""

    def test_run_with_schema(self):
        old_argv = sys.argv
        sys.argv = ["test", "-p"]

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            run(SimpleConfig)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
            sys.argv = old_argv

        assert "name: world" in output

    def test_run_with_function(self):
        results = []

        def my_func(cfg: SimpleConfig):
            results.append(cfg.name)
            return cfg.name

        old_argv = sys.argv
        sys.argv = ["test"]

        try:
            result = run(my_func, args=[])
        finally:
            sys.argv = old_argv

        assert results == ["world"]
        assert result == "world"

    def test_run_with_overrides(self):
        results = []

        def my_func(cfg: SimpleConfig):
            results.append(cfg.name)
            return cfg

        result = run(my_func, args=["name=custom"])
        assert result.name == "custom"

    def test_run_help(self):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            run(SimpleConfig, args=["-h"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Usage" in output or "Options" in output

    def test_run_no_schema_error(self):
        def no_config_func():
            pass

        with pytest.raises(ValueError, match="Could not infer config schema"):
            run(no_config_func, args=[])


class TestConfigHelp:
    """Tests for help display with config options."""

    def test_command_help_shows_config(self):
        from nanocli import option

        @dataclass
        class DetailedConfig:
            epochs: int = option(100, help="Number of epochs")
            lr: float = option(0.001, help="Learning rate")

        app = NanoCLI()

        @app.command()
        def train(cfg: DetailedConfig):
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["train", "-h"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "epochs" in output or "Config" in output

    def test_group_help_shows_commands(self):
        app = NanoCLI(name="myapp", help="My application")
        data = app.group("data", help="Data commands")

        @app.command()
        def train(cfg: SimpleConfig):
            """Train a model."""
            pass

        @data.command()
        def download(cfg: SimpleConfig):
            """Download data."""
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["-h"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "train" in output or "Commands" in output

    def test_nested_group_help(self):
        app = NanoCLI(name="myapp")
        data = app.group("data", help="Data commands")

        @data.command()
        def download(cfg: SimpleConfig):
            pass

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["data", "-h"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "download" in output or "data" in output


class TestCollectConfigsWithOverrides:
    """Tests for _collect_configs with various override scenarios."""

    def test_collect_with_dotted_overrides(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        configs = app._collect_configs(None, ["train.name=custom"])
        assert configs["train"].name == "custom"

    def test_collect_with_simple_overrides(self):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        configs = app._collect_configs(None, ["name=simple"])
        assert "train" in configs


class TestPrintWithYaml:
    """Tests for print functionality with YAML files."""

    def test_print_with_yaml_file(self, tmp_path: Path):
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        # Global tree structure
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text("train:\n  name: yaml_test\n  count: 99")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["train", "-c", str(yaml_file), "-p"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "yaml_test" in output
        assert "99" in output

    def test_print_global_with_yaml(self, tmp_path: Path):
        app = NanoCLI()
        data = app.group("data")

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        @data.command()
        def download(cfg: SimpleConfig):
            pass

        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text("name: global_yaml\ncount: 50")

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            app(["data", "download", "-c", str(yaml_file), "-g"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "train:" in output or "data:" in output


class TestErrorHandling:
    """Tests for graceful error handling."""

    def test_invalid_override_key_in_run(self):
        """Test that invalid config keys show helpful error."""
        from nanocli.config import ConfigError, compile_config

        with pytest.raises(ConfigError, match="Invalid config key"):
            compile_config(schema=SimpleConfig, overrides=["invalid_key=value"])

    def test_invalid_nested_key(self):
        """Test that invalid nested keys are caught."""
        from nanocli.config import ConfigError, compile_config

        with pytest.raises(ConfigError, match="Invalid config key"):
            compile_config(schema=ParentConfig, overrides=["child.invalid=value"])

    def test_run_with_invalid_key_exits(self):
        """Test that run() exits gracefully with invalid key."""
        with pytest.raises(SystemExit):
            run(SimpleConfig, args=["invalid_key=value"])

    def test_nanocli_command_with_invalid_key(self):
        """Test NanoCLI command with invalid key shows error and help."""
        app = NanoCLI()

        @app.command()
        def train(cfg: SimpleConfig):
            pass

        with pytest.raises(SystemExit):
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                app(["train", "invalid_key=value"])
            finally:
                sys.stdout = old_stdout


class TestConfigErrorMessages:
    """Tests for config error message formatting."""

    def test_parse_overrides_no_equals(self):
        """Test that missing = raises ConfigError."""
        from nanocli.config import ConfigError, parse_overrides

        with pytest.raises(ConfigError, match="Invalid override"):
            parse_overrides(["no_equals_sign"])

    def test_config_error_message_has_key(self):
        """Test that error message contains the invalid key."""
        from nanocli.config import ConfigError, compile_config

        try:
            compile_config(schema=SimpleConfig, overrides=["typo_key=value"])
        except ConfigError as e:
            assert "typo_key" in str(e)
