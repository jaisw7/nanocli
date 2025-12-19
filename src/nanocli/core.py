"""Core NanoCLI framework - CLI as a unified YAML tree.

The CLI structure is a tree:
    - Root = entry point (YAML root)
    - Groups = non-leaf nodes (subtrees)
    - Commands = leaf nodes (functions with schemas)
    - Overrides = dotted paths into the tree

No Click dependency - custom argument parser.
"""

import sys
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any, TypeVar

from rich.console import Console
from rich.panel import Panel

from nanocli.config import (
    ConfigError,
    compile_config,
    get_schema_structure,
    load_yaml,
    to_yaml,
)

T = TypeVar("T")


def parse_args(args: list[str]) -> tuple[list[str], list[str], dict[str, Any]]:
    """Parse CLI arguments into path, overrides, and flags.

    Separates command path segments, key=value overrides, and special flags.

    Args:
        args: List of CLI arguments.

    Returns:
        Tuple of (path_parts, overrides, flags):
        - path_parts: List of path segments (e.g., ["data", "download"])
        - overrides: List of key=value overrides
        - flags: Dict with "print", "print_global", "help", "cfg"

    Examples:
        >>> path, overrides, flags = parse_args(["train", "epochs=100", "-p"])
        >>> path
        ['train']
        >>> overrides
        ['epochs=100']
        >>> flags["print"]
        True
    """
    path_parts = []
    overrides = []
    flags: dict[str, Any] = {
        "print": False,
        "print_global": False,
        "help": False,
        "cfg": None,
    }

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("-p", "--print"):
            flags["print"] = True
        elif arg in ("-g", "--global"):
            flags["print_global"] = True
        elif arg in ("-h", "--help"):
            flags["help"] = True
        elif arg in ("-c", "--cfg"):
            if i + 1 < len(args):
                flags["cfg"] = args[i + 1]
                i += 1
        elif "=" in arg:
            overrides.append(arg)
        else:
            # Path segment (command or group name)
            path_parts.append(arg)

        i += 1

    return path_parts, overrides, flags


class NanoCLI:
    """Unified CLI tree for command groups and commands.

    The CLI is structured as a tree where groups are non-leaf nodes
    and commands are leaf nodes with associated config schemas.

    Attributes:
        _name: Name of this group.
        _help: Help text for this group.
        _parent: Parent NanoCLI instance (None for root).
        _commands: Registered commands.
        _groups: Nested groups.

    Examples:
        >>> app = NanoCLI(name="myapp")
        >>> @app.command()
        ... def hello(cfg):
        ...     print("Hello!")
        >>> "hello" in app._commands
        True
    """

    def __init__(
        self,
        name: str | None = None,
        help: str | None = None,
        parent: "NanoCLI | None" = None,
    ):
        self._name = name
        self._help = help
        self._parent = parent
        self._commands: dict[str, tuple[Callable[..., Any], type | None]] = {}
        self._groups: dict[str, NanoCLI] = {}

    def __call__(self, args: list[str] | None = None) -> None:
        """Run the CLI."""
        if args is None:
            args = sys.argv[1:]

        self._execute(args)

    def _get_root(self) -> "NanoCLI":
        """Get the root NanoCLI by traversing up the parent chain."""
        current = self
        while current._parent is not None:
            current = current._parent
        return current

    def _get_path(self) -> str:
        """Get the dotted path from root to this group."""
        path_parts = []
        current = self
        while current._parent is not None:
            if current._name:
                path_parts.append(current._name)
            current = current._parent
        return ".".join(reversed(path_parts))

    def _collect_configs(
        self, cfg_file: str | None, overrides: list[str], prefix: str = ""
    ) -> dict[str, Any]:
        """Recursively collect configs from all commands and groups."""
        configs: dict[str, Any] = {}

        for cmd_name, (_, schema) in self._commands.items():
            if schema is not None:
                full_name = f"{prefix}{cmd_name}" if prefix else cmd_name
                cmd_overrides = []
                for ov in overrides:
                    if ov.startswith(f"{full_name}."):
                        cmd_overrides.append(ov[len(full_name) + 1 :])
                    elif "." not in ov.split("=")[0]:
                        cmd_overrides.append(ov)

                base = load_yaml(cfg_file) if cfg_file else None
                config = compile_config(base=base, overrides=cmd_overrides, schema=schema)
                configs[cmd_name] = config

        for group_name, sub_app in self._groups.items():
            configs[group_name] = sub_app._collect_configs(
                cfg_file, overrides, f"{prefix}{group_name}."
            )

        return configs

    def _execute(self, args: list[str]) -> None:
        """Execute the CLI with given arguments."""
        path_parts, overrides, flags = parse_args(args)
        console = Console()

        # Handle help at current level
        if flags["help"] and not path_parts:
            self._show_help(console)
            return

        # Find the target node (traverse path)
        current = self
        consumed_path = []

        for part in path_parts:
            if part in current._groups:
                consumed_path.append(part)
                current = current._groups[part]
            elif part in current._commands:
                consumed_path.append(part)
                # Found a command - execute it
                func, schema = current._commands[part]

                if flags["help"]:
                    current._show_command_help(console, part, schema)
                    return

                # Get cfg file - load global tree and extract subtree
                cfg_file = flags["cfg"]
                base = None
                if cfg_file:
                    global_cfg = load_yaml(cfg_file)
                    # Extract subtree based on consumed path
                    base = global_cfg
                    for path_key in consumed_path:
                        if hasattr(base, path_key) or (isinstance(base, dict) and path_key in base):
                            base = base[path_key]
                        else:
                            # Path doesn't exist in config, use None
                            base = None
                            break

                # Compile config
                if schema:
                    try:
                        config = compile_config(base=base, overrides=overrides, schema=schema)
                    except ConfigError as e:
                        console.print(f"[red]Error:[/red] {e}")
                        console.print()
                        current._show_command_help(console, part, schema)
                        sys.exit(1)
                else:
                    config = None

                if flags["print"]:
                    # Print just this command's config
                    if config:
                        console.print(to_yaml(config), end="")
                    return

                if flags["print_global"]:
                    # Print full tree from root
                    root = current._get_root()
                    full_path = ".".join(consumed_path)
                    root_overrides = [f"{full_path}.{ov}" for ov in overrides]
                    all_configs = root._collect_configs(cfg_file, root_overrides)
                    console.print(to_yaml(all_configs), end="")
                    return

                # Execute the command
                if config:
                    func(config)
                else:
                    func()
                return
            else:
                console.print(f"[red]Error: Unknown command or group '{part}'[/red]")
                console.print(
                    f"[dim]Available: {list(current._commands.keys()) + list(current._groups.keys())}[/dim]"
                )
                sys.exit(1)

        # No command found - we're at a group level
        if flags["help"]:
            current._show_help(console)
            return

        if flags["print"] or flags["print_global"]:
            cfg_file = flags["cfg"]
            if flags["print_global"] and current._parent:
                # Print from root
                root = current._get_root()
                path = current._get_path()
                root_overrides = [f"{path}.{ov}" if path else ov for ov in overrides]
                all_configs = root._collect_configs(cfg_file, root_overrides)
            else:
                # Print from current node
                all_configs = current._collect_configs(cfg_file, overrides)
            console.print(to_yaml(all_configs), end="")
            return

        # No command specified - show help
        console.print("[yellow]No command specified. Use -h to see available commands.[/yellow]")

    def _show_help(self, console: Console) -> None:
        """Show help for this group."""
        name = self._name or "app"

        # Usage
        console.print(
            Panel(
                f"[bold]{name}[/bold] [cyan][OPTIONS][/cyan] [magenta]COMMAND[/magenta]",
                title="[bold blue]Usage[/bold blue]",
                border_style="blue",
            )
        )
        console.print()

        # Description
        if self._help:
            console.print(self._help)
            console.print()

        # Options
        if self._parent:
            options_text = (
                "  [cyan]-c, --cfg PATH[/cyan]  Load config from YAML file\n"
                "  [cyan]-p[/cyan]              Print config and exit\n"
                "  [cyan]-g[/cyan]              Print root config (global) and exit\n"
                "  [cyan]-h[/cyan]              Show this help"
            )
        else:
            options_text = (
                "  [cyan]-c, --cfg PATH[/cyan]  Load config from YAML file\n"
                "  [cyan]-p[/cyan]              Print config and exit\n"
                "  [cyan]-h[/cyan]              Show this help"
            )
        console.print(
            Panel(
                options_text,
                title="[bold green]Options[/bold green]",
                title_align="left",
                border_style="green",
            )
        )
        console.print()

        # Commands
        all_items = list(self._commands.keys()) + list(self._groups.keys())
        if all_items:
            command_lines = []
            max_len = max(len(name) for name in all_items) + 4

            for cmd_name, (func, _) in self._commands.items():
                help_text = (func.__doc__ or "").strip().split("\n")[0]
                padding = " " * (max_len - len(cmd_name) - 2)
                command_lines.append(
                    f"  [magenta]{cmd_name}[/magenta]{padding}[dim]{help_text}[/dim]"
                )

            for group_name, sub_app in self._groups.items():
                help_text = sub_app._help or ""
                padding = " " * (max_len - len(group_name) - 2)
                command_lines.append(
                    f"  [magenta]{group_name}[/magenta]{padding}[dim]{help_text}[/dim]"
                )

            console.print(
                Panel(
                    "\n".join(command_lines),
                    title="[bold magenta]Commands[/bold magenta]",
                    title_align="left",
                    border_style="magenta",
                )
            )

    def _show_command_help(self, console: Console, name: str, schema: type | None) -> None:
        """Show help for a specific command."""
        func, _ = self._commands[name]
        group_name = self._name or "app"

        # Usage
        console.print(
            Panel(
                f"[bold]{group_name} {name}[/bold] [cyan][OPTIONS][/cyan]",
                title="[bold blue]Usage[/bold blue]",
                border_style="blue",
            )
        )
        console.print()

        if func.__doc__:
            console.print(func.__doc__.strip())
            console.print()

        console.print("Override options with [cyan]key=value[/cyan]")
        console.print()

        # Options
        if self._parent:
            options_text = (
                "  [cyan]-p[/cyan]  Print compiled config and exit\n"
                "  [cyan]-g[/cyan]  Print root config (global) and exit\n"
                "  [cyan]-h[/cyan]  Show this help"
            )
        else:
            options_text = (
                "  [cyan]-p[/cyan]  Print compiled config and exit\n"
                "  [cyan]-g[/cyan]  Print root config (global) and exit\n"
                "  [cyan]-h[/cyan]  Show this help"
            )
        console.print(
            Panel(
                options_text,
                title="[bold green]Options[/bold green]",
                title_align="left",
                border_style="green",
            )
        )
        console.print()

        # Config
        if schema:
            _, config_opts = get_schema_structure(schema)
            if config_opts:
                config_lines = []
                lengths = []
                for name_opt, type_name, default, _ in config_opts:
                    default_str = f" = {default}" if default is not None else ""
                    lengths.append(len(f"  {name_opt}: {type_name}{default_str}"))

                max_len = max(lengths) if lengths else 0

                for i, (name_opt, type_name, default, help_text) in enumerate(config_opts):
                    default_str = f" [yellow]= {default}[/yellow]" if default is not None else ""
                    padding = " " * (max_len - lengths[i] + 2)
                    help_str = f"[dim]{help_text}[/dim]" if help_text else ""
                    config_lines.append(
                        f"  [cyan]{name_opt}[/cyan]: [green]{type_name}[/green]{default_str}{padding}{help_str}"
                    )

                console.print(
                    Panel(
                        "\n".join(config_lines),
                        title="[bold cyan]Config[/bold cyan]",
                        title_align="left",
                        border_style="cyan",
                    )
                )

    def command(
        self, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a command."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cmd_name = name or func.__name__

            # Infer schema from function signature
            from typing import get_type_hints

            hints = get_type_hints(func)
            schema = None
            for pname, ptype in hints.items():
                if pname != "return" and is_dataclass(ptype):
                    schema = ptype
                    break

            self._commands[cmd_name] = (func, schema)  # type: ignore[assignment]
            return func

        return decorator

    def group(self, name: str, help: str | None = None) -> "NanoCLI":
        """Create a nested group."""
        sub_app = NanoCLI(name=name, help=help, parent=self)
        self._groups[name] = sub_app
        return sub_app


def group(name: str | None = None, help: str | None = None) -> NanoCLI:
    """Create a command group (CLI entry point).

    This is the main entry point for creating a CLI application.

    Args:
        name: Name of the CLI application.
        help: Help text shown in CLI help.

    Returns:
        NanoCLI instance.

    Examples:
        >>> app = group(name="myapp")
        >>> isinstance(app, NanoCLI)
        True
        >>> @app.command()
        ... def train(cfg):
        ...     pass
        >>> "train" in app._commands
        True
    """
    return NanoCLI(name=name, help=help)


def run(
    schema_or_func: type[T] | Any,
    args: list[str] | None = None,
) -> T | Any:
    """Run a single-command CLI from a config schema or function.

    This provides a simple API for single-command CLIs without groups.

    Args:
        schema_or_func: A dataclass type (returns compiled config) or
            a callable (infers schema from type hints and executes).
        args: CLI arguments. Defaults to sys.argv[1:].

    Returns:
        Compiled config if schema provided, or function return value.

    Raises:
        ValueError: If function has no dataclass argument.

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Config:
        ...     name: str = "world"
        >>> cfg = run(Config, args=[])
        >>> cfg.name
        'world'
    """
    from typing import get_type_hints

    console = Console()

    # Detect mode: Schema or Function
    func_to_run = None
    schema = schema_or_func

    if not isinstance(schema_or_func, type) and callable(schema_or_func):
        func_to_run = schema_or_func
        # Infer schema from function signature
        hints = get_type_hints(func_to_run)
        found_schema = None
        for pname, ptype in hints.items():
            if pname != "return" and is_dataclass(ptype):
                found_schema = ptype
                break

        if not found_schema:
            raise ValueError(
                f"Could not infer config schema from function {func_to_run.__name__}. Ensure one argument is a dataclass."
            )

        schema = found_schema

    if args is None:
        args = sys.argv[1:]

    path_parts, overrides, flags = parse_args(args)

    # Handle help
    if flags["help"]:
        _show_run_help(console, schema)
        return None

    # Load base config
    cfg_file = flags["cfg"]
    base = load_yaml(cfg_file) if cfg_file else None

    # Compile config
    try:
        config = compile_config(base=base, overrides=overrides, schema=schema)
    except ConfigError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print()
        _show_run_help(console, schema)
        sys.exit(1)

    # Handle print
    if flags["print"] or flags["print_global"]:
        console.print(to_yaml(config), end="")
        return config

    # Execute function if provided
    if func_to_run:
        return func_to_run(config)

    return config


def _show_run_help(console: Console, schema: type) -> None:
    """Show help for run() API."""
    name = "app"

    # Usage
    console.print(
        Panel(
            f"[bold]{name}[/bold] [cyan][OPTIONS][/cyan]",
            title="[bold blue]Usage[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    # Schema docstring
    if schema.__doc__:
        console.print(f"{schema.__name__} - {schema.__doc__.strip()}")
        console.print()

    console.print("Override options with [cyan]key=value[/cyan]")
    console.print("Load YAML with [cyan]-c[/cyan], subtree with [cyan]key=@file.yml[/cyan]")
    console.print()

    # Options
    options_text = (
        "  [cyan]-c, --cfg PATH[/cyan]  Load config from YAML file\n"
        "  [cyan]-p[/cyan]              Print compiled config and exit\n"
        "  [cyan]-h[/cyan]              Show this help"
    )
    console.print(
        Panel(
            options_text,
            title="[bold green]Options[/bold green]",
            title_align="left",
            border_style="green",
        )
    )
    console.print()

    # Config
    _, config_opts = get_schema_structure(schema)
    if config_opts:
        config_lines = []
        lengths = []
        for name_opt, type_name, default, _ in config_opts:
            default_str = f" = {default}" if default is not None else ""
            lengths.append(len(f"  {name_opt}: {type_name}{default_str}"))

        max_len = max(lengths) if lengths else 0

        for i, (name_opt, type_name, default, help_text) in enumerate(config_opts):
            default_str = f" [yellow]= {default}[/yellow]" if default is not None else ""
            padding = " " * (max_len - lengths[i] + 2)
            help_str = f"[dim]{help_text}[/dim]" if help_text else ""
            config_lines.append(
                f"  [cyan]{name_opt}[/cyan]: [green]{type_name}[/green]{default_str}{padding}{help_str}"
            )

        console.print(
            Panel(
                "\n".join(config_lines),
                title="[bold cyan]Config[/bold cyan]",
                title_align="left",
                border_style="cyan",
            )
        )
