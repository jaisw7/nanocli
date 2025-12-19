.PHONY: install dev test lint format type-check pre-commit clean help docs docs-serve docs-deploy

# Default target
help:
	@echo "NanoCLI Development Commands"
	@echo ""
	@echo "  make install     - Install package in development mode"
	@echo "  make dev         - Install with dev dependencies"
	@echo "  make test        - Run tests"
	@echo "  make test-cov    - Run tests with coverage"
	@echo "  make lint        - Run linter"
	@echo "  make lint-fix    - Run linter with auto-fix"
	@echo "  make format      - Format code"
	@echo "  make type-check  - Run type checker"
	@echo "  make pre-commit  - Run format, lint-fix, type-check"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make docs        - Build docs"
	@echo "  make docs-serve  - Serve docs locally"
	@echo "  make docs-deploy - Deploy docs to GitHub Pages"
	@echo ""

# Install package
install:
	uv sync

# Install with dev dependencies
dev:
	uv sync --all-extras
	pre-commit install

# Run tests
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=nanocli --cov-report=term-missing

# Run linter
lint:
	uv run ruff check src tests

# Run linter with auto-fix
lint-fix:
	uv run ruff check --fix src tests

# Format code
format:
	uv run ruff format --diff src tests

# Format code
format-fix:
	uv run ruff format src tests

# Run type checker
type-check:
	uv run mypy src/nanocli

# Run all pre-commit checks
pre-commit:
	uv run pre-commit run --all-files

# Build docs
docs:
	uv run mkdocs build --strict

# Serve docs locally
docs-serve:
	uv run mkdocs serve --watch docs --watch src

docs-deploy:
	uv run mkdocs gh-deploy --force

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
