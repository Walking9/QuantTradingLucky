.PHONY: help install install-uv lint format typecheck test test-fast cov notebook clean precommit bootstrap

# Default target shows the help message.
help:
	@echo "QuantTradingLucky — Developer commands"
	@echo ""
	@echo "  make bootstrap    - First-time setup: install uv + deps + pre-commit hooks"
	@echo "  make install      - Sync dependencies with uv"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Format code + auto-fix lint issues"
	@echo "  make typecheck    - Run mypy"
	@echo "  make test         - Run full test suite with coverage"
	@echo "  make test-fast    - Run tests excluding slow + integration"
	@echo "  make cov          - Open HTML coverage report (after test)"
	@echo "  make notebook     - Launch Jupyter Lab"
	@echo "  make precommit    - Run all pre-commit hooks against all files"
	@echo "  make clean        - Remove caches and build artifacts"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}

install:
	uv sync --all-extras --dev

bootstrap: install-uv install
	uv run pre-commit install
	@echo ""
	@echo "✅ Bootstrap complete. Next steps:"
	@echo "   1. cp .env.example .env  (and fill credentials)"
	@echo "   2. make test             (verify setup)"
	@echo "   3. make notebook         (start exploring)"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

precommit:
	uv run pre-commit run --all-files

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow and not integration and not live"

cov:
	uv run coverage html
	@echo "Open htmlcov/index.html in your browser"

# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------
notebook:
	uv run jupyter lab

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
