# Development Setup

Guide for setting up a development environment for aind-zarr-utils.

## Prerequisites

- **Python 3.10+** (recommended: 3.11 or 3.12)
- **Git** for version control
- **uv** (recommended) or pip for package management

### Installing uv

```bash
# Install uv (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

## Development Installation

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/aind-zarr-utils.git
cd aind-zarr-utils

# Add upstream remote
git remote add upstream https://github.com/AllenNeuralDynamics/aind-zarr-utils.git
```

### 2. Set Up Environment

#### Using uv (Recommended)

```bash
# Create and activate virtual environment with all dependencies
uv sync

# Activate the environment (if not done automatically)
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

#### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install in development mode with all dependencies
pip install -e .[dev]
```

### 3. Verify Installation

```bash
# Run tests to verify setup
uv run pytest
# or
pytest

# Check linting
uv run ruff check
# or
ruff check

# Verify imports work
python -c "import aind_zarr_utils; print('✓ Installation successful')"
```

## Development Workflow

### Branch Management

```bash
# Always start from main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Code Quality Tools

The project uses several tools to maintain code quality:

#### Linting and Formatting

```bash
# Format code
uv run ruff format
# or
ruff format

# Check linting issues
uv run ruff check
# or
ruff check

# Auto-fix linting issues where possible
uv run ruff check --fix
# or
ruff check --fix
```

#### Type Checking

```bash
# Run type checking
uv run mypy
# or
mypy src/aind_zarr_utils
```

#### All Checks at Once

```bash
# Run all linting and checks
./scripts/run_linters_and_checks.sh

# Run all checks including tests and coverage
./scripts/run_linters_and_checks.sh -c
```

### Testing

#### Running Tests

```bash
# Run all tests
uv run pytest
# or
pytest

# Run specific test file
uv run pytest tests/test_zarr.py

# Run with coverage
uv run pytest --cov=aind_zarr_utils --cov-report=html

# Run tests in parallel (faster)
uv run pytest -n auto
```

#### Test Structure

```
tests/
├── test_annotations.py     # Point transformation tests
├── test_json_utils.py      # JSON loading tests
├── test_neuroglancer.py    # Neuroglancer processing tests
├── test_zarr.py           # ZARR conversion tests
├── test_pipeline_*.py     # Pipeline-specific tests
├── test_s3_cache.py       # S3 caching tests
└── conftest.py            # Test configuration and fixtures
```

#### Writing Tests

```python
# Example test structure
import pytest
import numpy as np
from aind_zarr_utils.zarr import zarr_to_ants

def test_zarr_conversion():
    """Test ZARR to ANTs conversion."""
    # Arrange
    zarr_uri = "test_data/sample.zarr"
    metadata = {"acquisition": {...}}
    
    # Act
    result = zarr_to_ants(zarr_uri, metadata, level=3)
    
    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result.spacing, expected_spacing)
    
@pytest.mark.parametrize("level,expected_size", [
    (0, (1000, 1000, 500)),
    (1, (500, 500, 250)),
    (2, (250, 250, 125)),
])
def test_zarr_levels(level, expected_size):
    """Test different resolution levels."""
    # Test implementation...
```

### Documentation

#### Building Documentation

```bash
# Install documentation dependencies
uv sync --group docs

# Build documentation
uv run --group docs sphinx-build -b html docs/source docs/build/html

# Build with live reload during development
uv run --group docs sphinx-autobuild docs/source docs/build/html
```

#### Documentation Structure

```
docs/
├── source/
│   ├── api-reference/       # Auto-generated API docs
│   ├── user-guide/         # User guides and tutorials
│   ├── getting-started/    # Installation and quickstart
│   ├── contributing/       # Development guides
│   └── reference/          # Reference materials
└── build/                  # Generated HTML (not in git)
```

#### Writing Documentation

- Use **NumPy-style docstrings** for all functions
- Include **examples** in docstrings where helpful
- **Cross-reference** related functions using `:func:` and `:class:`
- Use **MyST Markdown** for user guides

```python
def example_function(param1: str, param2: int = 5) -> bool:
    """Brief description of the function.
    
    Longer description explaining the purpose, behavior, and important
    details about the function.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default 5
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Examples
    --------
    >>> result = example_function("test", 10)
    >>> print(result)
    True
    
    See Also
    --------
    related_function : Brief description
    """
    return True
```

## Development Environment

### IDE Setup

#### VS Code

Recommended extensions:
- Python
- Pylance
- Ruff
- MyST-Markdown

#### Settings (`.vscode/settings.json`)

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

### Environment Variables

For development with S3 access:

```bash
# Optional: AWS credentials for private S3 access
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# Optional: Custom cache directory
export AIND_ZARR_CACHE_DIR=/path/to/fast/storage
```

### Docker Development

For containerized development:

```dockerfile
# Dockerfile.dev
FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Install in development mode
RUN uv sync

# Default command
CMD ["bash"]
```

```bash
# Build and run development container
docker build -f Dockerfile.dev -t aind-zarr-utils-dev .
docker run -it -v $(pwd):/workspace aind-zarr-utils-dev
```

## Contributing Workflow

### 1. Make Changes

```bash
# Make your changes
git add .
git commit -m "feat: add new functionality"

# Follow conventional commit format:
# feat: new feature
# fix: bug fix
# docs: documentation changes
# test: adding tests
# refactor: code refactoring
# style: formatting changes
# chore: maintenance tasks
```

### 2. Run Quality Checks

```bash
# Format code
uv run ruff format

# Check and fix linting issues
uv run ruff check --fix

# Run type checking
uv run mypy

# Run tests
uv run pytest

# Or run all checks at once
./scripts/run_linters_and_checks.sh -c
```

### 3. Update Documentation

```bash
# If you added new functions, update docstrings
# Build docs to verify formatting
uv run --group docs sphinx-build -b html docs/source docs/build/html

# Check for documentation warnings
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
# Fill out the PR template with:
# - Description of changes
# - Testing performed
# - Breaking changes (if any)
```

## Debugging

### Common Development Issues

#### Import Errors

```bash
# Ensure package is installed in development mode
pip install -e .
# or
uv sync

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Test Failures

```bash
# Run specific failing test with verbose output
uv run pytest tests/test_zarr.py::test_specific_function -v -s

# Run with debugger on failure
uv run pytest --pdb tests/test_zarr.py

# Check test coverage
uv run pytest --cov=aind_zarr_utils --cov-report=term-missing
```

#### S3 Connection Issues

```bash
# Test S3 connectivity
python -c "
from aind_s3_cache.json_utils import get_json
try:
    data = get_json('s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/metadata.json')
    print('✓ S3 access working')
except Exception as e:
    print(f'✗ S3 error: {e}')
"
```

### Performance Profiling

```python
# Profile ZARR loading performance
import cProfile
from aind_zarr_utils.zarr import zarr_to_ants

def profile_zarr_loading():
    zarr_uri = "s3://aind-open-data/dataset/data.ome.zarr/0"
    metadata = {...}
    zarr_to_ants(zarr_uri, metadata, level=3)

# Run profiler
cProfile.run('profile_zarr_loading()', 'profile_output.prof')

# Analyze results
python -m pstats profile_output.prof
```

## Release Process

### Version Management

The project uses semantic versioning:

```bash
# Check current version
python -c "import aind_zarr_utils; print(aind_zarr_utils.__version__)"

# Version is managed in src/aind_zarr_utils/__init__.py
```

### Creating a Release

1. **Update version** in `src/aind_zarr_utils/__init__.py`
2. **Update CHANGELOG** with new features and fixes
3. **Create release commit**:
   ```bash
   git commit -m "release: v0.2.0"
   git tag v0.2.0
   ```
4. **Push to main**:
   ```bash
   git push upstream main --tags
   ```
5. **Create GitHub release** from the tag

## Getting Help

- **Documentation**: Read the user guides and API reference
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Request review from maintainers

### Maintainer Contact

- Primary maintainer: [Name] (@github-username)
- Team: Allen Institute Neural Dynamics
- Email: [team-email]

## Code Style Guide

### Python Conventions

- **Line length**: 79 characters (enforced by ruff)
- **Imports**: Organize with isort (handled by ruff)
- **Docstrings**: NumPy style for all public functions
- **Type hints**: Required for all public APIs
- **Variable naming**: Use descriptive names

### Error Handling

```python
# Good: Specific error messages
if not os.path.exists(path):
    raise FileNotFoundError(f"ZARR file not found: {path}")

# Good: Type checking
if not isinstance(level, int):
    raise TypeError(f"level must be int, got {type(level)}")

# Good: Value validation
if level < 0:
    raise ValueError(f"level must be non-negative, got {level}")
```

### Function Design

```python
# Good: Clear function signature with defaults
def zarr_to_ants(
    zarr_uri: str,
    metadata: dict,
    level: int = 3,
    scale_unit: str = "millimeter"
) -> "ants.core.ants_image.ANTsImage":
    """Convert ZARR to ANTs image.
    
    Parameters
    ----------
    zarr_uri : str
        URI to ZARR dataset
    metadata : dict
        ZARR metadata containing acquisition info
    level : int, default 3
        Resolution level to load (higher = lower resolution)
    scale_unit : str, default "millimeter"  
        Physical units for output spacing
        
    Returns
    -------
    ants.core.ants_image.ANTsImage
        Image in LPS coordinate system
    """
```

This development setup ensures consistent code quality and makes it easy for contributors to get started with the project.