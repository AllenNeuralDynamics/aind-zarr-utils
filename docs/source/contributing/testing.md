# Testing Guide

Comprehensive guide to testing aind-zarr-utils.

## Testing Philosophy

aind-zarr-utils uses a comprehensive testing strategy:

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test interactions between components
- **End-to-end tests**: Test complete workflows with real data
- **Property-based tests**: Test invariants across input ranges

## Test Framework

### Tools Used

- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **hypothesis**: Property-based testing (where applicable)

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_annotations.py           # Point transformation tests
├── test_json_utils.py             # JSON loading tests  
├── test_neuroglancer.py           # Neuroglancer processing tests
├── test_zarr.py                   # ZARR conversion tests
├── test_pipeline_domain_selector.py  # Domain correction tests
├── test_pipeline_transformed.py   # Pipeline transformation tests
├── test_s3_cache.py               # S3 caching tests
├── test_uri_utils.py              # URI manipulation tests
├── data/                          # Test data files
│   ├── sample_metadata.json
│   ├── sample_neuroglancer.json
│   └── mock_zarr/
└── integration/                   # Integration tests
    ├── test_s3_integration.py
    └── test_end_to_end.py
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
uv run pytest
# or
pytest

# Run specific test file
uv run pytest tests/test_zarr.py

# Run specific test function
uv run pytest tests/test_zarr.py::test_zarr_to_ants

# Run tests matching pattern
uv run pytest -k "test_zarr"
```

### Test Options

```bash
# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x

# Run in parallel (faster)
uv run pytest -n auto

# Run only failed tests from last run
uv run pytest --lf

# Run tests that failed or were not run
uv run pytest --ff
```

### Coverage Testing

```bash
# Run with coverage
uv run pytest --cov=aind_zarr_utils

# Generate HTML coverage report
uv run pytest --cov=aind_zarr_utils --cov-report=html

# Check specific coverage threshold
uv run pytest --cov=aind_zarr_utils --cov-fail-under=30

# Show missing lines
uv run pytest --cov=aind_zarr_utils --cov-report=term-missing
```

## Writing Tests

### Test Organization

Each module has corresponding test file:

```python
# tests/test_zarr.py
import pytest
import numpy as np
from aind_zarr_utils.zarr import zarr_to_ants, zarr_to_sitk

class TestZarrConversion:
    """Test ZARR conversion functions."""
    
    def test_zarr_to_ants_basic(self, sample_zarr_uri, sample_metadata):
        """Test basic ZARR to ANTs conversion."""
        result = zarr_to_ants(sample_zarr_uri, sample_metadata, level=3)
        
        assert hasattr(result, 'shape')
        assert hasattr(result, 'spacing')
        assert hasattr(result, 'origin')
        
    def test_zarr_to_ants_scale_units(self, sample_zarr_uri, sample_metadata):
        """Test different scale units."""
        mm_result = zarr_to_ants(sample_zarr_uri, sample_metadata, scale_unit="millimeter")
        um_result = zarr_to_ants(sample_zarr_uri, sample_metadata, scale_unit="micrometer")
        
        # Spacing should differ by factor of 1000
        np.testing.assert_allclose(
            np.array(mm_result.spacing) * 1000,
            np.array(um_result.spacing),
            rtol=1e-10
        )
    
    @pytest.mark.parametrize("level,expected_min_size", [
        (0, 1000),  # Full resolution
        (1, 500),   # Half resolution
        (2, 250),   # Quarter resolution
        (3, 125),   # Eighth resolution
    ])
    def test_zarr_levels(self, sample_zarr_uri, sample_metadata, level, expected_min_size):
        """Test different resolution levels."""
        result = zarr_to_ants(sample_zarr_uri, sample_metadata, level=level)
        
        # Check that size is approximately what we expect
        assert min(result.shape) >= expected_min_size // 2
        assert max(result.shape) >= expected_min_size
```

### Test Fixtures

Common test fixtures are defined in `conftest.py`:

```python
# tests/conftest.py
import pytest
import json
import tempfile
from pathlib import Path

@pytest.fixture
def sample_metadata():
    """Sample ZARR metadata for testing."""
    return {
        "session_id": "test_session_123",
        "subject_id": "test_subject",
        "acquisition": {
            "acquisition_datetime": "2024-01-01T12:00:00",
            "instrument": "ExaSPIM",
            "voxel_size": [7.2, 7.2, 8.0],  # micrometers
            "coordinate_transformations": [
                {
                    "type": "scale",
                    "scale": [0.0072, 0.0072, 0.008]  # millimeters
                }
            ]
        }
    }

@pytest.fixture
def sample_neuroglancer_data():
    """Sample Neuroglancer state for testing."""
    return {
        "layers": {
            "test_layer": {
                "annotations": [
                    {"point": [100, 200, 150], "description": "test point 1"},
                    {"point": [120, 180, 160], "description": "test point 2"}
                ]
            }
        }
    }

@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope="session")
def sample_zarr_uri():
    """URI to test ZARR dataset."""
    # Use public AIND data for integration tests
    return "s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/exaspim.ome.zarr/0"

@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    from unittest.mock import Mock
    client = Mock()
    # Configure mock behavior
    return client
```

### Test Categories

#### Unit Tests

Test individual functions in isolation:

```python
def test_parse_s3_uri():
    """Test S3 URI parsing."""
    from aind_s3_cache.uri_utils import parse_s3_uri
    
    # Test valid URI
    bucket, key = parse_s3_uri("s3://my-bucket/path/to/file.json")
    assert bucket == "my-bucket"
    assert key == "path/to/file.json"
    
    # Test invalid URI
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        parse_s3_uri("not-s3-uri")

def test_annotation_coordinate_transform():
    """Test coordinate transformation logic."""
    from aind_zarr_utils.annotations import _transform_indices_to_physical
    
    indices = np.array([[100, 200, 50]])
    spacing = (0.01, 0.01, 0.02)  # mm
    origin = (0.0, 0.0, 0.0)
    
    physical = _transform_indices_to_physical(indices, spacing, origin)
    
    expected = np.array([[1.0, 2.0, 1.0]])  # mm
    np.testing.assert_allclose(physical, expected)
```

#### Integration Tests

Test component interactions:

```python
@pytest.mark.integration
def test_neuroglancer_to_anatomical_workflow(sample_zarr_uri, sample_metadata, sample_neuroglancer_data):
    """Test complete Neuroglancer to anatomical workflow."""
    from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical
    
    # This tests the interaction between:
    # - URI parsing
    # - ZARR metadata loading  
    # - Coordinate transformations
    # - Neuroglancer data processing
    
    physical_coords, descriptions = neuroglancer_annotations_to_anatomical(
        sample_neuroglancer_data, sample_zarr_uri, sample_metadata
    )
    
    assert "test_layer" in physical_coords
    assert physical_coords["test_layer"].shape == (2, 3)  # 2 points, 3D coordinates
    assert len(descriptions["test_layer"]) == 2
```

#### End-to-End Tests

Test complete workflows with real data:

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_complete_pipeline_workflow():
    """Test complete pipeline workflow with real data."""
    # This test uses actual S3 data and tests the full pipeline
    
    dataset_uri = "s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44"
    
    # Load metadata
    from aind_s3_cache.json_utils import get_json
    metadata = get_json(f"{dataset_uri}/metadata.json")
    
    # Convert ZARR to image
    from aind_zarr_utils.zarr import zarr_to_ants
    zarr_uri = f"{dataset_uri}/exaspim.ome.zarr/0"
    ants_img = zarr_to_ants(zarr_uri, metadata, level=3)
    
    # Verify result properties
    assert ants_img.shape[0] > 100  # Reasonable size
    assert all(s > 0 for s in ants_img.spacing)  # Positive spacing
    
    # Test coordinate system consistency
    # ... additional verification
```

### Property-Based Testing

Test invariants across input ranges:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    indices=npst.arrays(
        dtype=np.float64,
        shape=(st.integers(1, 100), 3),  # 1-100 points, 3D
        elements=st.floats(0, 1000, allow_nan=False)
    ),
    spacing=st.tuples(
        st.floats(0.001, 1.0),  # Reasonable spacing values
        st.floats(0.001, 1.0),
        st.floats(0.001, 1.0)
    )
)
def test_coordinate_transform_properties(indices, spacing):
    """Test coordinate transformation properties."""
    from aind_zarr_utils.annotations import _transform_indices_to_physical
    
    origin = (0.0, 0.0, 0.0)
    physical = _transform_indices_to_physical(indices, spacing, origin)
    
    # Properties that should always hold:
    # 1. Output shape matches input shape
    assert physical.shape == indices.shape
    
    # 2. Origin maps to origin
    origin_physical = _transform_indices_to_physical(
        np.array([[0, 0, 0]]), spacing, origin
    )
    np.testing.assert_allclose(origin_physical, [[0, 0, 0]])
    
    # 3. Scaling is linear
    scaled_indices = indices * 2
    scaled_physical = _transform_indices_to_physical(scaled_indices, spacing, origin)
    expected_scaled = physical * 2
    np.testing.assert_allclose(scaled_physical, expected_scaled, rtol=1e-10)
```

## Test Data Management

### Test Data Strategy

- **Small synthetic data**: Created in tests for unit testing
- **Real data samples**: Use public AIND datasets for integration tests
- **Mock data**: Mock external services (S3, network) for isolation

### Creating Test Data

```python
# Create minimal test ZARR structure
def create_test_zarr(temp_dir):
    """Create a minimal ZARR for testing."""
    import zarr
    
    zarr_path = Path(temp_dir) / "test.zarr"
    
    # Create multiscale ZARR
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.group(store=store)
    
    # Level 0: 100x100x50
    level0 = root.create_dataset(
        "0", 
        shape=(50, 100, 100),  # ZYX order
        chunks=(10, 50, 50),
        dtype=np.uint16
    )
    level0[:] = np.random.randint(0, 1000, size=(50, 100, 100))
    
    # Level 1: 50x50x25
    level1 = root.create_dataset(
        "1",
        shape=(25, 50, 50),
        chunks=(10, 25, 25), 
        dtype=np.uint16
    )
    level1[:] = np.random.randint(0, 1000, size=(25, 50, 50))
    
    # Add metadata
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ],
        "datasets": [
            {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [0.008, 0.0072, 0.0072]}]},
            {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [0.016, 0.0144, 0.0144]}]}
        ],
        "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]
    }]
    
    return str(zarr_path)
```

### Mocking External Services

```python
@pytest.fixture
def mock_s3_responses():
    """Mock S3 responses for testing."""
    import responses
    
    with responses.RequestsMock() as rsps:
        # Mock metadata file
        rsps.add(
            responses.GET,
            "https://aind-open-data.s3.amazonaws.com/test/metadata.json",
            json={"session_id": "test", "acquisition": {}},
            status=200
        )
        
        yield rsps

def test_json_loading_with_mock(mock_s3_responses):
    """Test JSON loading with mocked S3."""
    from aind_s3_cache.json_utils import get_json
    
    data = get_json("s3://aind-open-data/test/metadata.json")
    assert data["session_id"] == "test"
```

## Test Configuration

### pytest Configuration

```ini
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    --strict-markers
    --strict-config
    --cov=aind_zarr_utils
    --cov-report=term-missing:skip-covered
    --cov-fail-under=30
testpaths = tests
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (moderate speed) 
    e2e: End-to-end tests (slow, uses real data)
    slow: Slow tests (long running)
    network: Tests requiring network access
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
```

### Test Environment

```bash
# Environment variables for testing
export PYTEST_CURRENT_TEST=1
export AWS_DEFAULT_REGION=us-west-2

# For testing with real S3 data (optional)
export TEST_WITH_REAL_S3=1
export TEST_S3_BUCKET=aind-test-data
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Install dependencies
      run: uv sync
    
    - name: Run tests
      run: uv run pytest --cov=aind_zarr_utils --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Categories in CI

```bash
# Fast tests (run on every commit)
uv run pytest -m "not slow and not e2e"

# Integration tests (run on PR)
uv run pytest -m "integration"

# Full test suite (run on main branch)
uv run pytest
```

## Debugging Tests

### Common Debug Techniques

```bash
# Run with debugger on failure
uv run pytest --pdb

# Run specific test with output
uv run pytest tests/test_zarr.py::test_specific -s -v

# Run with Python debugger
uv run pytest tests/test_zarr.py -s --capture=no --pdb

# Show local variables on failure
uv run pytest --tb=long -v
```

### Debug-Friendly Test Writing

```python
def test_coordinate_transformation_debug():
    """Example of debug-friendly test."""
    # Arrange - with debug info
    input_indices = np.array([[100, 200, 50]])
    spacing = (0.01, 0.01, 0.02)
    print(f"Debug - Input indices: {input_indices}")
    print(f"Debug - Spacing: {spacing}")
    
    # Act
    result = transform_coordinates(input_indices, spacing)
    print(f"Debug - Result: {result}")
    
    # Assert with clear error messages
    expected = np.array([[1.0, 2.0, 1.0]])
    np.testing.assert_allclose(
        result, expected,
        err_msg=f"Expected {expected}, got {result}, diff: {result - expected}"
    )
```

## Performance Testing

### Benchmark Tests

```python
import time
import pytest

@pytest.mark.benchmark
def test_zarr_loading_performance(sample_zarr_uri, sample_metadata):
    """Benchmark ZARR loading performance."""
    from aind_zarr_utils.zarr import zarr_to_ants
    
    start_time = time.time()
    result = zarr_to_ants(sample_zarr_uri, sample_metadata, level=3)
    load_time = time.time() - start_time
    
    # Assert reasonable performance
    assert load_time < 30.0, f"Loading took {load_time:.2f}s, expected < 30s"
    
    print(f"ZARR loading performance: {load_time:.2f}s for {result.shape}")
```

### Memory Usage Tests

```python
import psutil
import os

def test_memory_usage():
    """Test memory usage during ZARR processing."""
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline_memory = process.memory_info().rss / 1024**2  # MB
    
    # Perform operation
    from aind_zarr_utils.zarr import zarr_to_ants
    result = zarr_to_ants(sample_zarr_uri, sample_metadata, level=3)
    
    # Check peak memory
    peak_memory = process.memory_info().rss / 1024**2  # MB
    memory_increase = peak_memory - baseline_memory
    
    # Assert reasonable memory usage
    assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.1f}MB"
```

## Best Practices

### Test Writing Guidelines

1. **One concept per test**: Each test should verify one specific behavior
2. **Clear test names**: Names should describe what is being tested
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Independent tests**: Tests should not depend on each other
5. **Use fixtures**: Share setup code via fixtures

### Error Testing

```python
def test_error_conditions():
    """Test various error conditions."""
    from aind_zarr_utils.zarr import zarr_to_ants
    
    # Test missing file
    with pytest.raises(FileNotFoundError, match="ZARR file not found"):
        zarr_to_ants("nonexistent://path", {})
    
    # Test invalid metadata
    with pytest.raises(ValueError, match="metadata must contain"):
        zarr_to_ants("valid://path", {})
    
    # Test invalid level
    with pytest.raises(ValueError, match="level must be non-negative"):
        zarr_to_ants("valid://path", valid_metadata, level=-1)
```

### Test Maintenance

- **Update tests** when changing functionality
- **Remove obsolete tests** when removing features
- **Keep test data current** with real-world usage
- **Monitor test performance** and optimize slow tests
- **Review test coverage** regularly

The testing framework ensures aind-zarr-utils remains reliable and maintainable as it evolves.