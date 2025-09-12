# Installation

This guide covers installing aind-zarr-utils and its dependencies.

## Requirements

- **Python**: 3.10 or later (recommended: 3.11+)
- **Operating System**: Linux, macOS, Windows
- **Memory**: 4GB+ RAM recommended for ZARR processing

## Quick Install

### Using pip (Recommended)

```bash
pip install aind-zarr-utils
```

### Using uv (Development)

For development or if you prefer uv:

```bash
uv add aind-zarr-utils
```

## Installation Options

### Standard Installation

The basic installation includes core functionality:

```bash
pip install aind-zarr-utils
```

**Includes**:
- ZARR to SimpleITK/ANTs conversion
- JSON utilities (S3, HTTP, local files)
- Basic coordinate transformations
- Neuroglancer annotation processing

### Development Installation

For contributing or advanced usage:

```bash
# Clone the repository
git clone https://github.com/AllenNeuralDynamics/aind-zarr-utils.git
cd aind-zarr-utils

# Install with development dependencies
pip install -e .[dev]

# Or using uv
uv sync
```

**Additional includes**:
- Testing framework (pytest)
- Linting tools (ruff, mypy)
- Documentation tools (sphinx)
- Code coverage analysis

## Dependencies

### Core Dependencies

These are installed automatically:

- **numpy**: Numerical computing
- **ome-zarr**: ZARR file format support
- **requests**: HTTP requests
- **boto3**: AWS S3 integration
- **SimpleITK**: Medical image processing
- **antspyx**: Advanced registration and processing

### Optional Dependencies

For specific use cases:

```bash
# Enhanced S3 support
pip install s3fs

# Jupyter notebook support
pip install jupyter ipywidgets

# Additional neuroimaging tools
pip install nibabel
```

## Verification

Test your installation:

```python
import aind_zarr_utils
from aind_zarr_utils.json_utils import get_json
from aind_zarr_utils.zarr import zarr_to_sitk

# Test basic functionality
print(f"aind-zarr-utils version: {aind_zarr_utils.__version__}")

# Test S3 connectivity (anonymous)
try:
    metadata = get_json("s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/metadata.json")
    print("✓ S3 access working")
except Exception as e:
    print(f"⚠ S3 access issue: {e}")

# Test ZARR processing
try:
    import SimpleITK as sitk
    print("✓ SimpleITK available")
except ImportError:
    print("⚠ SimpleITK not available")

try:
    import ants
    print("✓ ANTs available")
except ImportError:
    print("⚠ ANTs not available")
```

## Platform-Specific Notes

### Linux

Standard installation works out of the box:

```bash
pip install aind-zarr-utils
```

### macOS

May need additional system libraries:

```bash
# Install system dependencies (if using Homebrew)
brew install cmake

# Then install normally
pip install aind-zarr-utils
```

### Windows

Recommended to use conda for better dependency management:

```bash
# Create conda environment
conda create -n zarr-utils python=3.11
conda activate zarr-utils

# Install via pip
pip install aind-zarr-utils
```

## Docker Installation

For containerized environments:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install aind-zarr-utils
RUN pip install aind-zarr-utils

# Your application code
COPY . /app
WORKDIR /app
```

## AWS Configuration

For S3 access beyond public data:

### Option 1: AWS CLI Configuration

```bash
aws configure
# Enter your AWS credentials
```

### Option 2: Environment Variables

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

### Option 3: IAM Roles (EC2/ECS)

No additional configuration needed when running on AWS with appropriate IAM roles.

## Troubleshooting

### Common Issues

#### ImportError: No module named 'SimpleITK'

```bash
# Reinstall SimpleITK explicitly
pip install --upgrade SimpleITK
```

#### ANTs installation issues

```bash
# Try installing antspyx with specific version
pip install antspyx==0.3.8
```

#### S3 connection timeouts

```python
import boto3
from botocore.config import Config

# Configure custom timeouts
config = Config(
    read_timeout=60,
    connect_timeout=30,
    retries={'max_attempts': 3}
)

# Use with S3 operations
```

#### Memory issues with large ZARR files

```python
# Use higher resolution levels (smaller files)
from aind_zarr_utils.zarr import zarr_to_sitk

# Instead of level=0 (full resolution)
img = zarr_to_sitk(uri, metadata, level=3)  # More manageable size
```

### Getting Help

If you encounter issues:

1. **Check the troubleshooting guide**: See common solutions
2. **Review the logs**: Most functions provide detailed error messages
3. **File an issue**: [GitHub Issues](https://github.com/AllenNeuralDynamics/aind-zarr-utils/issues)
4. **Contact support**: Include error messages and environment details

## Next Steps

- **Quick Start**: Jump into basic usage examples
- **User Guides**: Comprehensive feature documentation
- **API Reference**: Detailed function documentation
- **Examples**: Real-world usage scenarios

## Version Compatibility

| aind-zarr-utils | Python | SimpleITK | ANTs |
|----------------|--------|-----------|------|
| 0.1.x          | 3.10+  | 2.2+      | 0.3+ |
| 0.2.x          | 3.10+  | 2.3+      | 0.3+ |
| latest         | 3.10+  | 2.3+      | 0.3+ |

Always use the latest version for best performance and features.