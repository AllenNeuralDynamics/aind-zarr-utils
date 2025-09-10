# Bug Reporting Guide

Guide for reporting bugs and issues in aind-zarr-utils.

## Before Reporting a Bug

### 1. Search Existing Issues

Before creating a new issue, please search the [GitHub Issues](https://github.com/AllenNeuralDynamics/aind-zarr-utils/issues) to see if your problem has already been reported.

### 2. Check Documentation

Review the relevant documentation:
- [User Guides](../user-guide/index.md) for usage questions
- [API Reference](../api-reference/index.md) for function details
- [Troubleshooting](../reference/troubleshooting.md) for common issues

### 3. Verify Your Setup

Ensure your installation is correct:

```bash
# Check version
python -c "import aind_zarr_utils; print(aind_zarr_utils.__version__)"

# Test basic functionality
python -c "
from aind_zarr_utils.json_utils import get_json
try:
    data = get_json('s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/metadata.json')
    print('✓ Basic functionality working')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

## What Makes a Good Bug Report

### Essential Information

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Environment details** (Python version, OS, etc.)
5. **Minimal reproducible example**
6. **Error messages** and stack traces

### Bug Report Template

When creating a new issue, use this template:

```markdown
## Bug Description
Brief description of what went wrong.

## Steps to Reproduce
1. Step one
2. Step two  
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- **aind-zarr-utils version**: 0.x.x
- **Python version**: 3.11.x
- **Operating System**: Ubuntu 22.04 / macOS 13.x / Windows 11
- **Installation method**: pip / uv / conda

## Minimal Example
```python
# Minimal code that reproduces the issue
from aind_zarr_utils.zarr import zarr_to_ants

# Your code here
```

## Error Output
```
Full error message and stack trace
```

## Additional Context
Any other relevant information.
```

## Common Bug Categories

### 1. Installation Issues

**Symptoms**:
- ImportError when importing aind_zarr_utils
- Missing dependencies
- Version conflicts

**Information to Include**:
```bash
# Python environment info
python --version
pip list | grep -E "(aind|zarr|sitk|ants)"

# Installation method
pip show aind-zarr-utils

# System info
uname -a  # Linux/macOS
# or
systeminfo  # Windows
```

**Example**:
```markdown
## Bug Description
Cannot import aind_zarr_utils after installation

## Steps to Reproduce
1. `pip install aind-zarr-utils`
2. `python -c "import aind_zarr_utils"`

## Error Output
```
ImportError: No module named 'SimpleITK'
```

## Environment
- Python 3.11.5
- pip 23.2.1
- Ubuntu 22.04
```

### 2. ZARR Processing Issues

**Symptoms**:
- Incorrect image dimensions or spacing
- Memory errors with large files
- Coordinate system problems

**Information to Include**:
- ZARR URI or file path
- Metadata structure (if possible)
- Resolution level used
- Expected vs actual image properties

**Example**:
```markdown
## Bug Description
zarr_to_ants returns incorrect spacing values

## Minimal Example
```python
from aind_zarr_utils.zarr import zarr_to_ants
from aind_zarr_utils.json_utils import get_json

metadata = get_json("s3://bucket/metadata.json")
zarr_uri = "s3://bucket/data.ome.zarr/0"

result = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit="millimeter")
print(f"Got spacing: {result.spacing}")
```

## Expected Behavior
Spacing should be (0.0576, 0.0576, 0.064) mm

## Actual Behavior  
Spacing is (0.064, 0.0576, 0.0576) mm - values are swapped
```

### 3. S3 Access Issues

**Symptoms**:
- Connection timeouts
- Permission errors
- Authentication failures

**Information to Include**:
- S3 URI being accessed
- Whether bucket is public or private
- AWS configuration method
- Network environment (VPN, proxy, etc.)

**Example**:
```markdown
## Bug Description
S3 access fails with SSL error

## Minimal Example
```python
from aind_zarr_utils.json_utils import get_json
data = get_json("s3://aind-open-data/dataset/metadata.json")
```

## Error Output
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

## Environment
- Behind corporate firewall
- Python 3.11.5
- boto3 1.28.x
```

### 4. Coordinate System Issues

**Symptoms**:
- Points in wrong anatomical locations
- Incorrect transformations
- Inconsistent coordinate systems

**Information to Include**:
- Input coordinates and format
- Expected output coordinates
- Coordinate system assumptions
- Transformation chain used

**Example**:
```markdown
## Bug Description
Neuroglancer annotations transformed to incorrect anatomical locations

## Minimal Example
```python
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

ng_data = {
    "layers": {
        "test": {
            "annotations": [{"point": [100, 200, 150]}]
        }
    }
}

result, _ = neuroglancer_annotations_to_anatomical(
    ng_data, zarr_uri, metadata, scale_unit="millimeter"
)
print(f"Result: {result['test']}")
```

## Expected Behavior
Point should be in brain region, approximately (5.0, 10.0, 7.5) mm LPS

## Actual Behavior
Point is at (-5.0, -10.0, 7.5) mm LPS - X and Y signs are flipped
```

### 5. Performance Issues

**Symptoms**:
- Unexpectedly slow operations
- Memory usage spikes
- Hanging or frozen processes

**Information to Include**:
- Dataset size and type
- Operation being performed
- Performance expectations
- System specifications

**Example**:
```markdown
## Bug Description
zarr_to_ants takes extremely long for small datasets

## Minimal Example
```python
import time
from aind_zarr_utils.zarr import zarr_to_ants

start = time.time()
result = zarr_to_ants(zarr_uri, metadata, level=5)  # Lowest resolution
elapsed = time.time() - start
print(f"Took {elapsed:.2f}s for {result.shape}")
```

## Expected Behavior
Should complete in < 10 seconds for level 5

## Actual Behavior
Takes 3+ minutes for tiny (64, 64, 32) image

## Environment
- 16GB RAM
- SSD storage
- Good internet connection
```

## Gathering Debug Information

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here - will show detailed debug info
```

### Get System Information

```python
import platform
import sys
import aind_zarr_utils

print("=== System Information ===")
print(f"Platform: {platform.platform()}")
print(f"Python: {sys.version}")
print(f"aind-zarr-utils: {aind_zarr_utils.__version__}")

# Check key dependencies
try:
    import SimpleITK as sitk
    print(f"SimpleITK: {sitk.Version_VersionString()}")
except ImportError:
    print("SimpleITK: Not installed")

try:
    import ants
    print(f"ANTs: {ants.__version__}")
except ImportError:
    print("ANTs: Not installed")

try:
    import boto3
    print(f"boto3: {boto3.__version__}")
except ImportError:
    print("boto3: Not installed")
```

### Create Minimal Reproducible Example

```python
# Minimal example that reproduces the issue
# Remove all unnecessary code
# Use public data when possible

from aind_zarr_utils.zarr import zarr_to_ants
from aind_zarr_utils.json_utils import get_json

# Use public dataset
metadata = get_json("s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/metadata.json")
zarr_uri = "s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/exaspim.ome.zarr/0"

# Minimal operation that fails
result = zarr_to_ants(zarr_uri, metadata, level=3)
print(f"Problem: {result.spacing}")  # Shows the issue
```

## Sensitive Information

### What NOT to Include

- **AWS credentials** or access keys
- **Private S3 bucket names** or paths
- **Internal hostnames** or IP addresses
- **Personal data** or PHI

### Safe Alternatives

- Use **public AIND datasets** for examples
- **Redact sensitive paths**: `s3://my-bucket/...` → `s3://PRIVATE_BUCKET/...`
- **Mock credentials**: Show structure without real values
- **Generalize hostnames**: `internal.company.com` → `INTERNAL_HOST`

## Issue Triage Process

### Issue Labels

Issues are labeled for triage:

- **bug**: Confirmed bugs
- **enhancement**: Feature requests  
- **documentation**: Documentation issues
- **help wanted**: Good for contributors
- **priority-high**: Critical issues
- **priority-low**: Minor issues

### Response Times

- **Critical bugs**: 1-2 business days
- **Standard bugs**: 1 week
- **Enhancement requests**: 2-4 weeks
- **Documentation**: 1-2 weeks

### Resolution Process

1. **Triage**: Label and prioritize issue
2. **Investigation**: Reproduce and analyze
3. **Fix**: Implement solution
4. **Testing**: Verify fix works
5. **Release**: Include in next version

## Contributing Fixes

### Found the Problem?

If you can fix the bug yourself:

1. **Fork the repository**
2. **Create a branch**: `git checkout -b fix/issue-description`
3. **Write a test** that reproduces the bug
4. **Implement the fix**
5. **Verify the test passes**
6. **Submit a pull request**

### Pull Request Guidelines

- **Reference the issue**: "Fixes #123"
- **Include tests** for the fix
- **Update documentation** if needed
- **Follow code style** (ruff formatting)

## Feature Requests

### Enhancement vs Bug

- **Bug**: Something doesn't work as documented
- **Enhancement**: New functionality or improvement

### Enhancement Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why this feature would be valuable.

## Proposed API
```python
# Example of how the feature might work
new_function(param1, param2)
```

## Alternatives Considered
Other approaches you've considered.

## Additional Context
Any other relevant information.
```

## Getting Help

If you're unsure whether something is a bug:

1. **Ask in Discussions**: Use GitHub Discussions for questions
2. **Check Slack/Teams**: Internal users can ask on team channels
3. **Email maintainers**: For urgent issues

### Contact Information

- **GitHub Issues**: Primary bug reporting
- **GitHub Discussions**: Questions and feature discussions
- **Team Email**: [team-email] for urgent issues
- **Maintainer**: [@github-username] for direct contact

## Bug Report Quality

### Good Bug Reports Include:

✅ **Clear title**: "zarr_to_ants returns incorrect spacing for level 3"  
✅ **Minimal example**: Code that reproduces the issue  
✅ **Expected vs actual**: What should happen vs what does happen  
✅ **Environment details**: Versions, OS, installation method  
✅ **Full error messages**: Complete stack traces  

### Poor Bug Reports:

❌ **Vague title**: "It doesn't work"  
❌ **No example**: "Function fails when I call it"  
❌ **Missing details**: No versions, environment, or error messages  
❌ **Too complex**: Hundreds of lines of unrelated code  
❌ **No error info**: "It just crashes"  

Taking time to write a good bug report helps us fix issues faster and makes the library better for everyone!

## Security Issues

For security vulnerabilities, please **do not** create public issues. Instead:

1. **Email the maintainers** directly
2. **Include "SECURITY"** in the subject line
3. **Provide details** privately
4. **Allow time** for private resolution

We will acknowledge security reports within 48 hours and work to resolve them promptly.