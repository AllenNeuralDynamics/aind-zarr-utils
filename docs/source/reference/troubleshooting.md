# Troubleshooting

Common issues and solutions when using aind-zarr-utils.

## Installation Issues

### ImportError: No module named 'SimpleITK'

**Problem**: SimpleITK is not installed or not found.

**Solution**:
```bash
# Install/reinstall SimpleITK
pip install --upgrade SimpleITK

# Or force reinstall
pip install --force-reinstall SimpleITK
```

### ImportError: No module named 'ants'

**Problem**: ANTs (antspyx) is not installed properly.

**Solutions**:
```bash
# Try installing specific version
pip install antspyx==0.3.8

# If that fails, try conda
conda install -c conda-forge antspyx

# For M1/M2 Macs
pip install antspyx --no-deps
pip install numpy scipy matplotlib Pillow pynrrd webcolors
```

### Dependency Conflicts

**Problem**: Conflicting package versions.

**Solution**:
```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/macOS
# or fresh_env\Scripts\activate  # Windows

# Install aind-zarr-utils only
pip install aind-zarr-utils

# Or use uv for better dependency resolution
uv sync
```

## S3 Access Issues

### SSL Certificate Errors

**Problem**: 
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**Solutions**:
```python
# Option 1: Update certificates (macOS)
/Applications/Python\ 3.x/Install\ Certificates.command

# Option 2: Set SSL context (temporary workaround)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Option 3: Use specific CA bundle
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()
```

### Connection Timeouts

**Problem**: S3 requests timing out.

**Solutions**:
```python
import boto3
from botocore.config import Config

# Configure longer timeouts
config = Config(
    read_timeout=60,
    connect_timeout=30,
    retries={'max_attempts': 5, 'mode': 'adaptive'}
)

# Use with S3 operations
from aind_zarr_utils.json_utils import get_json_s3
data = get_json_s3("bucket", "key", s3_client=boto3.client('s3', config=config))
```

### Access Denied Errors

**Problem**: 
```
ClientError: An error occurred (403) when calling the GetObject operation: Forbidden
```

**Solutions**:
```python
# For public buckets, ensure anonymous access
from aind_zarr_utils.json_utils import get_json_s3_uri
data = get_json_s3_uri("s3://aind-open-data/file.json", anonymous=True)

# For private buckets, check credentials
import boto3
session = boto3.Session()
credentials = session.get_credentials()
print(f"Access key: {credentials.access_key[:5]}...")  # Verify credentials exist

# Check bucket policy and IAM permissions
```

### Region Mismatch

**Problem**: Bucket in different region than default.

**Solution**:
```python
import boto3

# Specify correct region
s3_client = boto3.client('s3', region_name='us-west-2')

# Or set environment variable
import os
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
```

## ZARR Processing Issues

### Memory Errors

**Problem**: 
```
MemoryError: Unable to allocate array
```

**Solutions**:
```python
# Use higher resolution level (smaller images)
from aind_zarr_utils.zarr import zarr_to_ants

# Instead of level=0 (full resolution)
img = zarr_to_ants(zarr_uri, metadata, level=3)  # Much smaller

# Check memory requirements first
from aind_zarr_utils.zarr import zarr_to_sitk_stub
stub, size = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
memory_gb = (size[0] * size[1] * size[2] * 4) / (1024**3)
print(f"Estimated memory: {memory_gb:.1f} GB")

if memory_gb > 8:  # Adjust based on available RAM
    print("Use higher level or process in chunks")
```

### Incorrect Image Spacing

**Problem**: Image spacing values don't match expectations.

**Solutions**:
```python
# Check metadata structure
from aind_zarr_utils.json_utils import get_json
metadata = get_json("s3://bucket/metadata.json")

# Verify acquisition metadata
print("Acquisition metadata:")
print(metadata.get("acquisition", {}))

# Check coordinate transformations
ome_zarr_meta = metadata.get("ome_zarr_metadata", {})
print("OME-ZARR transformations:")
print(ome_zarr_meta.get("multiscales", []))

# Use explicit scale unit
from aind_zarr_utils.zarr import zarr_to_ants
img = zarr_to_ants(zarr_uri, metadata, scale_unit="millimeter")
print(f"Spacing in mm: {img.spacing}")
```

### Wrong Coordinate System

**Problem**: Coordinates don't match expected anatomical locations.

**Solution**:
```python
# Verify coordinate system
from aind_zarr_utils.zarr import zarr_to_ants
img = zarr_to_ants(zarr_uri, metadata, level=3)

print(f"Direction matrix:\n{img.direction}")
print(f"Origin: {img.origin}")

# Check if direction matrix is identity
import numpy as np
if not np.allclose(img.direction, np.eye(3)):
    print("⚠ Non-standard orientation detected")
    
# Verify LPS convention
# +X should be Left, +Y should be Posterior, +Z should be Superior
```

### ZARR File Not Found

**Problem**: 
```
FileNotFoundError: ZARR file not found
```

**Solutions**:
```python
# Check URI format
zarr_uri = "s3://aind-open-data/dataset/data.ome.zarr/0"  # Correct
# not: "s3://aind-open-data/dataset/data.ome.zarr"      # Missing /0

# List available levels
import zarr
for level in range(10):
    try:
        store = zarr.open(f"{base_uri}/{level}")
        print(f"Level {level}: available")
    except:
        break

# Check if file exists
from aind_zarr_utils.json_utils import get_json
try:
    # Try to access the ZARR metadata first
    metadata_uri = zarr_uri.replace("/0", "") + "/.zmetadata"
    zarr_meta = get_json(metadata_uri)
    print("ZARR file exists")
except:
    print("ZARR file may not exist at that URI")
```

## Coordinate Transformation Issues

### Points in Wrong Locations

**Problem**: Transformed coordinates don't match expected anatomical positions.

**Solutions**:
```python
# Check input coordinate format
# Neuroglancer points should be [z, y, x] or [z, y, x, t]
ng_data = {
    "layers": {
        "test": {
            "annotations": [
                {"point": [100, 200, 150]}  # Should be [z, y, x]
            ]
        }
    }
}

# Verify transformation
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical
physical_coords, descriptions = neuroglancer_annotations_to_anatomical(
    ng_data, zarr_uri, metadata, scale_unit="millimeter"
)

print(f"Input indices: [100, 200, 150] (z, y, x)")
print(f"Output LPS mm: {physical_coords['test'][0]} (x, y, z)")

# Check coordinate system consistency
from aind_zarr_utils.zarr import zarr_to_sitk
img = zarr_to_sitk(zarr_uri, metadata, level=3)
sitk_physical = img.TransformIndexToPhysicalPoint([150, 200, 100])  # [x, y, z]
print(f"SimpleITK result: {sitk_physical} (should match)")
```

### Axis Order Confusion

**Problem**: X/Y/Z coordinates seem swapped.

**Solution**:
```python
# Remember coordinate conventions:
# - Neuroglancer: [z, y, x] indices
# - SimpleITK: [x, y, z] for transforms
# - ANTs: [z, y, x] array shape but [x, y, z] coordinates
# - Output: Always LPS [x, y, z] physical coordinates

# Check array vs coordinate order
from aind_zarr_utils.zarr import zarr_to_ants, zarr_to_sitk

ants_img = zarr_to_ants(zarr_uri, metadata, level=3)
sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3)

print("ANTs array shape (ZYX):", ants_img.shape)
print("ANTs spacing (ZYX):", ants_img.spacing)
print("SimpleITK size (XYZ):", sitk_img.GetSize())
print("SimpleITK spacing (XYZ):", sitk_img.GetSpacing())

# Both should have same physical extent
ants_extent = np.array(ants_img.shape) * np.array(ants_img.spacing)
sitk_extent = np.array(sitk_img.GetSize()) * np.array(sitk_img.GetSpacing())
print("Physical extents should match:", ants_extent, sitk_extent)
```

## Pipeline Processing Issues

### Missing Pipeline Version

**Problem**: 
```
KeyError: 'pipeline_version'
```

**Solution**:
```python
# Check processing metadata structure
from aind_zarr_utils.json_utils import get_json
processing_data = get_json("s3://bucket/processing.json")

# Required structure:
required_structure = {
    "processing": {
        "pipeline_version": "smartspim-pipeline v0.0.25"
    }
}

# Check if structure exists
if "processing" not in processing_data:
    print("Missing 'processing' key in metadata")
elif "pipeline_version" not in processing_data["processing"]:
    print("Missing 'pipeline_version' in processing metadata")
else:
    print(f"Pipeline version: {processing_data['processing']['pipeline_version']}")
```

### Unknown Pipeline Version

**Problem**: No corrections available for pipeline version.

**Solution**:
```python
# Check available corrections
from aind_zarr_utils.pipeline_domain_selector import get_overlays_for_version

version = "smartspim-pipeline v0.0.25"
overlays = get_overlays_for_version(version)

if not overlays:
    print(f"No corrections for version: {version}")
    print("Available versions with corrections:")
    # List known versions with corrections
    known_versions = [
        "smartspim-pipeline v0.0.25",
        "smartspim-pipeline v0.1.0",
        # Add others as discovered
    ]
    for v in known_versions:
        if get_overlays_for_version(v):
            print(f"  {v}")
else:
    print(f"Found {len(overlays)} corrections for {version}")
```

### Transform File Access Issues

**Problem**: Cannot access ANTs transform files for pipeline processing.

**Solution**:
```python
# Check transform paths
from aind_zarr_utils.pipeline_transformed import pipeline_transforms

try:
    individual_paths, template_paths = pipeline_transforms(zarr_uri, processing_data)
    print("Individual transforms:", individual_paths)
    print("Template transforms:", template_paths)
    
    # Check if files exist
    for path in individual_paths + template_paths:
        try:
            from aind_zarr_utils.json_utils import get_json
            # Try to access the file
            result = get_json(path)
            print(f"✓ {path}")
        except Exception as e:
            print(f"✗ {path}: {e}")
            
except Exception as e:
    print(f"Error getting transform paths: {e}")
```

## Performance Issues

### Slow S3 Access

**Problem**: S3 operations are very slow.

**Solutions**:
```python
# Enable persistent caching
from aind_zarr_utils.s3_cache import CacheManager

with CacheManager(persistent=True, cache_dir="~/.aind_cache") as cm:
    # All S3 operations will be cached
    from aind_zarr_utils.json_utils import get_json
    data = get_json("s3://bucket/file.json", cache_dir=cm.dir)

# Use parallel downloads for multiple files
import concurrent.futures
from aind_zarr_utils.s3_cache import get_local_path_for_resource

def download_file(uri):
    return get_local_path_for_resource(uri, cache_dir="~/.cache")

uris = ["s3://bucket/file1.json", "s3://bucket/file2.json"]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(download_file, uris))
```

### High Memory Usage

**Problem**: Process uses too much memory.

**Solutions**:
```python
# Use stub images for coordinate-only operations
from aind_zarr_utils.zarr import zarr_to_sitk_stub

stub, original_size = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
# Stub has full coordinate system but minimal memory usage

# Process in chunks for large datasets
def process_in_chunks(zarr_uri, metadata, chunk_size=1000):
    from aind_zarr_utils.zarr import _open_zarr
    
    image_node, zarr_meta = _open_zarr(zarr_uri)
    level_data = image_node['3']  # Use lower resolution
    
    # Process chunks instead of full array
    for z in range(0, level_data.shape[0], chunk_size):
        chunk = level_data[z:z+chunk_size]
        # Process chunk...
        yield process_chunk(chunk)
```

### Slow Coordinate Transformations

**Problem**: Transforming large numbers of points is slow.

**Solution**:
```python
# Vectorize transformations
import numpy as np

def batch_transform_points(points, sitk_image):
    """Transform multiple points efficiently."""
    # Convert to numpy array if needed
    points = np.array(points)
    
    # Get transform parameters once
    origin = np.array(sitk_image.GetOrigin())
    spacing = np.array(sitk_image.GetSpacing())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
    
    # Vectorized transformation
    # points are in [x, y, z] order for SimpleITK
    transformed = origin + (points @ direction.T) * spacing
    
    return transformed

# Use for large point sets
large_point_set = np.random.rand(10000, 3) * 100
transformed = batch_transform_points(large_point_set, sitk_img)
```

## Error Diagnosis

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all operations will show detailed debug info
from aind_zarr_utils.zarr import zarr_to_ants
result = zarr_to_ants(zarr_uri, metadata, level=3)
```

### Check Environment

```python
import sys
import platform
import aind_zarr_utils

print("=== Environment Diagnosis ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"aind-zarr-utils: {aind_zarr_utils.__version__}")

# Check key dependencies
dependencies = ['SimpleITK', 'ants', 'boto3', 'zarr', 'requests']
for dep in dependencies:
    try:
        module = __import__(dep)
        version = getattr(module, '__version__', 'unknown')
        print(f"{dep}: {version}")
    except ImportError:
        print(f"{dep}: NOT INSTALLED")

# Check S3 connectivity
try:
    from aind_zarr_utils.json_utils import get_json
    test_data = get_json("s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/metadata.json")
    print("S3 connectivity: ✓ Working")
except Exception as e:
    print(f"S3 connectivity: ✗ Error - {e}")
```

### Validate Data

```python
def validate_zarr_data(zarr_uri, metadata):
    """Validate ZARR data and metadata."""
    print("=== Data Validation ===")
    
    # Check metadata structure
    required_keys = ['session_id', 'acquisition']
    for key in required_keys:
        if key not in metadata:
            print(f"⚠ Missing metadata key: {key}")
        else:
            print(f"✓ Found metadata key: {key}")
    
    # Check acquisition metadata
    if 'acquisition' in metadata:
        acq = metadata['acquisition']
        if 'coordinate_transformations' in acq:
            print("✓ Found coordinate transformations")
        else:
            print("⚠ Missing coordinate transformations")
    
    # Try to access ZARR levels
    from aind_zarr_utils.zarr import zarr_to_sitk_stub
    for level in range(6):
        try:
            stub, size = zarr_to_sitk_stub(zarr_uri, metadata, level=level)
            print(f"✓ Level {level}: {size} voxels")
        except Exception as e:
            print(f"✗ Level {level}: {e}")
            break

# Use to diagnose data issues
validate_zarr_data(zarr_uri, metadata)
```

## Getting Help

If these solutions don't resolve your issue:

1. **Search existing issues**: [GitHub Issues](https://github.com/AllenNeuralDynamics/aind-zarr-utils/issues)
2. **Create a bug report**: Use the [bug reporting template](../contributing/bug-discovery.md)
3. **Ask in discussions**: [GitHub Discussions](https://github.com/AllenNeuralDynamics/aind-zarr-utils/discussions)
4. **Check documentation**: Review relevant [user guides](../user-guide/index.md)

### Include This Information

When asking for help, always include:

- **aind-zarr-utils version**: `python -c "import aind_zarr_utils; print(aind_zarr_utils.__version__)"`
- **Python version**: `python --version`
- **Operating system**: Linux/macOS/Windows
- **Full error message**: Complete stack trace
- **Minimal example**: Code that reproduces the issue

### Common Solutions Summary

| Problem | Quick Solution |
|---------|----------------|
| ImportError | `pip install --upgrade package-name` |
| SSL/Certificate | Update certificates or use `certifi` |
| S3 timeout | Increase timeout config in boto3 |
| Memory error | Use higher resolution level |
| Wrong coordinates | Check input format and coordinate system |
| Missing pipeline version | Verify processing metadata structure |
| Slow performance | Enable caching, use appropriate resolution |

Most issues can be resolved by checking your installation, using appropriate resolution levels, and ensuring proper data formats!