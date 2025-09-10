# ZARR Conversion Guide

This guide covers converting ZARR datasets to SimpleITK and ANTs images using aind-zarr-utils.

## Overview

ZARR files store multi-resolution image data with metadata. aind-zarr-utils provides functions to convert this data into formats suitable for analysis:

- **ANTs images**: For registration, segmentation, and analysis workflows
- **SimpleITK images**: For ITK-based processing and visualization  
- **Stub images**: Memory-efficient coordinate system representation

## Basic Conversion

### Prerequisites

First, load your metadata:

```python
from aind_zarr_utils.json_utils import get_json

# Load ZARR metadata (required for coordinate system information)
metadata = get_json("s3://aind-open-data/path/to/metadata.json")
zarr_uri = "s3://aind-open-data/path/to/data.ome.zarr/0"
```

### Convert to ANTs Image

ANTs images are ideal for registration and analysis workflows:

```python
from aind_zarr_utils.zarr import zarr_to_ants

# Convert ZARR to ANTs image
ants_img = zarr_to_ants(
    zarr_uri, 
    metadata, 
    level=3,                    # Resolution level (3 = typical working resolution)
    scale_unit="millimeter"     # Output units
)

print(f"Image size: {ants_img.shape}")
print(f"Spacing: {ants_img.spacing}")
print(f"Origin: {ants_img.origin}")
```

### Convert to SimpleITK Image

SimpleITK images work with ITK-based tools and provide extensive image processing capabilities:

```python
from aind_zarr_utils.zarr import zarr_to_sitk

# Convert ZARR to SimpleITK image
sitk_img = zarr_to_sitk(
    zarr_uri,
    metadata,
    level=3,
    scale_unit="millimeter"
)

print(f"Image size: {sitk_img.GetSize()}")
print(f"Spacing: {sitk_img.GetSpacing()}")
print(f"Origin: {sitk_img.GetOrigin()}")
```

## Resolution Levels

ZARR files contain multiple resolution levels (multiscale data):

- **Level 0**: Full resolution (largest file size)
- **Level 3**: Typical working resolution (balanced size/detail)
- **Level 5+**: Preview resolution (fastest to load)

```python
# Compare different resolution levels
for level in [0, 3, 5]:
    ants_img = zarr_to_ants(zarr_uri, metadata, level=level)
    print(f"Level {level}: {ants_img.shape} voxels")
```

**Choosing Resolution Levels:**
- **Level 0**: Final analysis, publication figures
- **Level 3**: Development, testing, most analysis workflows  
- **Level 5+**: Quick previews, coordinate system validation

## Scale Units

aind-zarr-utils supports automatic unit conversion:

```python
# Available scale units
units = ["micrometer", "millimeter", "centimeter", "meter"]

for unit in units:
    ants_img = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit=unit)
    print(f"{unit}: spacing = {ants_img.spacing}")
```

**Recommended Units:**
- **Millimeters**: Standard for medical imaging, ANTs, SimpleITK
- **Micrometers**: Original acquisition units, detailed analysis

## Memory-Efficient Workflows

### Stub Images

For coordinate transformations without loading pixel data:

```python
from aind_zarr_utils.zarr import zarr_to_sitk_stub

# Create stub image (minimal memory usage)
stub_img, original_size = zarr_to_sitk_stub(
    zarr_uri,
    metadata,
    level=0,  # Use level 0 for full-resolution coordinate system
    scale_unit="millimeter"
)

# Stub has same coordinate system as full image
print(f"Stub size: {stub_img.GetSize()}")  # Always (1,1,1)
print(f"Original size: {original_size}")   # Actual image dimensions
print(f"Same spacing: {stub_img.GetSpacing()}")
print(f"Same origin: {stub_img.GetOrigin()}")

# Use for coordinate transformations
physical_point = stub_img.TransformIndexToPhysicalPoint([100, 200, 50])
print(f"Physical coordinates: {physical_point}")
```

## Advanced Features

### Custom Origin Positioning

Position specific image corners at known anatomical locations:

```python
# Position RAS corner at specific coordinates
ants_img = zarr_to_ants(
    zarr_uri,
    metadata,
    level=3,
    set_corner="RAS",                    # Right-Anterior-Superior corner
    set_corner_lps=(10.0, 5.0, 15.0)   # LPS coordinates in mm
)

# Alternative: set origin directly
ants_img = zarr_to_ants(
    zarr_uri,
    metadata,
    level=3,
    set_origin=(0.0, 0.0, 0.0)  # LPS coordinates
)
```

### Pre-opened ZARR Files

For efficiency when processing multiple levels:

```python
from aind_zarr_utils.zarr import _open_zarr

# Open ZARR once, reuse for multiple operations
image_node, zarr_meta = _open_zarr(zarr_uri)

# Create multiple stub images efficiently
stub_level0, _ = zarr_to_sitk_stub(
    zarr_uri, metadata, level=0, opened_zarr=(image_node, zarr_meta)
)
stub_level3, _ = zarr_to_sitk_stub(
    zarr_uri, metadata, level=3, opened_zarr=(image_node, zarr_meta)
)
```

## Coordinate System Details

### LPS Convention

All functions output **LPS (Left-Posterior-Superior)** coordinates:

- **L**: Left direction = +X
- **P**: Posterior direction = +Y  
- **S**: Superior direction = +Z

This matches ITK, SimpleITK, and medical imaging standards.

### SimpleITK vs ANTs Differences

**Axis Ordering:**
- **SimpleITK**: Uses Fortran-style indexing (column-major)
- **ANTs**: Uses C-style indexing (row-major)

```python
# Same coordinate system, different axis conventions
sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3)
ants_img = zarr_to_ants(zarr_uri, metadata, level=3)

# SimpleITK: [x, y, z] indexing
sitk_size = sitk_img.GetSize()           # (nx, ny, nz)
sitk_spacing = sitk_img.GetSpacing()     # (sx, sy, sz)

# ANTs: [z, y, x] indexing  
ants_shape = ants_img.shape              # (nz, ny, nx)
ants_spacing = ants_img.spacing          # (sz, sy, sx)
```

## Performance Tips

1. **Start with Level 3**: Good balance of detail and speed
2. **Use Stub Images**: For coordinate-only operations
3. **Cache Metadata**: Reuse metadata objects across conversions
4. **Choose Appropriate Units**: Millimeters for most workflows

## Common Issues

### Metadata Requirements

Ensure metadata contains acquisition information:

```python
# Check metadata structure
required_keys = ["acquisition"]
if not all(key in metadata for key in required_keys):
    raise ValueError("Metadata missing required acquisition information")
```

### Memory Considerations

Large ZARR files at level 0 can exceed available memory:

```python
# Check image size before loading
stub_img, size = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
total_voxels = size[0] * size[1] * size[2]
memory_gb = total_voxels * 4 / (1024**3)  # Assuming 4 bytes per voxel

print(f"Estimated memory: {memory_gb:.2f} GB")
if memory_gb > 8:  # Adjust threshold based on available RAM
    print("Consider using a higher level (lower resolution)")
```

### Coordinate System Validation

Verify coordinate system correctness:

```python
# Check direction matrix
ants_img = zarr_to_ants(zarr_uri, metadata, level=3)
print(f"Direction matrix:\n{ants_img.direction}")

# Verify expected anatomical orientation
if not np.allclose(ants_img.direction, np.eye(3)):
    print("Non-identity direction matrix - verify orientation")
```