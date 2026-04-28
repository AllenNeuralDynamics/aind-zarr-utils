# aind-zarr-utils

A Python utility library developed by Allen Institute for Neural Dynamics for
working with ZARR files and AIND metadata. This package enables converting ZARR
datasets to SimpleITK and ANTs images, processing neuroimaging annotation data
from Neuroglancer, and handling anatomical coordinate transformations.

## Key Features

- **ZARR ↔ Image Conversion**: Convert ZARR datasets to SimpleITK and ANTs images with proper coordinate system handling
- **Neuroglancer Integration**: Process annotation layers and coordinate transforms from Neuroglancer
- **Coordinate Transformations**: Handle point transformations from image space to anatomical space (LPS coordinates)
- **Multi-source JSON Reading**: Unified JSON loading from local files, HTTP URLs, and S3 URIs
- **Pipeline-specific Corrections**: Version-based spatial domain corrections for pipeline compatibility
- **CCF Registration**: Pipeline-specific coordinate transformations and CCF registration utilities

## Quick Start

```python
from aind_zarr_utils import zarr_to_ants, get_json

# Convert ZARR to ANTs image
ants_img = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit="millimeter")

# Load JSON from any source
data = get_json("s3://aind-open-data/path/to/file.json")
```

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started/index
user-guide/index
tutorials/index
api-reference/index
contributing/index
reference/index
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
