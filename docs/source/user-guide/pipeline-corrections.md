# Pipeline Corrections Guide

This guide explains the pipeline domain correction system in aind-zarr-utils, which handles version-specific spatial domain corrections for SmartSPIM pipeline compatibility.

## Overview

Different versions of the SmartSPIM pipeline have produced ZARR files with varying spatial properties (spacing, origin positioning, etc.). To maintain compatibility and reproduce pipeline-specific analyses, aind-zarr-utils provides a correction system that can mimic the exact spatial domain produced by any pipeline version.

## Why Pipeline Corrections?

```python
# Different pipeline versions = different spatial properties
pipeline_v0_25 = "smartspim-pipeline v0.0.25"  # Has spacing bugs
pipeline_v1_0  = "smartspim-pipeline v1.0.0"   # Fixed spacing

# Same raw data, different final coordinate systems!
# Different spacing → Different physical coordinates → Analysis errors
```

**The Solution**: Pipeline corrections allow you to reproduce historical analyses using the exact spatial domain from any pipeline version.

## Basic Usage

```python
from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_anatomical_stub

# Basic usage with automatic pipeline correction
stub = mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri="s3://bucket/data.ome.zarr/0",
    metadata=zarr_metadata,
    processing_data=processing_metadata
)

# Result: SimpleITK image stub with pipeline-corrected spatial domain
print(f"Corrected spacing: {stub.GetSpacing()}")
print(f"Corrected origin: {stub.GetOrigin()}")
```

## Core Concepts

### Header System

The correction system uses immutable `Header` objects representing spatial properties:

```python
from aind_zarr_utils.pipeline_domain_selector import Header
import numpy as np

# A Header represents complete spatial domain information
header = Header(
    origin=(0.0, 0.0, 0.0),           # Physical origin coordinates
    spacing=(0.0144, 0.0144, 0.016),  # Voxel spacing in millimeters
    direction=np.eye(3),              # 3x3 direction matrix
    size_ijk=(2000, 1500, 800)       # Image dimensions in voxels
)

# Header provides origin, spacing, direction, and size_ijk
print(f"Origin: {header.origin}")
print(f"Spacing: {header.spacing}")
```

### Overlay System

Corrections are applied through **overlays** - version-specific modifications:

```python
from aind_zarr_utils.pipeline_domain_selector import SetLpsWorldSpacingOverlay

# Example: Create spacing overlay
spacing_overlay = SetLpsWorldSpacingOverlay(lps_spacing_mm=(0.0144, 0.0144, 0.016))
corrected_header = spacing_overlay(base_header, {}, 3)
```

## Processing Metadata Structure

The `processing_data` must contain pipeline version information:

```python
# Required structure in processing.json
processing_data = {
    "processing": {
        "pipeline_version": "smartspim-pipeline v0.0.25",  # Must be present
        # Other processing information...
    }
}
```

## Available Overlays

### SetLpsWorldSpacingOverlay

Forces specific spacing values regardless of ZARR metadata:

```python
from aind_zarr_utils.pipeline_domain_selector import SetLpsWorldSpacingOverlay

overlay = SetLpsWorldSpacingOverlay(lps_spacing_mm=(0.0144, 0.0144, 0.016))
corrected = overlay(header, {}, 3)
assert corrected.spacing == (0.0144, 0.0144, 0.016)
```

### ForceCornerAnchorOverlay

Positions the RAS (Right-Anterior-Superior) corner at specific coordinates:

```python
from aind_zarr_utils.pipeline_domain_selector import ForceCornerAnchorOverlay

overlay = ForceCornerAnchorOverlay(
    corner_code="RAS",
    target_point_labeled=(0.0, 0.0, 0.0),  # Place RAS corner at origin
)
corrected = overlay(header, {}, 3)
# RAS corner position can be calculated from corrected header
```

## Error Handling

### Missing Pipeline Version

```python
try:
    stub = mimic_pipeline_zarr_to_anatomical_stub(
        zarr_uri, zarr_metadata, processing_metadata
    )
except KeyError as e:
    print("Missing pipeline version in processing metadata")
    print("Required: {'processing': {'pipeline_version': 'smartspim-pipeline v0.0.25'}}")
```

### Unsupported Pipeline Version

If no overlays match the version, the function falls back to base ZARR metadata.

## Best Practices

1. **Always include pipeline version** in processing metadata
2. **Use `mimic_pipeline_zarr_to_anatomical_stub()`** for pipeline data (not `zarr_to_sitk_stub()` directly)
3. **Document which pipeline version** your analysis assumes
4. **Test analysis across versions** when possible

## Summary

The pipeline correction system enables:

- **Historical reproducibility**: Recreate exact spatial domains from any pipeline version
- **Version comparison**: Understand how spatial properties changed across versions
- **Data compatibility**: Work with mixed-version datasets consistently

**Key points**:
- Always include `pipeline_version` in processing metadata
- Use `mimic_pipeline_zarr_to_anatomical_stub()` for pipeline data
- Different versions = different coordinate systems