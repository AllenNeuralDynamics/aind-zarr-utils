# Pipeline Corrections Guide

This guide explains the pipeline domain correction system in aind-zarr-utils, which handles version-specific spatial domain corrections for SmartSPIM pipeline compatibility.

## Overview

Different versions of the SmartSPIM pipeline have produced ZARR files with varying spatial properties (spacing, origin positioning, etc.). To maintain compatibility and reproduce pipeline-specific analyses, aind-zarr-utils provides a correction system that can mimic the exact spatial domain produced by any pipeline version.

## Why Pipeline Corrections?

### The Problem

```python
# Different pipeline versions = different spatial properties
pipeline_v0_25 = "smartspim-pipeline v0.0.25"  # Has spacing bugs
pipeline_v1_0  = "smartspim-pipeline v1.0.0"   # Fixed spacing

# Same raw data, different final coordinate systems!
stub_v25 = zarr_to_sitk_stub(zarr_uri, metadata)  # Inherits v0.0.25 bugs
stub_v10 = zarr_to_sitk_stub(zarr_uri, metadata)  # Uses v1.0.0 fixes

# Different spacing → Different physical coordinates → Analysis errors
assert stub_v25.GetSpacing() != stub_v10.GetSpacing()  
```

### The Solution

Pipeline corrections allow you to:
- **Reproduce historical analyses** using the exact spatial domain from any pipeline version
- **Compare results across versions** by understanding spatial differences  
- **Maintain compatibility** when working with mixed-version datasets
- **Debug coordinate issues** by isolating pipeline-specific effects

## Core Concepts

### Header System

The correction system uses immutable `Header` objects representing spatial properties:

```python
from aind_zarr_utils.pipeline_domain_selector import Header

# A Header represents complete spatial domain information
header = Header(
    origin=(0.0, 0.0, 0.0),           # Physical origin coordinates
    spacing=(0.0144, 0.0144, 0.016),  # Voxel spacing in millimeters
    direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),  # 3x3 direction matrix (flattened)
    size=(2000, 1500, 800)           # Image dimensions in voxels
)

print(f"Physical extent: {header.physical_extent()}")
print(f"Center coordinates: {header.center()}")
```

### Overlay System

Corrections are applied through **overlays** - version-specific modifications:

```python
from aind_zarr_utils.pipeline_domain_selector import (
    FixedSpacingOverlay, RASCornerAnchorOverlay, apply_overlays
)

# Example overlay for v0.0.25 (fixed spacing)
fixed_spacing = FixedSpacingOverlay(
    new_spacing=(0.0144, 0.0144, 0.016),  # Specific spacing values
    range_spec=SpecifierSet(">=0.0.18,<0.0.34")  # Version range
)

# Apply overlay to base header
corrected_header = apply_overlays(base_header, [fixed_spacing])
```

## Common Corrections

### 1. Fixed Spacing Correction

Historical pipeline versions used hardcoded spacing values:

```python
from aind_zarr_utils.pipeline_domain_selector import get_overlays_for_version

# Get all corrections for a specific version
overlays = get_overlays_for_version("smartspim-pipeline v0.0.25")

# Apply to your header
base_header = Header.from_metadata(zarr_metadata)
corrected_header = apply_overlays(base_header, overlays)

print(f"Original spacing: {base_header.spacing}")
print(f"Pipeline v0.0.25 spacing: {corrected_header.spacing}")
# Output: (0.0144, 0.0144, 0.016) - the hardcoded values
```

### 2. RAS Corner Anchoring

Some versions positioned specific image corners at fixed coordinates:

```python
# RAS corner anchoring places Right-Anterior-Superior corner at origin
ras_anchor = RASCornerAnchorOverlay(
    anchor_lps=(0.0, 0.0, 0.0),  # LPS coordinates for RAS corner
    range_spec=SpecifierSet(">=0.0.18,<0.0.34")
)

# This automatically adjusts origin based on image size and spacing
corrected_header = apply_overlays(base_header, [ras_anchor])

# RAS corner is now at (0,0,0), origin shifted accordingly
print(f"New origin: {corrected_header.origin}")
print(f"RAS corner: {corrected_header.ras_corner()}")  # Should be (0,0,0)
```

## High-Level Functions

### Automatic Pipeline Correction

The simplest approach - let aind-zarr-utils handle everything:

```python
from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_anatomical_stub
from aind_zarr_utils.json_utils import get_json

# Load your data
zarr_metadata = get_json("s3://bucket/zarr_metadata.json")
processing_metadata = get_json("s3://bucket/processing.json") 

# Create stub with automatic pipeline version detection and correction
stub = mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri="s3://bucket/data.ome.zarr/0",
    zarr_metadata=zarr_metadata,
    processing_metadata=processing_metadata
)

# Result: SimpleITK image stub with pipeline-corrected spatial domain
print(f"Corrected spacing: {stub.GetSpacing()}")
print(f"Corrected origin: {stub.GetOrigin()}")
```

### Processing Metadata Structure

The `processing_metadata` must contain pipeline version information:

```python
# Required structure in processing.json
processing_metadata = {
    "processing": {
        "pipeline_version": "smartspim-pipeline v0.0.25",  # Must be present
        "pipeline_url": "...",
        "code_url": "..."
    }
    # Other processing information...
}

# The pipeline_version string is used for overlay selection
```

## Advanced Usage

### Custom Overlay Rules

Create your own correction rules for specific use cases:

```python
from aind_zarr_utils.pipeline_domain_selector import (
    OverlayRule, FixedSpacingOverlay, extend_selector
)
from packaging.specifiers import SpecifierSet

# Define custom overlay
custom_spacing_rule = OverlayRule(
    name="custom_high_res",
    spec=SpecifierSet(">=1.0.0"),  # Apply to versions >= 1.0.0
    overlay=FixedSpacingOverlay((0.005, 0.005, 0.008))  # High-resolution spacing
)

# Extend default selector with custom rule
custom_selector = extend_selector([custom_spacing_rule])

# Apply custom corrections
stub = mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri, zarr_metadata, processing_metadata,
    overlay_selector=custom_selector
)
```

### Manual Header Manipulation

For complete control over spatial domain:

```python
from aind_zarr_utils.pipeline_domain_selector import Header
from aind_zarr_utils.zarr import _open_zarr

# Start with ZARR metadata  
image_node, zarr_meta = _open_zarr(zarr_uri)
base_header = Header.from_zarr_metadata(zarr_meta, level=0)

print("Original header:")
print(f"  Spacing: {base_header.spacing}")
print(f"  Origin: {base_header.origin}")

# Apply specific corrections manually
spacing_overlay = FixedSpacingOverlay((0.010, 0.010, 0.012))
corrected_header = spacing_overlay.apply(base_header)

print("After spacing correction:")
print(f"  Spacing: {corrected_header.spacing}")
print(f"  Origin: {corrected_header.origin}")  # Origin unchanged

# Convert to SimpleITK image
sitk_image = corrected_header.to_sitk_image()
print(f"Final SimpleITK spacing: {sitk_image.GetSpacing()}")
```

## Practical Examples

### Reproducing Historical Analysis

```python
# You have data processed with v0.0.25, need to reproduce results
def reproduce_v025_analysis(zarr_uri, zarr_metadata, processing_metadata):
    """Reproduce analysis using exact v0.0.25 spatial domain."""
    
    # Get the corrected spatial domain for v0.0.25
    stub = mimic_pipeline_zarr_to_anatomical_stub(
        zarr_uri, zarr_metadata, processing_metadata
    )
    
    # This stub now has the exact same spatial properties 
    # that v0.0.25 would have produced
    
    # Load your annotations (e.g., from Neuroglancer)
    from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical
    
    # Transform using corrected spatial domain  
    physical_points, descriptions = neuroglancer_annotations_to_anatomical(
        neuroglancer_data, zarr_uri, zarr_metadata, scale_unit="millimeter"
    )
    
    # Results match what v0.0.25 would have produced
    return analyze_with_historical_coordinates(physical_points)
```

### Version Comparison Study

```python
def compare_pipeline_versions(zarr_uri, zarr_metadata, neuroglancer_data):
    """Compare coordinate transformations across pipeline versions."""
    
    # Test different pipeline versions
    versions = ["v0.0.25", "v0.1.0", "v1.0.0"]
    results = {}
    
    for version in versions:
        # Create fake processing metadata for each version
        processing_meta = {
            "processing": {"pipeline_version": f"smartspim-pipeline {version}"}
        }
        
        # Get version-specific spatial domain
        stub = mimic_pipeline_zarr_to_anatomical_stub(
            zarr_uri, zarr_metadata, processing_meta
        )
        
        # Transform same annotations with each version's domain
        physical_points, _ = neuroglancer_annotations_to_anatomical(
            neuroglancer_data, zarr_uri, zarr_metadata
        )
        
        results[version] = {
            "spacing": stub.GetSpacing(),
            "origin": stub.GetOrigin(), 
            "transformed_points": physical_points
        }
    
    # Analyze differences
    print("Pipeline version comparison:")
    for version, data in results.items():
        print(f"{version}:")
        print(f"  Spacing: {data['spacing']}")
        print(f"  Origin: {data['origin']}")
        
    return results
```

### Debugging Spatial Issues

```python
def diagnose_pipeline_spatial_domain(zarr_uri, zarr_metadata, processing_metadata):
    """Diagnose potential spatial domain issues."""
    
    # Get base (uncorrected) header
    image_node, zarr_meta = _open_zarr(zarr_uri)  
    base_header = Header.from_zarr_metadata(zarr_meta, level=0)
    
    # Get pipeline-corrected header
    stub = mimic_pipeline_zarr_to_anatomical_stub(
        zarr_uri, zarr_metadata, processing_metadata
    )
    corrected_spacing = stub.GetSpacing()
    corrected_origin = stub.GetOrigin()
    
    print("=== Spatial Domain Diagnosis ===")
    print(f"Pipeline version: {processing_metadata['processing']['pipeline_version']}")
    print()
    print("Base (ZARR metadata) spatial domain:")
    print(f"  Spacing: {base_header.spacing}")
    print(f"  Origin: {base_header.origin}")
    print()
    print("Pipeline-corrected spatial domain:")  
    print(f"  Spacing: {corrected_spacing}")
    print(f"  Origin: {corrected_origin}")
    print()
    
    # Check for corrections applied
    spacing_changed = base_header.spacing != corrected_spacing
    origin_changed = base_header.origin != corrected_origin
    
    print("Corrections applied:")
    print(f"  Spacing modified: {spacing_changed}")
    print(f"  Origin modified: {origin_changed}")
    
    if spacing_changed:
        spacing_ratio = tuple(
            c/b for b, c in zip(base_header.spacing, corrected_spacing)
        )
        print(f"  Spacing ratio: {spacing_ratio}")
    
    return {
        "base_header": base_header,
        "corrected_spacing": corrected_spacing,
        "corrected_origin": corrected_origin,
        "corrections_applied": spacing_changed or origin_changed
    }
```

## Available Overlays

### FixedSpacingOverlay

Forces specific spacing values regardless of ZARR metadata:

```python
from aind_zarr_utils.pipeline_domain_selector import FixedSpacingOverlay

overlay = FixedSpacingOverlay(
    new_spacing=(0.0144, 0.0144, 0.016),  # X, Y, Z spacing in mm
    range_spec=SpecifierSet(">=0.0.18,<0.0.34")  # Version range
)

# Effect: Sets exact spacing, preserves origin
corrected = overlay.apply(header)
assert corrected.spacing == (0.0144, 0.0144, 0.016)
```

### RASCornerAnchorOverlay  

Positions the RAS (Right-Anterior-Superior) corner at specific coordinates:

```python
from aind_zarr_utils.pipeline_domain_selector import RASCornerAnchorOverlay

overlay = RASCornerAnchorOverlay(
    anchor_lps=(0.0, 0.0, 0.0),  # Place RAS corner at origin
    range_spec=SpecifierSet(">=0.0.18,<0.0.34")
)

# Effect: Calculates new origin so RAS corner is at anchor_lps
corrected = overlay.apply(header)
assert corrected.ras_corner() == (0.0, 0.0, 0.0)
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
    print("Required structure:")
    print("""{
        "processing": {
            "pipeline_version": "smartspim-pipeline v0.0.25"
        }
    }""")
```

### Unsupported Pipeline Version

```python
# If no overlays match the version, function falls back to base ZARR metadata
processing_metadata = {
    "processing": {"pipeline_version": "unknown-pipeline v99.0.0"}
}

stub = mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri, zarr_metadata, processing_metadata
)
# Result: Uses spacing/origin from ZARR metadata directly
```

### Version Parsing Issues

```python
from packaging.version import InvalidVersion

try:
    overlays = get_overlays_for_version("invalid version string")
except InvalidVersion:
    print("Pipeline version string must be valid semver format")
    print("Examples: 'smartspim-pipeline v0.0.25', 'pipeline v1.0.0'")
```

## Best Practices

### 1. Always Use Processing Metadata

```python
# GOOD: Include pipeline version information
processing_data = {
    "processing": {
        "pipeline_version": "smartspim-pipeline v0.0.25",
        "pipeline_url": "https://...",
        "code_url": "https://..."
    }
}

stub = mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri, zarr_metadata, processing_data
)

# AVOID: Using zarr_to_sitk_stub() directly for pipeline data
# This bypasses pipeline-specific corrections
```

### 2. Document Pipeline Versions

```python
def analyze_with_version_info(data_path, expected_version=None):
    """Analyze data with explicit pipeline version tracking.
    
    Parameters
    ----------
    data_path : str
        Path to dataset
    expected_version : str, optional
        Expected pipeline version for validation
    """
    
    processing_data = get_json(f"{data_path}/processing.json")
    actual_version = processing_data["processing"]["pipeline_version"]
    
    if expected_version and actual_version != expected_version:
        print(f"Warning: Expected {expected_version}, got {actual_version}")
    
    print(f"Using pipeline version: {actual_version}")
    
    # Proceed with analysis using version-aware corrections
    stub = mimic_pipeline_zarr_to_anatomical_stub(
        f"{data_path}/data.ome.zarr/0",
        get_json(f"{data_path}/metadata.json"),
        processing_data
    )
    
    return analyze_data(stub)
```

### 3. Test Across Versions

```python
def validate_cross_version_compatibility(test_data):
    """Test that analysis works across different pipeline versions."""
    
    versions_to_test = [
        "smartspim-pipeline v0.0.25", 
        "smartspim-pipeline v0.1.0",
        "smartspim-pipeline v1.0.0"
    ]
    
    results = []
    for version in versions_to_test:
        processing_meta = {
            "processing": {"pipeline_version": version}
        }
        
        try:
            result = analyze_with_pipeline_corrections(test_data, processing_meta)
            results.append((version, "SUCCESS", result))
        except Exception as e:
            results.append((version, "ERROR", str(e)))
    
    return results
```

## Summary

The pipeline correction system enables:

- **Historical reproducibility**: Recreate exact spatial domains from any pipeline version
- **Version comparison**: Understand how spatial properties changed across versions
- **Data compatibility**: Work with mixed-version datasets consistently  
- **Error diagnosis**: Isolate pipeline-specific spatial domain issues

**Key points**:
- Always include `pipeline_version` in processing metadata
- Use `mimic_pipeline_zarr_to_anatomical_stub()` for pipeline data
- Document which pipeline version your analysis assumes
- Test analysis across versions when possible
- Be aware that different versions = different coordinate systems

The correction system ensures that spatial transformations remain accurate and reproducible regardless of which pipeline version originally processed the data.