# Coordinate Systems Guide

This guide explains coordinate systems used in aind-zarr-utils and neuroimaging data processing.

## Overview

Coordinate systems define how 3D positions are represented and interpreted. Different neuroimaging tools use different conventions, making coordinate system management crucial for accurate data processing and analysis.

aind-zarr-utils standardizes on **LPS (Left-Posterior-Superior)** coordinates for all outputs while handling conversions from various input formats.

## Core Concepts

### What Are Coordinate Systems?

A coordinate system defines:
- **Origin**: The (0,0,0) reference point
- **Axes orientation**: Which direction each axis points
- **Units**: Spatial units (micrometers, millimeters, etc.)
- **Handedness**: Left-handed vs right-handed coordinate systems

### Why Coordinate Systems Matter

```python
# Same physical point, different coordinate representations
point_ras = [120.5, -45.2, 78.1]  # RAS coordinates  
point_lps = [-120.5, 45.2, 78.1]  # Same point in LPS coordinates

# Without proper conversion, analyses will be incorrect!
```

**Critical for**:
- Cross-dataset comparisons
- Registration between images
- Anatomical annotation accuracy
- Transform chain composition

## Standard Coordinate Systems

### LPS (Left-Posterior-Superior)

**Used by**: ITK, SimpleITK, medical imaging standards, **aind-zarr-utils**

```
+X → Left
+Y → Posterior  
+Z → Superior
```

```python
from aind_zarr_utils.zarr import zarr_to_ants, zarr_to_sitk

# All functions output LPS coordinates
ants_img = zarr_to_ants(zarr_uri, metadata)
print(f"LPS origin: {ants_img.origin}")        # [-X, +Y, +Z]
print(f"LPS spacing: {ants_img.spacing}")      # [sx, sy, sz]

sitk_img = zarr_to_sitk(zarr_uri, metadata)  
print(f"LPS origin: {sitk_img.GetOrigin()}")   # [-X, +Y, +Z]
```

### RAS (Right-Anterior-Superior)

**Used by**: Neurological imaging, FreeSurfer, FSL, Neuroglancer

```
+X → Right
+Y → Anterior
+Z → Superior  
```

```python
# RAS to LPS conversion (flip X and Y)
def ras_to_lps(ras_point):
    """Convert RAS coordinates to LPS."""
    return [-ras_point[0], -ras_point[1], ras_point[2]]

def lps_to_ras(lps_point):
    """Convert LPS coordinates to RAS."""  
    return [-lps_point[0], -lps_point[1], lps_point[2]]
```

### Image/Voxel Coordinates

**Used by**: Array indexing, Neuroglancer annotations

```python
# Voxel indices (array coordinates)
voxel_indices = [z, y, x]  # Note: z-first ordering common

# Convert to physical coordinates
from aind_zarr_utils.annotations import annotation_indices_to_anatomical

physical_points = annotation_indices_to_anatomical(
    {"region": np.array([[z, y, x]])},
    zarr_uri, metadata,
    scale_unit="millimeter"
)
# Returns: LPS physical coordinates
```

## Coordinate Transformations

### Automatic Conversions

aind-zarr-utils handles coordinate conversions automatically:

```python
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

# Neuroglancer typically uses image coordinates  
ng_data = {"layers": {...}}

# Automatically converts image → physical → LPS
physical_points, descriptions = neuroglancer_annotations_to_anatomical(
    ng_data, zarr_uri, metadata, scale_unit="millimeter"
)

# Result: points in LPS millimeter coordinates
for layer, points in physical_points.items():
    print(f"{layer}: {points.shape} points in LPS space")
```

### Manual Transformations

When working with external data:

```python
import numpy as np

def transform_coordinates(points, from_system, to_system):
    """Transform between coordinate systems."""
    if from_system == to_system:
        return points
        
    # RAS ↔ LPS conversion
    if (from_system, to_system) in [("RAS", "LPS"), ("LPS", "RAS")]:
        # Flip X and Y axes
        transformed = points.copy()
        transformed[:, 0] *= -1  # Flip X
        transformed[:, 1] *= -1  # Flip Y
        return transformed
        
    raise ValueError(f"Unsupported conversion: {from_system} → {to_system}")

# Example usage
ras_points = np.array([[120, -45, 78], [100, -30, 82]])
lps_points = transform_coordinates(ras_points, "RAS", "LPS")
print(f"RAS: {ras_points}")
print(f"LPS: {lps_points}")
```

## Working with Different Libraries

### SimpleITK vs ANTs Differences

Both use LPS coordinates but have different array conventions:

```python
from aind_zarr_utils.zarr import zarr_to_sitk, zarr_to_ants

sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3)
ants_img = zarr_to_ants(zarr_uri, metadata, level=3)

# Same coordinate system, different axis ordering
print("SimpleITK (Fortran-style):")
print(f"  Size: {sitk_img.GetSize()}")           # (nx, ny, nz)
print(f"  Spacing: {sitk_img.GetSpacing()}")     # (sx, sy, sz)

print("ANTs (C-style):")  
print(f"  Shape: {ants_img.shape}")              # (nz, ny, nx)
print(f"  Spacing: {ants_img.spacing}")          # (sz, sy, sx)
```

### Point Transformations

```python
# SimpleITK: Transform index to physical point
index = [100, 200, 50]  # [x, y, z] order
physical = sitk_img.TransformIndexToPhysicalPoint(index)
print(f"Physical point (LPS): {physical}")

# ANTs: Manual calculation
# Note: ANTs arrays are [z, y, x] but coordinates are still LPS
origin = ants_img.origin      # [ox, oy, oz] in LPS
spacing = ants_img.spacing    # [sz, sy, sx] in LPS order
index_zyx = [50, 200, 100]    # [z, y, x] for ANTs array

# Convert to LPS physical coordinates  
physical_x = origin[0] + index_zyx[2] * spacing[2]  # X component
physical_y = origin[1] + index_zyx[1] * spacing[1]  # Y component
physical_z = origin[2] + index_zyx[0] * spacing[0]  # Z component
physical_lps = [physical_x, physical_y, physical_z]
```

## ZARR Coordinate Systems

### Multi-Resolution Coordinates

ZARR files contain multi-resolution data with consistent coordinate systems:

```python
# Different levels have different resolutions but same coordinate system
for level in [0, 3, 5]:
    stub_img, size = zarr_to_sitk_stub(zarr_uri, metadata, level=level)
    
    print(f"Level {level}:")
    print(f"  Size: {size}")
    print(f"  Spacing: {stub_img.GetSpacing()}")  # Consistent coordinate system
    print(f"  Origin: {stub_img.GetOrigin()}")    # Same origin
```

### Coordinate System Validation

```python
def validate_coordinate_system(image):
    """Check coordinate system properties."""
    if hasattr(image, 'GetDirection'):  # SimpleITK
        direction = np.array(image.GetDirection()).reshape(3, 3)
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
    else:  # ANTs
        direction = image.direction
        origin = image.origin
        spacing = image.spacing
    
    print(f"Origin: {origin}")
    print(f"Spacing: {spacing}")
    print(f"Direction matrix:\n{direction}")
    
    # Check for standard orientation
    if np.allclose(direction, np.eye(3)):
        print("✓ Standard orientation (identity matrix)")
    else:
        print("⚠ Non-standard orientation detected")
        
validate_coordinate_system(sitk_img)
```

## Common Pitfalls and Solutions

### 1. Axis Order Confusion

```python
# WRONG: Mixing coordinate conventions
neuroglancer_point = [z, y, x, t]
physical_wrong = sitk_img.TransformIndexToPhysicalPoint(neuroglancer_point[:3])

# CORRECT: Use proper conversion functions
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical
physical_correct = neuroglancer_annotations_to_anatomical(ng_data, zarr_uri, metadata)
```

### 2. Unit Mismatches

```python
# WRONG: Assuming units without checking
micrometers_data = get_metadata_coordinates()  
millimeters_target = registration_target_coords  # Different units!

# CORRECT: Explicit unit handling
from aind_zarr_utils.zarr import zarr_to_ants

# Specify desired output units
ants_img_mm = zarr_to_ants(zarr_uri, metadata, scale_unit="millimeter")
ants_img_um = zarr_to_ants(zarr_uri, metadata, scale_unit="micrometer")

print(f"Millimeter spacing: {ants_img_mm.spacing}")
print(f"Micrometer spacing: {ants_img_um.spacing}")
```

### 3. Coordinate System Assumptions

```python
# WRONG: Assuming RAS when data is LPS
external_point_ras = [120, -45, 78]  # Assumes RAS
result = register_to_target(external_point_ras)  # May be incorrect

# CORRECT: Explicit coordinate system handling
def safe_coordinate_transform(point, from_system="RAS", to_system="LPS"):
    """Safely transform coordinates with explicit systems."""
    if from_system == "RAS" and to_system == "LPS":
        return [-point[0], -point[1], point[2]]
    elif from_system == to_system:
        return point
    else:
        raise ValueError(f"Unsupported: {from_system} → {to_system}")

lps_point = safe_coordinate_transform(external_point_ras, "RAS", "LPS")
result = register_to_target(lps_point)
```

## Best Practices

### 1. Always Specify Coordinate Systems

```python
# GOOD: Explicit about coordinate systems
def process_anatomical_points(points_lps, coordinate_system="LPS"):
    """Process anatomical points.
    
    Parameters
    ----------
    points_lps : ndarray
        Points in LPS coordinate system.
    coordinate_system : str
        Coordinate system of input points (default: "LPS").
    """
    if coordinate_system != "LPS":
        points_lps = transform_to_lps(points, coordinate_system)
    
    return analyze_points(points_lps)
```

### 2. Document Coordinate Conventions

```python
def neuroglancer_to_physical(ng_points, zarr_uri, metadata):
    """Convert Neuroglancer annotations to physical coordinates.
    
    Parameters
    ----------
    ng_points : ndarray, shape (N, 4)
        Points in Neuroglancer format [z, y, x, t].
    zarr_uri : str
        URI to ZARR dataset.
    metadata : dict
        ZARR metadata containing spatial information.
        
    Returns
    -------
    physical_points : ndarray, shape (N, 3)  
        Points in LPS physical coordinates [x, y, z].
    """
    # Implementation...
```

### 3. Validate Inputs and Outputs

```python
def validate_lps_coordinates(points, name="points"):
    """Validate that points are in expected LPS format."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3) array for LPS coordinates")
    
    # Reasonable range checks for anatomical data
    if np.any(np.abs(points) > 1000):  # > 1 meter seems unreasonable
        print(f"Warning: {name} contains very large coordinates")
        
    return points

# Use in functions
def register_points(source_lps, target_lps):
    source_lps = validate_lps_coordinates(source_lps, "source_lps") 
    target_lps = validate_lps_coordinates(target_lps, "target_lps")
    # Proceed with registration...
```

### 4. Use Library Functions When Available

```python
# GOOD: Use aind-zarr-utils functions that handle conversions
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

# Handles all coordinate conversions automatically
physical_points, descriptions = neuroglancer_annotations_to_anatomical(
    neuroglancer_data, zarr_uri, metadata, scale_unit="millimeter"
)

# AVOID: Manual coordinate calculations unless necessary
# Manual conversions are error-prone and may miss edge cases
```

## Debugging Coordinate Issues

### Visual Verification

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_coordinate_systems(points_dict):
    """Visualize points in different coordinate systems."""
    fig = plt.figure(figsize=(15, 5))
    
    systems = list(points_dict.keys())
    for i, (system, points) in enumerate(points_dict.items(), 1):
        ax = fig.add_subplot(1, len(systems), i, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'{system} Coordinates')
        
    plt.tight_layout()
    plt.show()

# Example usage
ras_points = np.array([[120, -45, 78]])
lps_points = transform_coordinates(ras_points, "RAS", "LPS")

visualize_coordinate_systems({
    "RAS": ras_points,
    "LPS": lps_points
})
```

### Coordinate System Checks

```python
def diagnose_coordinate_system(image, expected_spacing=None):
    """Diagnose coordinate system properties."""
    print("=== Coordinate System Diagnosis ===")
    
    if hasattr(image, 'GetOrigin'):  # SimpleITK
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = np.array(image.GetDirection()).reshape(3, 3)
        size = image.GetSize()
        print("Image type: SimpleITK")
    else:  # ANTs  
        origin = image.origin
        spacing = image.spacing
        direction = image.direction
        size = image.shape[::-1]  # Reverse for display
        print("Image type: ANTs")
    
    print(f"Origin: {origin}")
    print(f"Spacing: {spacing}")
    print(f"Size: {size}")
    print(f"Direction matrix:\n{direction}")
    
    if expected_spacing:
        spacing_diff = np.array(spacing) - np.array(expected_spacing)
        print(f"Spacing difference from expected: {spacing_diff}")
        
    # Check coordinate system properties
    if np.allclose(direction, np.eye(3)):
        print("✓ Identity direction matrix (standard orientation)")
    else:
        print("⚠ Non-identity direction matrix")
        
    # Estimate physical bounds
    physical_bounds = [
        origin[i] + spacing[i] * size[i] for i in range(3)
    ]
    print(f"Physical bounds: {list(zip(origin, physical_bounds))}")

# Usage
diagnose_coordinate_system(ants_img, expected_spacing=[0.0144, 0.0144, 0.016])
```

## References and Further Reading

- **ITK Coordinate System**: [ITK Software Guide](https://itk.org/ItkSoftwareGuide.pdf) - Chapter on Spatial Objects
- **SimpleITK Conventions**: [SimpleITK Notebooks](https://simpleitk.readthedocs.io/en/master/notebooks.html)
- **Neuroimaging Coordinates**: [BIDS Specification](https://bids-specification.readthedocs.io/en/stable/02-common-principles.html#coordinate-systems) 
- **ANTs Documentation**: [ANTs Registration](https://github.com/ANTsX/ANTs/wiki)

## Summary

- **aind-zarr-utils standardizes on LPS coordinates** for all outputs
- **Different libraries use different conventions** (SimpleITK vs ANTs axis ordering)
- **Always specify coordinate systems** in documentation and function signatures
- **Use library functions** for conversions rather than manual calculations
- **Validate coordinate transformations** with visual checks and bounds testing
- **Handle units explicitly** to avoid scaling errors

Understanding coordinate systems is essential for accurate neuroimaging analysis. When in doubt, use aind-zarr-utils functions that handle conversions automatically and always validate your results.