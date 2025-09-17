# Coordinate Systems Guide

This guide explains coordinate systems used in aind-zarr-utils and neuroimaging data processing.

## Overview

Coordinate systems define how 3D positions are represented and interpreted.
Different neuroimaging tools use different conventions, making coordinate system
management crucial for accurate data processing and analysis.

aind-zarr-utils standardizes on **LPS (Left-Posterior-Superior)** coordinates
for all outputs while handling conversions from various input formats.

## Core Concepts

### What Are Anatomical Coordinate Systems?

Here is an excellent guide about anatomical coordinate systems: [https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html]

An anatomical coordinate system is used to map indices in an image or volume to
physical space. Different microscopes, MRIs, or other devices will store their
data in different ways, but as long as you know how to interpret the data, you
make sense of where different anatomical structures are. To understand which
voxel of a 3D volume corresponds to a injection in the left hemisphere of the
brain, you need to which direction in the image data (i.e. the i,j,k indices)
corresponds to moving to the subject's left

This mapping from indices to physical points requires:
- **Origin**: Where the (0,0,0) index is in physical space
- **Axes orientation**: How to translate directions in i,j,k indices to
  directions in physical space. Usually a cosine direction matrix.
- **Units**: Spatial units (micrometers, millimeters, etc.)

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

Anatomical coordinate systems are often abbreviated by three letters, indicating
which direction each axis corresponds to. These are direction pairs, L/R for
left and right, A/P for anterior and posterior, and S/I for superior (dorsal)
and inferior (ventral). The letter triplets also indicate which direction is
positive going. LPS is a way to encode physical coordinates, so that `(1, 2, 3)`
would mean +1 to the left, +2 to the posterior, and +3 to the superior of the
origin.

Confusingly, these triplets are _also_ used to describe how data are arranged in
arrays. In that case, the first letter is the **physical direction** of the
_fastest_ moving index as you iterate through the data, with the direction
indicating which way you move in anatomical space as you increase that index.

In column-major numpy arrays, `arr[i,j,k]` would be LPS if `k` (the fastest
moving index as you move through sequential positions) increases as you move to
the left, `j` increases as you move posterior, and `i` increases as you move
superior. In other programming languages or packages, like SimpleITK and ITK,
indices are assumed to be row-major, so even if the underlying data are the same
(LPS, with `k` being the fastest moving index), the data would be indexed like
`arr[k,j,i]` instead of `arr[i,j,k]`.

### LPS (Left-Posterior-Superior)

**Used by**: ITK, SimpleITK, ANTs, medical imaging standards, and
**aind-zarr-utils**


### RAS (Right-Anterior-Superior)

**Used by**: Neurological imaging, FreeSurfer, FSL, Neuroglancer

### Image/Voxel Coordinates

**Used by**: Array indexing, Neuroglancer annotations

```python
# Voxel indices (array coordinates)
# Note: z-first ordering common in python, where z is the slowest going
voxel_indices = [z, y, x]

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

aind-zarr-utils handles coordinate conversions automatically because it uses
SimpleITK and ANTs under the hood. All points will be returned as LPS.

```python
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

# Neuroglancer typically uses image coordinates
ng_data = {"layers": {...}}

# Automatically converts image inides → LPS
physical_points, descriptions = neuroglancer_annotations_to_anatomical(
    ng_data, zarr_uri, metadata, scale_unit="millimeter"
)

# Result: points in LPS millimeter coordinates
for layer, points in physical_points.items():
    print(f"{layer}: {points.shape} points in LPS space")
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
