pipeline_domain_selector module
===============================

.. automodule:: aind_zarr_utils.pipeline_domain_selector
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``pipeline_domain_selector`` module provides a sophisticated system for reproducing the exact physical domain (coordinate system) that a registration pipeline used, including historical and buggy variants. This ensures that voxel indices from ZARR files are mapped to the correct anatomical coordinates before applying transforms.

Core Classes
------------

.. autosummary::
   :toctree: generated/
   
   Header
   OverlayRule
   OverlaySelector

Key Functions
-------------

.. autosummary::
   :toctree: generated/
   
   get_selector
   apply_overlays
   estimate_pipeline_multiscale

Overlay Functions
-----------------

.. autosummary::
   :toctree: generated/
   
   overlay_ras_corner_anchor
   overlay_spacing_fix
   overlay_axis_permutation

Design Principles
-----------------

**Immutable Headers**
  The :class:`Header` class is immutable, ensuring reproducible coordinate systems.

**Composable Overlays**
  Small, pure functions modify headers in deterministic order:
  
  1. Base header from ZARR metadata
  2. Apply version-specific overlays
  3. Result matches pipeline's coordinate system

**Version-based Selection**
  :class:`OverlaySelector` chooses overlays based on:
  
  - Pipeline version (PEP 440 specifiers)
  - Acquisition date ranges  
  - Arbitrary metadata predicates

Coordinate Convention
---------------------

All coordinates use **ITK LPS convention** in **millimeters**:

- **L**: Left = +X
- **P**: Posterior = +Y  
- **S**: Superior = +Z

Direction matrices are 3×3, flattened row-major. **Columns** represent the LPS-world unit vectors of image index axes.

Examples
--------

Basic header creation and modification::

    from aind_zarr_utils.pipeline_domain_selector import Header, overlay_spacing_fix
    import numpy as np

    # Create base header
    header = Header(
        origin=(0.0, 0.0, 0.0),
        spacing=(0.01, 0.01, 0.016),  # mm
        direction=np.eye(3),
        size_ijk=(1024, 1024, 512)
    )

    # Apply spacing correction overlay
    fixed_header = overlay_spacing_fix(
        header, 
        spacing=(0.0144, 0.0144, 0.016)  # Corrected spacing
    )

Version-based overlay selection::

    from aind_zarr_utils.pipeline_domain_selector import get_selector, apply_overlays

    # Get overlays for specific pipeline version
    selector = get_selector()
    overlays = selector.select("0.0.25", acquisition_date=None, metadata={})
    
    # Apply overlays to base header
    corrected_header = apply_overlays(base_header, overlays)

Pipeline-specific coordinate system estimation::

    from aind_zarr_utils.pipeline_domain_selector import estimate_pipeline_multiscale

    # Estimate coordinate system from processing metadata
    header = estimate_pipeline_multiscale(
        zarr_uri="s3://bucket/data.ome.zarr/0",
        zarr_metadata=zarr_meta,
        processing_metadata=processing_data,
        level=0
    )
    
    # Result matches what the pipeline would have produced
    print(f"Origin: {header.origin}")
    print(f"Spacing: {header.spacing}")

Custom overlay creation::

    def custom_flip_overlay(header: Header) -> Header:
        \"\"\"Example: flip X axis direction.\"\"\"
        new_direction = header.direction.copy()
        new_direction[:, 0] *= -1  # Flip first column (X axis)
        return header.replace(direction=new_direction)

    # Apply custom overlay
    flipped_header = custom_flip_overlay(original_header)