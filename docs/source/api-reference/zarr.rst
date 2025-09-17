zarr module
===========

.. automodule:: aind_zarr_utils.zarr
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``zarr`` module provides the core functionality for converting ZARR datasets to ANTs and SimpleITK images. It handles multi-resolution data, coordinate system transformations, and metadata extraction while maintaining anatomical accuracy.

Main Functions
--------------

.. autosummary::
   :toctree: generated/
   
   zarr_to_ants
   zarr_to_sitk
   zarr_to_sitk_stub

Utility Functions
-----------------

.. autosummary::
   :toctree: generated/
   
   zarr_to_numpy
   direction_from_acquisition_metadata
   direction_from_nd_metadata
   compute_origin_for_corner

Key Concepts
------------

**Resolution Levels**
  ZARR files contain multiple resolution levels. Higher numbers = lower resolution:
  
  - Level 0: Full resolution
  - Level 3: Typical working resolution (default)
  - Level 5+: Preview/thumbnail resolution

**Coordinate Systems**
  All functions output LPS (Left-Posterior-Superior) coordinates:
  
  - **ANTs**: Uses LPS natively
  - **SimpleITK**: Requires axis reversal due to Fortran-style indexing

**Scale Units**
  Supports automatic unit conversion:
  
  - ``"micrometer"``: Original acquisition units
  - ``"millimeter"``: Standard for medical imaging (default)

Examples
--------

Basic ZARR to image conversion::

    from aind_zarr_utils.zarr import zarr_to_ants, zarr_to_sitk
    from aind_s3_cache.json_utils import get_json

    # Load metadata
    metadata = get_json("s3://bucket/metadata.json")
    zarr_uri = "s3://bucket/data.ome.zarr/0"

    # Convert to ANTs image (for registration/analysis)
    ants_img = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit="millimeter")
    
    # Convert to SimpleITK image (for ITK operations)  
    sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3, scale_unit="millimeter")

Memory-efficient coordinate transformations::

    from aind_zarr_utils.zarr import zarr_to_sitk_stub

    # Create stub image (minimal memory, same coordinate system)
    stub_img, size = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
    
    # Use stub for coordinate transformations without loading pixel data
    physical_point = stub_img.TransformIndexToPhysicalPoint([100, 200, 50])

Custom origin positioning::

    # Position image corner at specific anatomical location
    ants_img = zarr_to_ants(
        zarr_uri, metadata,
        set_corner="RAS",  # Right-Anterior-Superior corner
        set_corner_lps=(10.0, 5.0, 15.0)  # Position in LPS coordinates
    )