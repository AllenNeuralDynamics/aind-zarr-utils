neuroglancer module
===================

.. automodule:: aind_zarr_utils.neuroglancer
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``neuroglancer`` module processes annotation data from Neuroglancer, converting point annotations from Neuroglancer's coordinate system to anatomically-aware coordinates. It handles layer detection, coordinate transformations, and integration with ZARR metadata.

Functions
---------

.. autosummary::
   :toctree: generated/
   
   neuroglancer_annotations_to_indices
   neuroglancer_annotations_to_anatomical

Coordinate System Notes
-----------------------

**Neuroglancer Coordinates**
  - Order: ``[z, y, x, t]`` (time dimension optional)
  - Units: Typically in voxels/indices
  - Origin: Depends on layer configuration

**Output Coordinates**
  - Order: LPS (Left-Posterior-Superior)
  - Units: Physical units (millimeters by default)
  - Origin: Anatomically meaningful

Important Assumptions
---------------------

.. warning::
   This module assumes that Neuroglancer layers have **not** had their transforms altered from the default. If any layer has custom transforms applied, the coordinate conversions may be incorrect.

Examples
--------

Extract annotation points from Neuroglancer JSON::

    from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_indices
    from aind_s3_cache.json_utils import get_json

    # Load Neuroglancer state
    ng_data = get_json("path/to/neuroglancer_state.json")
    
    # Extract annotation indices (voxel coordinates)
    annotations, descriptions = neuroglancer_annotations_to_indices(
        ng_data,
        layer_names=["my_annotations"],  # Optional: specify layers
        return_description=True
    )
    
    print(annotations)  # Dict: {layer_name: array of [z,y,x] points}
    print(descriptions)  # Dict: {layer_name: list of descriptions}

Convert to anatomical coordinates::

    from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

    # Load ZARR metadata
    metadata = get_json("path/to/zarr_metadata.json")
    zarr_uri = "s3://bucket/data.ome.zarr/0"

    # Convert to anatomical space (LPS coordinates)
    physical_points, descriptions = neuroglancer_annotations_to_anatomical(
        ng_data,
        zarr_uri,
        metadata,
        scale_unit="millimeter",
        layer_names=["region_annotations", "landmarks"]
    )
    
    # Result: points in LPS physical coordinates
    for layer, points in physical_points.items():
        print(f"{layer}: {points.shape} points in LPS coordinates")

Using pre-created stub image::

    from aind_zarr_utils.zarr import zarr_to_sitk_stub

    # Pre-create stub for efficiency
    stub_img, _ = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
    
    # Reuse stub for multiple annotation sets
    physical_points, descriptions = neuroglancer_annotations_to_anatomical(
        ng_data,
        zarr_uri="",  # Not used when stub_image provided
        metadata={},  # Not used when stub_image provided  
        stub_image=stub_img
    )