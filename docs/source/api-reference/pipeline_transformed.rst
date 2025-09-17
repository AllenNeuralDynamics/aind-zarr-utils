pipeline_transformed module
============================

.. automodule:: aind_zarr_utils.pipeline_transformed
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``pipeline_transformed`` module provides utilities to reconstruct a SmartSPIM pipeline's spatial domain and apply ANTs transform chains for mapping coordinates from light sheet space to CCF (Common Coordinate Framework). It handles the complete workflow from ZARR indices to anatomically registered coordinates.


Workflow Overview
-----------------

The complete transformation pipeline follows these steps:

1. **Domain Reconstruction**: Create SimpleITK stub matching pipeline's coordinate system
2. **Transform Chain Setup**: Configure individual → template → CCF transform chains  
3. **Point Transformation**: Apply ANTs transforms to map coordinates
4. **CCF Registration**: Final coordinates in Allen CCF space

Transform Chain Structure
-------------------------

**Individual → Template**
  Maps from individual specimen to template space using:
  
  - ANTs warp (nonlinear transformation)
  - ANTs affine (linear transformation)

**Template → CCF**  
  Maps from template to Allen Common Coordinate Framework using:
  
  - Template-specific transform files
  - Pre-computed registration chains

Examples
--------

Create pipeline-compatible stub image::

    from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_anatomical_stub
    from aind_s3_cache.json_utils import get_json

    # Load processing metadata
    processing_data = get_json("s3://bucket/processing.json")
    zarr_metadata = get_json("s3://bucket/zarr_metadata.json")
    
    # Create stub that matches pipeline's coordinate system
    stub = mimic_pipeline_zarr_to_anatomical_stub(
        zarr_uri="s3://bucket/data.ome.zarr/0",
        zarr_metadata=zarr_metadata,
        processing_metadata=processing_data
    )
    
    # Stub can now be used for coordinate transformations

Transform Neuroglancer annotations to CCF::

    from aind_zarr_utils.pipeline_transformed import neuroglancer_to_ccf

    # Load Neuroglancer state
    ng_data = get_json("path/to/neuroglancer_state.json")
    
    # Transform annotations to CCF coordinates
    ccf_points, descriptions = neuroglancer_to_ccf(
        neuroglancer_data=ng_data,
        zarr_uri="s3://bucket/data.ome.zarr/0", 
        zarr_metadata=zarr_metadata,
        processing_metadata=processing_data,
        template_used="SmartSPIM-template_2024-05-16_11-26-14"
    )
    
    # Result: annotations in Allen CCF coordinate system
    for layer, points in ccf_points.items():
        print(f"{layer}: {points.shape} points in CCF space")

Manual transform chain application::

    from aind_zarr_utils.pipeline_transformed import (
        pipeline_transforms,
        pipeline_point_transforms_local_paths,
        indices_to_ccf
    )

    # Get transform paths
    individual_paths, template_paths = pipeline_transforms(
        zarr_uri, processing_data
    )
    
    # Download transforms to local cache
    local_paths, inverted = pipeline_point_transforms_local_paths(
        zarr_uri, processing_data, cache_dir="/tmp/transforms"
    )
    
    # Apply transforms to point data
    annotation_indices = {"region1": np.array([[100, 200, 50]])}
    ccf_points = indices_to_ccf(
        annotation_indices,
        zarr_metadata,
        zarr_uri,
        processing_data,
        template_used="SmartSPIM-template_2024-05-16_11-26-14"
    )

