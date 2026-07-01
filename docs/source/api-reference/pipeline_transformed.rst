pipeline_transformed module
===========================

.. automodule:: aind_zarr_utils.pipeline_transformed
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``pipeline_transformed`` module contains the legacy explicit-metadata
functions for reconstructing a SmartSPIM pipeline spatial domain and applying
ANTs transform chains. New user code should prefer ``Asset`` and ``Points``:

.. code-block:: python

    from aind_zarr_utils import Asset, Points, Space

    asset = Asset.from_zarr("s3://bucket/dataset/image.ome.zarr/0")
    points = Points.from_neuroglancer(ng_state)
    ccf = asset.transform(points, to=Space.CCF_MM)

Legacy Workflow
---------------

The explicit transformation pipeline still follows these steps:

1. Reconstruct a SimpleITK stub matching the pipeline coordinate system.
2. Resolve individual-to-template and template-to-CCF transform chains.
3. Apply ANTs transforms to point arrays.
4. Return Allen CCF coordinates.

Legacy Examples
---------------

Create a pipeline-compatible stub image::

    from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_anatomical_stub

    stub, native_size_ijk = mimic_pipeline_zarr_to_anatomical_stub(
        zarr_uri,
        zarr_metadata,
        processing_metadata,
    )

Transform Neuroglancer annotations to CCF::

    from aind_zarr_utils.pipeline_transformed import neuroglancer_to_ccf

    ccf_points, descriptions = neuroglancer_to_ccf(
        neuroglancer_data=ng_state,
        zarr_uri=zarr_uri,
        zarr_metadata=zarr_metadata,
        processing_metadata=processing_metadata,
        template_used="SmartSPIM-template_2024-05-16_11-26-14",
    )

Manual transform-chain application::

    from aind_zarr_utils.pipeline_transformed import (
        indices_to_ccf,
        pipeline_point_transforms_local_paths,
        pipeline_transforms,
    )

    individual_paths, template_paths = pipeline_transforms(
        zarr_uri,
        processing_metadata,
    )

    local_paths, inverted = pipeline_point_transforms_local_paths(
        zarr_uri,
        processing_metadata,
        cache_dir="/tmp/transforms",
    )

    ccf_points = indices_to_ccf(
        {"region1": annotation_indices},
        zarr_metadata,
        zarr_uri,
        processing_metadata,
        template_used="SmartSPIM-template_2024-05-16_11-26-14",
    )
