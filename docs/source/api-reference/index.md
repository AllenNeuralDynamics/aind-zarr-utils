# API Reference

Complete API documentation for aind-zarr-utils.

## Quick Reference

### Recommended API

```{eval-rst}
.. autosummary::
   :nosignatures:

   aind_zarr_utils.Asset
   aind_zarr_utils.Points
   aind_zarr_utils.Space
   aind_zarr_utils.Origin
```

### Legacy Core Functions

```{eval-rst}
.. autosummary::
   :nosignatures:

   aind_zarr_utils.zarr.zarr_to_ants
   aind_zarr_utils.zarr.zarr_to_sitk
   aind_zarr_utils.zarr.zarr_to_sitk_stub
   aind_zarr_utils.neuroglancer.neuroglancer_annotations_to_anatomical
```

### Legacy Pipeline Functions

```{eval-rst}
.. autosummary::
   :nosignatures:

   aind_zarr_utils.pipeline_transformed.mimic_pipeline_zarr_to_anatomical_stub
   aind_zarr_utils.pipeline_transformed.neuroglancer_to_ccf
   aind_zarr_utils.pipeline_domain_selector.estimate_pipeline_multiscale
```

## Module Documentation

```{toctree}
:maxdepth: 2

asset
points
origin
image
zarr
neuroglancer
annotations
pipeline_domain_selector
pipeline_transformed
```

## Module Overview

| Module | Purpose | Key API |
|--------|---------|---------|
| **asset** | Recommended asset-centric workflow | `Asset.from_zarr`, `Asset.image`, `Asset.stub`, `Asset.transform` |
| **points** | Coordinate-space tagged point arrays | `Points`, `Space` |
| **origin** | Explicit origin value type | `Origin.default`, `Origin.at`, `Origin.at_corner` |
| **image** | Shared image and pipeline-overlay internals | `apply_pipeline_overlays` |
| **zarr** | Legacy ZARR to image conversion | `zarr_to_ants`, `zarr_to_sitk`, `zarr_to_sitk_stub` |
| **neuroglancer** | Neuroglancer annotation processing | `neuroglancer_annotations_to_anatomical` |
| **annotations** | Point coordinate transformation | `annotation_indices_to_anatomical` |
| **pipeline_domain_selector** | Version-specific domain corrections | `estimate_pipeline_multiscale`, `apply_overlays` |
| **pipeline_transformed** | Legacy pipeline coordinate transformations | `neuroglancer_to_ccf`, `indices_to_ccf` |
