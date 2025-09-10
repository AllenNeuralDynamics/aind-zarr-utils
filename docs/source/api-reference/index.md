# API Reference

Complete API documentation for all modules in aind-zarr-utils.

## Quick Reference

### Core Functions

```{eval-rst}
.. autosummary::
   :nosignatures:

   aind_zarr_utils.zarr.zarr_to_ants
   aind_zarr_utils.zarr.zarr_to_sitk
   aind_zarr_utils.zarr.zarr_to_sitk_stub
   aind_zarr_utils.json_utils.get_json
   aind_zarr_utils.neuroglancer.neuroglancer_annotations_to_anatomical
```

### Pipeline Functions

```{eval-rst}
.. autosummary::
   :nosignatures:

   aind_zarr_utils.pipeline_transformed.mimic_pipeline_zarr_to_anatomical_stub
   aind_zarr_utils.pipeline_transformed.neuroglancer_to_ccf
   aind_zarr_utils.pipeline_domain_selector.estimate_pipeline_multiscale
```

### Utility Functions

```{eval-rst}
.. autosummary::
   :nosignatures:

   aind_zarr_utils.uri_utils.parse_s3_uri
   aind_zarr_utils.uri_utils.join_any
   aind_zarr_utils.s3_cache.get_local_path_for_resource
   aind_zarr_utils.annotations.annotation_indices_to_anatomical
```

## Module Documentation

```{toctree}
:maxdepth: 2

zarr
json_utils
uri_utils
neuroglancer
annotations
s3_cache
pipeline_domain_selector
pipeline_transformed
```

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **zarr** | ZARR ↔ image conversion | `zarr_to_ants`, `zarr_to_sitk`, `zarr_to_sitk_stub` |
| **json_utils** | Multi-source JSON loading | `get_json`, `get_json_s3_uri` |
| **uri_utils** | Path/URI manipulation | `parse_s3_uri`, `join_any`, `as_pathlike` |
| **neuroglancer** | Neuroglancer annotation processing | `neuroglancer_annotations_to_anatomical` |
| **annotations** | Point coordinate transformation | `annotation_indices_to_anatomical` |
| **s3_cache** | S3 resource caching | `get_local_path_for_resource`, `CacheManager` |
| **pipeline_domain_selector** | Version-specific domain corrections | `estimate_pipeline_multiscale`, `apply_overlays` |
| **pipeline_transformed** | Pipeline coordinate transformations | `neuroglancer_to_ccf`, `indices_to_ccf` |