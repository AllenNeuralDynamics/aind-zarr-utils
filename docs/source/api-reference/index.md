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

   aind_zarr_utils.annotations.annotation_indices_to_anatomical
```

## Module Documentation

```{toctree}
:maxdepth: 2

zarr
neuroglancer
annotations
pipeline_domain_selector
pipeline_transformed
```

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **zarr** | ZARR ↔ image conversion | `zarr_to_ants`, `zarr_to_sitk`, `zarr_to_sitk_stub` |
| **neuroglancer** | Neuroglancer annotation processing | `neuroglancer_annotations_to_anatomical` |
| **annotations** | Point coordinate transformation | `annotation_indices_to_anatomical` |
| **pipeline_domain_selector** | Version-specific domain corrections | `estimate_pipeline_multiscale`, `apply_overlays` |
| **pipeline_transformed** | Pipeline coordinate transformations | `neuroglancer_to_ccf`, `indices_to_ccf` |