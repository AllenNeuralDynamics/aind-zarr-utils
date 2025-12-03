"""AIND ZARR Utilities

Core functions for working with ZARR datasets and neuroimaging coordinates.
"""

__version__ = "0.11.6"

# Core ZARR conversion
# Basic coordinate transformation
from .annotations import annotation_indices_to_anatomical

# Neuroglancer annotation processing
from .neuroglancer import (
    neuroglancer_annotations_to_anatomical,
    neuroglancer_annotations_to_indices,
)

# Pipeline integration
from .pipeline_transformed import (
    indices_to_ccf_auto_metadata,
    mimic_pipeline_zarr_to_anatomical_stub,
    neuroglancer_to_ccf,
    neuroglancer_to_ccf_auto_metadata,
    swc_data_to_ccf_auto_metadata,
)
from .zarr import (
    scaled_points_to_indices,
    zarr_to_ants,
    zarr_to_sitk,
    zarr_to_sitk_stub,
)

__all__ = [
    # Core ZARR conversion
    "zarr_to_ants",
    "zarr_to_sitk",
    "zarr_to_sitk_stub",
    # Neuroglancer processing
    "neuroglancer_annotations_to_anatomical",
    "neuroglancer_annotations_to_indices",
    # Coordinate transformation
    "annotation_indices_to_anatomical",
    "scaled_points_to_indices",
    # Pipeline integration
    "mimic_pipeline_zarr_to_anatomical_stub",
    "neuroglancer_to_ccf",
    "neuroglancer_to_ccf_auto_metadata",
    "swc_data_to_ccf_auto_metadata",
    "indices_to_ccf_auto_metadata",
]
