"""AIND ZARR Utilities

Core functions for working with ZARR datasets and neuroimaging coordinates.
"""

__version__ = "0.11.4"

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
    mimic_pipeline_zarr_to_anatomical_stub,
    neuroglancer_to_ccf,
)
from .zarr import zarr_to_ants, zarr_to_sitk, zarr_to_sitk_stub

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
    # Pipeline integration
    "mimic_pipeline_zarr_to_anatomical_stub",
    "neuroglancer_to_ccf",
]
