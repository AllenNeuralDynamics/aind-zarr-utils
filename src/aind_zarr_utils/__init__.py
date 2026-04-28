"""AIND ZARR Utilities.

Core functions for working with ZARR datasets and neuroimaging coordinates.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aind-zarr-utils")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Basic coordinate transformation
from .annotations import annotation_indices_to_anatomical

# New asset-centric API (introduced in the refactor that begins at 0.15.0).
# These will become the primary public surface; the free functions below
# remain available and will continue to work through one minor release.
from .asset import Asset

# Neuroglancer annotation processing
from .neuroglancer import (
    neuroglancer_annotations_to_anatomical,
    neuroglancer_annotations_to_indices,
)
from .origin import Origin

# Pipeline integration
from .pipeline_transformed import (
    ccf_to_indices,
    ccf_to_indices_auto_metadata,
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
    # New asset-centric API
    "Asset",
    "Origin",
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
    "ccf_to_indices",
    "ccf_to_indices_auto_metadata",
]
