"""Reading Neuroglancer annotation state.

The implementation now lives in :mod:`aind_zarr_utils.formats.neuroglancer`.
This module is a re-export shim preserved so that
``from aind_zarr_utils.neuroglancer import …`` and existing test fixtures
that monkey-patch ``aind_zarr_utils.neuroglancer.zarr_to_sitk_stub`` etc.
continue to work. New code should import from
``aind_zarr_utils.formats.neuroglancer`` directly.
"""

from __future__ import annotations

# Re-export the auxiliary names that test fixtures patch on this module.
from aind_zarr_utils.annotations import (
    annotation_indices_to_anatomical as annotation_indices_to_anatomical,
)
from aind_zarr_utils.formats.neuroglancer import (
    _extract_spacing as _extract_spacing,
)
from aind_zarr_utils.formats.neuroglancer import (
    _get_layer_by_name as _get_layer_by_name,
)
from aind_zarr_utils.formats.neuroglancer import (
    _process_annotation_layers as _process_annotation_layers,
)
from aind_zarr_utils.formats.neuroglancer import (
    _process_layer_and_descriptions as _process_layer_and_descriptions,
)
from aind_zarr_utils.formats.neuroglancer import (
    _resolve_layer_names as _resolve_layer_names,
)
from aind_zarr_utils.formats.neuroglancer import (
    _sanitize_source_url as _sanitize_source_url,
)
from aind_zarr_utils.formats.neuroglancer import (
    get_image_sources as get_image_sources,
)
from aind_zarr_utils.formats.neuroglancer import (
    neuroglancer_annotations_to_anatomical as neuroglancer_annotations_to_anatomical,
)
from aind_zarr_utils.formats.neuroglancer import (
    neuroglancer_annotations_to_indices as neuroglancer_annotations_to_indices,
)
from aind_zarr_utils.formats.neuroglancer import (
    neuroglancer_annotations_to_scaled as neuroglancer_annotations_to_scaled,
)
from aind_zarr_utils.zarr import (
    zarr_to_sitk_stub as zarr_to_sitk_stub,
)
