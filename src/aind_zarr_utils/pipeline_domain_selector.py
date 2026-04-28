"""Determine and reproduce the *physical domain* that a registration pipeline used.

The implementation now lives in :mod:`aind_zarr_utils.domain.selector` and
:mod:`aind_zarr_utils.domain.overlays`. This module is a re-export shim
preserved so existing imports such as
``from aind_zarr_utils.pipeline_domain_selector import OverlaySelector``
continue to work, and so test fixtures can monkey-patch attributes (for
example, ``fix_corner_compute_origin``) at this module path. New code
should import from ``aind_zarr_utils.domain.*`` directly.

Notes
-----
- All coordinates are expressed in **ITK LPS** convention and **millimeters**.
- See :mod:`aind_zarr_utils.domain.selector` for the full design notes.
"""

from __future__ import annotations

# Make ``fix_corner_compute_origin`` available at this module path for tests
# that monkey-patch it via the legacy import location. The actual call site
# inside ``ForceCornerAnchorOverlay`` uses the binding in ``domain.overlays``.
from aind_anatomical_utils.anatomical_volume import (
    fix_corner_compute_origin as fix_corner_compute_origin,
)

from aind_zarr_utils.domain.overlays import (
    FlipIndexAxesOverlay as FlipIndexAxesOverlay,
)
from aind_zarr_utils.domain.overlays import (
    ForceCornerAnchorOverlay as ForceCornerAnchorOverlay,
)
from aind_zarr_utils.domain.overlays import (
    ForceCornerAnchorOverlayWithVersionWarning as ForceCornerAnchorOverlayWithVersionWarning,
)
from aind_zarr_utils.domain.overlays import (
    PermuteIndexAxesOverlay as PermuteIndexAxesOverlay,
)
from aind_zarr_utils.domain.overlays import (
    SetLpsWorldSpacingOverlay as SetLpsWorldSpacingOverlay,
)
from aind_zarr_utils.domain.overlays import (
    SpacingScaleOverlay as SpacingScaleOverlay,
)
from aind_zarr_utils.domain.overlays import (
    Vec3 as Vec3,
)
from aind_zarr_utils.domain.overlays import (
    _PIPELINE_MULTISCALE_FACTOR as _PIPELINE_MULTISCALE_FACTOR,
)
from aind_zarr_utils.domain.overlays import (
    _require_cardinal as _require_cardinal,
)
from aind_zarr_utils.domain.overlays import (
    estimate_pipeline_multiscale as estimate_pipeline_multiscale,
)
from aind_zarr_utils.domain.overlays import (
    lps_world_to_index_spacing_cardinal as lps_world_to_index_spacing_cardinal,
)
from aind_zarr_utils.domain.selector import (
    Overlay as Overlay,
)
from aind_zarr_utils.domain.selector import (
    OverlayRule as OverlayRule,
)
from aind_zarr_utils.domain.selector import (
    OverlaySelector as OverlaySelector,
)
from aind_zarr_utils.domain.selector import (
    _as_date as _as_date,
)
from aind_zarr_utils.domain.selector import (
    _base_rules as _base_rules,
)
from aind_zarr_utils.domain.selector import (
    apply_overlays as apply_overlays,
)
from aind_zarr_utils.domain.selector import (
    extend_selector as extend_selector,
)
from aind_zarr_utils.domain.selector import (
    get_selector as get_selector,
)
from aind_zarr_utils.domain.selector import (
    make_selector as make_selector,
)
