"""``Points``, ``Space``, and the transform graph.

This module makes coordinate spaces first-class. Each :class:`Points`
instance carries a :class:`Space` tag identifying *what* its values
mean, and the package's :class:`~aind_zarr_utils.asset.Asset` exposes
a single ``transform(points, to=Space.X)`` method that walks a small
graph of named edges to project them.

Graph topology — a tree rooted at ``ZARR_INDICES``::

    LS_SCALED_MM ─── ZARR_INDICES ──┬── LS_ANATOMICAL_MM
                                     └── LS_PIPELINE_ANATOMICAL_MM ── CCF_MM

The two anatomical spaces are *both* attached to ``ZARR_INDICES`` rather
than chained, because they are derived from different headers (the raw
Zarr metadata vs. the pipeline-corrected one); going from one to the
other is most naturally expressed as a round-trip through indices.

Each edge function is a small wrapper around an existing low-level
helper, with the asset's cached ``opened_zarr`` and ``transforms``
threaded through so a multi-hop walk doesn't re-open the Zarr or
re-download transform files.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import SimpleITK as sitk
from aind_registration_utils.ants import (
    apply_ants_transforms_to_point_arr,
)
from numpy.typing import NDArray

from aind_zarr_utils.annotations import annotation_indices_to_anatomical
from aind_zarr_utils.formats.neuroglancer import (
    neuroglancer_annotations_to_indices,
)
from aind_zarr_utils.io.metadata import _unit_conversion
from aind_zarr_utils.io.zarr import _zarr_to_scaled
from aind_zarr_utils.zarr import scaled_points_to_indices

if TYPE_CHECKING:
    from aind_zarr_utils.asset import Asset


class Space(Enum):
    """Coordinate spaces a :class:`Points` instance can live in.

    Attributes
    ----------
    ZARR_INDICES
        Continuous (sub-voxel) ``(z, y, x)`` indices into the Zarr at
        level 0.
    LS_SCALED_MM
        Voxel indices multiplied by per-axis level-0 spacing, in
        millimeters and ``(z, y, x)`` order. No anatomical orientation
        is encoded; this is the format SWC files are typically saved in.
    LS_ANATOMICAL_MM
        ITK LPS millimeters derived from the *raw* Zarr metadata header
        (no pipeline overlay corrections).
    LS_PIPELINE_ANATOMICAL_MM
        ITK LPS millimeters derived from the *pipeline-corrected* header.
        This is the space ANTs registration was trained against.
    CCF_MM
        Allen Common Coordinate Framework, in LPS millimeters.
    """

    ZARR_INDICES = "zarr_indices"
    LS_SCALED_MM = "ls_scaled_mm"
    LS_ANATOMICAL_MM = "ls_anatomical_mm"
    LS_PIPELINE_ANATOMICAL_MM = "ls_pipeline_anatomical_mm"
    CCF_MM = "ccf_mm"


@dataclass(frozen=True, slots=True, eq=False)
class Points:
    """A set of named ``(N, 3)`` point arrays, all in the same coordinate space.

    Attributes
    ----------
    values : dict[str, NDArray]
        Mapping ``layer_name → (N, 3)`` array. The interpretation of
        the columns depends on ``space``.
    space : Space
        What the values mean. See :class:`Space`.
    descriptions : dict[str, list[str | None]] or None, optional
        Per-point descriptions when relevant (e.g. Neuroglancer
        annotation labels). Keyed by layer name; each list parallels
        the ``(N, 3)`` array for that layer.

    Notes
    -----
    Frozen for reassignment-safety; the inner ``dict`` and arrays are
    not deep-copied — by convention, treat instances as immutable.
    """

    values: dict[str, NDArray]
    space: Space
    descriptions: dict[str, list[str | None]] | None = None

    @classmethod
    def from_neuroglancer(
        cls,
        ng_state: dict[str, Any],
        layer_names: str | list[str] | None = None,
        return_description: bool = True,
    ) -> Points:
        """Build a :class:`Points` from Neuroglancer annotation state.

        Returns points in :attr:`Space.ZARR_INDICES`. Wraps
        :func:`~aind_zarr_utils.formats.neuroglancer.neuroglancer_annotations_to_indices`
        and converts its description ``NDArray[object]`` values into
        ``list[str | None]`` for cleaner typing.
        """
        annotations, raw_desc = neuroglancer_annotations_to_indices(
            ng_state,
            layer_names=layer_names,
            return_description=return_description,
        )
        descriptions: dict[str, list[str | None]] | None = None
        if raw_desc is not None:
            descriptions = {k: list(arr) for k, arr in raw_desc.items()}
        return cls(values=annotations, space=Space.ZARR_INDICES, descriptions=descriptions)

    @classmethod
    def from_swc(
        cls,
        swc_data: dict[str, NDArray] | NDArray,
        *,
        axis_order: str = "zyx",
        units: str = "micrometer",
    ) -> Points:
        """Build a :class:`Points` from raw SWC neuron coordinates.

        Performs unit conversion to millimeters and reorders the columns
        to ``(z, y, x)``; no Zarr access is required. The resulting
        :attr:`Space.LS_SCALED_MM` points can then be projected onto
        Zarr indices, anatomical space, or CCF via
        :meth:`~aind_zarr_utils.asset.Asset.transform`.

        Parameters
        ----------
        swc_data : NDArray or dict[str, NDArray]
            ``(N, 3)`` array (single neuron) or mapping
            ``neuron_id → (N, 3)`` array.
        axis_order : str, optional
            Axis order of the input columns (any permutation of
            ``"zyx"``). Default ``"zyx"``.
        units : str, optional
            Length unit of the input coordinates. Default
            ``"micrometer"``.
        """
        if isinstance(swc_data, np.ndarray):
            swc_data = {"_": swc_data}

        unit_scale = _unit_conversion(units, "millimeter")
        order_lower = axis_order.lower()
        swc_to_zyx = [order_lower.index(ax) for ax in "zyx"]

        out: dict[str, NDArray] = {}
        for k, pts in swc_data.items():
            arr = np.asarray(pts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"Expected (N, 3) array for key {k}, got shape {arr.shape}")
            out[k] = unit_scale * arr[:, swc_to_zyx]

        return cls(values=out, space=Space.LS_SCALED_MM, descriptions=None)


# ---------------------------------------------------------- transform graph ---


# Edge functions: each takes (asset, points-with-known-space) and returns
# points in the next space. The dispatch table at the bottom of the module
# maps (src, dst) pairs to these functions.


def _indices_to_scaled(asset: Asset, pts: Points) -> Points:
    _, _, _, spacing_raw, _ = _zarr_to_scaled(asset.zarr_uri, level=0, opened_zarr=asset.opened_zarr)
    spacing = np.asarray(spacing_raw, dtype=float)
    out = {layer: arr * spacing for layer, arr in pts.values.items()}
    return Points(values=out, space=Space.LS_SCALED_MM, descriptions=pts.descriptions)


def _scaled_to_indices(asset: Asset, pts: Points) -> Points:
    out = scaled_points_to_indices(pts.values, asset.zarr_uri, opened_zarr=asset.opened_zarr)
    return Points(values=out, space=Space.ZARR_INDICES, descriptions=pts.descriptions)


def _indices_to_base_anat(asset: Asset, pts: Points) -> Points:
    base_stub, _ = asset.stub(pipeline=False)
    out = annotation_indices_to_anatomical(base_stub, pts.values)
    return Points(values=out, space=Space.LS_ANATOMICAL_MM, descriptions=pts.descriptions)


def _base_anat_to_indices(asset: Asset, pts: Points) -> Points:
    base_stub, _ = asset.stub(pipeline=False)
    return _physical_to_indices(base_stub, pts)


def _indices_to_pipeline_anat(asset: Asset, pts: Points) -> Points:
    pipeline_stub, _ = asset.stub(pipeline=True)
    out = annotation_indices_to_anatomical(pipeline_stub, pts.values)
    return Points(
        values=out,
        space=Space.LS_PIPELINE_ANATOMICAL_MM,
        descriptions=pts.descriptions,
    )


def _pipeline_anat_to_indices(asset: Asset, pts: Points) -> Points:
    pipeline_stub, _ = asset.stub(pipeline=True)
    return _physical_to_indices(pipeline_stub, pts)


def _physical_to_indices(stub: sitk.Image, pts: Points) -> Points:
    """Shared implementation for {ANATOMICAL, PIPELINE_ANATOMICAL} → INDICES.

    Mirrors the math from the legacy ``ccf_to_indices``: each LPS point
    is mapped to a continuous ``(i, j, k)`` index via SimpleITK, then
    reversed to ``(z, y, x)`` to match the package's index convention.
    """
    out: dict[str, NDArray] = {}
    for layer, arr in pts.values.items():
        layer_indices = []
        for point in arr:
            cont_idx = stub.TransformPhysicalPointToContinuousIndex(point.astype(np.float64))
            layer_indices.append(np.array(cont_idx)[::-1])
        out[layer] = np.array(layer_indices)
    return Points(
        values=out,
        space=Space.ZARR_INDICES,
        descriptions=pts.descriptions,
    )


def _pipeline_anat_to_ccf(asset: Asset, pts: Points) -> Points:
    """Apply the pipeline's *point* transform chain (per-layer, no batching).

    Matches the existing ``indices_to_ccf`` behaviour: ANTs is called
    once per layer.
    """
    paths = asset.transforms
    out: dict[str, NDArray] = {}
    for layer, arr in pts.values.items():
        out[layer] = apply_ants_transforms_to_point_arr(
            arr,
            transform_list=paths.point_paths,
            whichtoinvert=paths.point_invert,
        )
    return Points(values=out, space=Space.CCF_MM, descriptions=pts.descriptions)


def _ccf_to_pipeline_anat(asset: Asset, pts: Points) -> Points:
    """Apply the pipeline's *image* transform chain (batched across layers).

    Matches the existing ``ccf_to_indices`` behaviour: all layers are
    concatenated into one ANTs call to amortise per-call overhead, then
    re-segregated.
    """
    paths = asset.transforms

    layer_names = list(pts.values.keys())
    if not layer_names:
        return Points(
            values={},
            space=Space.LS_PIPELINE_ANATOMICAL_MM,
            descriptions=pts.descriptions,
        )
    layer_sizes = [len(pts.values[name]) for name in layer_names]
    all_pts = np.vstack([pts.values[name] for name in layer_names])
    all_anat = apply_ants_transforms_to_point_arr(
        all_pts,
        transform_list=paths.image_paths,
        whichtoinvert=paths.image_invert,
    )
    out: dict[str, NDArray] = {}
    start = 0
    for name, size in zip(layer_names, layer_sizes):
        end = start + size
        out[name] = all_anat[start:end]
        start = end
    return Points(
        values=out,
        space=Space.LS_PIPELINE_ANATOMICAL_MM,
        descriptions=pts.descriptions,
    )


# Adjacency list for the tree (undirected); each entry maps a Space to
# its neighbours.
_ADJ: dict[Space, tuple[Space, ...]] = {
    Space.LS_SCALED_MM: (Space.ZARR_INDICES,),
    Space.ZARR_INDICES: (
        Space.LS_SCALED_MM,
        Space.LS_ANATOMICAL_MM,
        Space.LS_PIPELINE_ANATOMICAL_MM,
    ),
    Space.LS_ANATOMICAL_MM: (Space.ZARR_INDICES,),
    Space.LS_PIPELINE_ANATOMICAL_MM: (
        Space.ZARR_INDICES,
        Space.CCF_MM,
    ),
    Space.CCF_MM: (Space.LS_PIPELINE_ANATOMICAL_MM,),
}

# Edge dispatch table. Each entry is a directed (src → dst) edge; the
# reverse direction is registered separately so the two halves can use
# different ANTs chains (point vs image direction).
_EDGES: dict[tuple[Space, Space], Callable[[Asset, Points], Points]] = {
    (Space.ZARR_INDICES, Space.LS_SCALED_MM): _indices_to_scaled,
    (Space.LS_SCALED_MM, Space.ZARR_INDICES): _scaled_to_indices,
    (Space.ZARR_INDICES, Space.LS_ANATOMICAL_MM): _indices_to_base_anat,
    (Space.LS_ANATOMICAL_MM, Space.ZARR_INDICES): _base_anat_to_indices,
    (Space.ZARR_INDICES, Space.LS_PIPELINE_ANATOMICAL_MM): _indices_to_pipeline_anat,
    (Space.LS_PIPELINE_ANATOMICAL_MM, Space.ZARR_INDICES): _pipeline_anat_to_indices,
    (Space.LS_PIPELINE_ANATOMICAL_MM, Space.CCF_MM): _pipeline_anat_to_ccf,
    (Space.CCF_MM, Space.LS_PIPELINE_ANATOMICAL_MM): _ccf_to_pipeline_anat,
}


def _path(src: Space, dst: Space) -> list[Space]:
    """Return the unique path through the transform graph from ``src`` to ``dst``.

    The graph is a tree, so the path is unique. Implemented as BFS for
    extensibility — if extra spaces are added in future commits the same
    code keeps working.
    """
    if src == dst:
        return [src]
    queue: deque[tuple[Space, list[Space]]] = deque([(src, [src])])
    visited: set[Space] = {src}
    while queue:
        node, trail = queue.popleft()
        for neighbour in _ADJ[node]:
            if neighbour in visited:
                continue
            new_trail = trail + [neighbour]
            if neighbour == dst:
                return new_trail
            visited.add(neighbour)
            queue.append((neighbour, new_trail))
    raise ValueError(f"No path in transform graph from {src} to {dst}")


def transform_points(asset: Asset, points: Points, *, to: Space) -> Points:
    """Project ``points`` to ``to``, walking the transform graph step by step.

    Parameters
    ----------
    asset : Asset
        The acquisition context. Provides the cached opened Zarr,
        pipeline-corrected stubs, and ANTs transform-chain paths.
    points : Points
        Source points. ``points.space`` is the starting node.
    to : Space
        Destination space.

    Returns
    -------
    Points
        New :class:`Points` with the same ``values`` keys (and
        descriptions, if any) projected to ``to``.
    """
    if points.space == to:
        return points
    path = _path(points.space, to)
    current = points
    for src, dst in zip(path[:-1], path[1:]):
        edge = _EDGES.get((src, dst))
        if edge is None:
            raise RuntimeError(f"No edge function registered for {src} → {dst}")
        current = edge(asset, current)
    return current
