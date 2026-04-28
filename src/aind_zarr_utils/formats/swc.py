"""Reading SWC neuron tracings.

A *pure* SWC → Zarr-index converter that takes the level-0 voxel spacing
explicitly, performs unit conversion + axis reordering, and rounds to
integer indices. The orchestration layer that opens the Zarr to obtain
the spacing lives elsewhere (currently in
:mod:`aind_zarr_utils.pipeline_transformed`; eventually on the ``Asset``
façade).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aind_zarr_utils.io.metadata import _unit_conversion


def swc_data_to_indices(
    swc_point_dict: dict[str, NDArray],
    spacing_zyx_mm: tuple[float, ...] | list[float] | NDArray,
    *,
    swc_point_order: str = "zyx",
    swc_point_units: str = "micrometer",
) -> dict[str, NDArray]:
    """Convert SWC coordinates to integer Zarr (z, y, x) indices.

    Parameters
    ----------
    swc_point_dict : dict[str, NDArray]
        Mapping neuron ID → ``(N, 3)`` array of SWC point coordinates.
    spacing_zyx_mm : sequence of float
        Level-0 Zarr voxel spacing in millimeters, in ``(z, y, x)`` order.
    swc_point_order : str, optional
        Axis order of ``swc_point_dict`` arrays (any permutation of ``"zyx"``).
        Default is ``"zyx"``.
    swc_point_units : str, optional
        Length unit of the SWC coordinates. Default is ``"micrometer"``.

    Returns
    -------
    dict[str, NDArray]
        Mapping neuron ID → ``(N, 3)`` array of **integer** Zarr indices in
        ``(z, y, x)`` order.

    Raises
    ------
    ValueError
        If any input array is not ``(N, 3)``.
    """
    unit_scale = _unit_conversion(swc_point_units, "millimeter")
    order_lower = swc_point_order.lower()
    swc_to_zarr_axis_order = [order_lower.index(ax) for ax in "zyx"]
    spacing = np.asarray(spacing_zyx_mm, dtype=float)

    out: dict[str, NDArray] = {}
    for k, pts in swc_point_dict.items():
        pts_arr = np.asarray(pts)
        if pts_arr.ndim != 2 or pts_arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array for key {k}, got shape {pts_arr.shape}")
        out[k] = np.round((unit_scale * pts_arr[:, swc_to_zarr_axis_order]) / spacing).astype(int)
    return out
