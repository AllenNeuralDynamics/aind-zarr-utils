"""Parsing helpers for AIND ND/acquisition metadata and unit conversions.

These functions extract anatomical orientation information from
``metadata.nd.json`` (or its embedded ``acquisition`` block), and convert
between common length units used by OME-Zarr metadata.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def direction_from_acquisition_metadata(
    acq_metadata: dict,
) -> tuple[NDArray, list[str], list[str]]:
    """
    Extract direction, axes, and dimensions from acquisition metadata.

    Parameters
    ----------
    acq_metadata : dict
        Acquisition metadata

    Returns
    -------
    dimensions : ndarray
        Sorted array of dimension names in the metadata (e.g. array[0, 1, 2]).
    axes : list
        List of axis names in lowercase (e.g. 'z', 'y', 'x').
    directions : list
        List of direction codes (e.g., 'L', 'R', etc.).
    """
    axes_dict = {d["dimension"]: d for d in acq_metadata["axes"]}
    dimensions = np.sort(np.array(list(axes_dict.keys())))
    axes = []
    directions = []
    for i in dimensions:
        axes.append(axes_dict[i]["name"].lower())
        directions.append(axes_dict[i]["direction"].split("_")[-1][0].upper())
    return dimensions, axes, directions


def direction_from_nd_metadata(
    nd_metadata: dict,
) -> tuple[NDArray, list[str], list[str]]:
    """
    Extract direction, axes, and dimensions from ND metadata.

    Parameters
    ----------
    nd_metadata : dict
        ND metadata

    Returns
    -------
    dimensions : ndarray
        Sorted array of dimension names in the metadata (e.g. array[0, 1, 2]).
    axes : list
        List of axis names in lowercase (e.g. 'z', 'y', 'x').
    directions : list
        List of direction codes (e.g., 'L', 'R', etc.).
    """
    return direction_from_acquisition_metadata(nd_metadata["acquisition"])


def _units_to_meter(unit: str) -> float:
    """
    Convert a unit of length to meters.

    Parameters
    ----------
    unit : str
        Unit of length (e.g., 'micrometer', 'millimeter').

    Returns
    -------
    float
        Conversion factor to meters.

    Raises
    ------
    ValueError
        If the unit is unknown.
    """
    if unit == "micrometer":
        return 1e-6
    elif unit == "millimeter":
        return 1e-3
    elif unit == "centimeter":
        return 1e-2
    elif unit == "meter":
        return 1.0
    elif unit == "kilometer":
        return 1e3
    else:
        raise ValueError(f"Unknown unit: {unit}")


def _unit_conversion(src: str, dst: str) -> float:
    """
    Convert between two units of length.

    Parameters
    ----------
    src : str
        Source unit.
    dst : str
        Destination unit.

    Returns
    -------
    float
        Conversion factor from src to dst.
    """
    if src == dst:
        return 1.0
    src_meters = _units_to_meter(src)
    dst_meters = _units_to_meter(dst)
    return src_meters / dst_meters
