"""Parsing helpers for AIND ND/acquisition metadata and unit conversions.

These functions extract anatomical orientation information from
``metadata.nd.json`` (or its embedded ``acquisition`` block), and convert
between common length units used by OME-Zarr metadata.
"""

from __future__ import annotations

import numpy as np
from aind_metadata_reader import Acquisition, CoordinateFrame, Document, read_acquisition
from numpy.typing import NDArray


def _acquisition_view(metadata: dict | Acquisition) -> Acquisition:
    """Accept either recorded JSON or an explicitly selected acquisition view."""
    return metadata if isinstance(metadata, Acquisition) else read_acquisition(Document(metadata))


def anatomical_axes(metadata: dict | Acquisition | CoordinateFrame) -> tuple[list[str], list[str]]:
    """Require a unique cardinal frame, without inventing array dimension numbers.

    An Acquisition view can carry an explicit device, stream and frame-scope
    selection; a CoordinateFrame carries an exact frame selection. Zarr consumers
    match the returned names to the actual Zarr axes.
    This reads frame directions only; it does not compose image/tile transforms.
    """
    frame = metadata if isinstance(metadata, CoordinateFrame) else _acquisition_view(metadata).one_frame()
    frame.orientation.require()
    names = [axis.name for axis in frame.axes]
    if len(set(names)) != len(names) or any(name not in {"x", "y", "z"} for name in names):
        raise ValueError("Anatomical conversion requires distinct x, y, z axis names")
    return [str(name) for name in names], [str(axis.anatomical_direction) for axis in frame.axes]


def direction_from_acquisition_metadata(
    acq_metadata: dict | Acquisition,
) -> tuple[NDArray, list[str], list[str]]:
    """
    Extract direction, axes, and dimensions from acquisition metadata.

    Parameters
    ----------
    acq_metadata : dict or Acquisition
        Recorded metadata or a selected acquisition view with explicit array dimensions.

    Returns
    -------
    dimensions : ndarray
        Sorted array of dimension names in the metadata (e.g. array[0, 1, 2]).
    axes : list
        List of axis names in lowercase (e.g. 'z', 'y', 'x').
    directions : list
        List of direction codes (e.g., 'L', 'R', etc.).
    """
    view = _acquisition_view(acq_metadata)
    axes = view.array_mapping.require()
    anatomical_axes(view)
    # Preserve the recorded dimension spelling for the legacy public API.
    dimensions = np.array([axis.document.field("/dimension").value for axis in axes])
    return dimensions, [str(axis.name) for axis in axes], [str(axis.anatomical_direction) for axis in axes]


def direction_from_nd_metadata(
    nd_metadata: dict | Acquisition,
) -> tuple[NDArray, list[str], list[str]]:
    """
    Extract direction, axes, and dimensions from ND metadata.

    Parameters
    ----------
    nd_metadata : dict or Acquisition
        Aggregate metadata, acquisition core, or selected acquisition view.

    Returns
    -------
    dimensions : ndarray
        Sorted array of dimension names in the metadata (e.g. array[0, 1, 2]).
    axes : list
        List of axis names in lowercase (e.g. 'z', 'y', 'x').
    directions : list
        List of direction codes (e.g., 'L', 'R', etc.).
    """
    return direction_from_acquisition_metadata(nd_metadata)


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
