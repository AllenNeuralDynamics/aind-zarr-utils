"""Low-level OME-Zarr opening and per-level metadata extraction.

These helpers wrap ``ome-zarr-py``'s ``Reader``/``Node`` interface and turn its
multiscale metadata into the spacing/size/axis tuples the rest of the package
operates on. They do not depend on AIND-specific metadata; for that, see
:mod:`aind_zarr_utils.io.metadata`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from ome_zarr.io import parse_url  # type: ignore[import-untyped]
from ome_zarr.reader import Node, Reader  # type: ignore[import-untyped]

from aind_zarr_utils.io.metadata import _unit_conversion


def ensure_native_endian(a: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    """Return ``a`` with native-endian dtype.

    - View if already native or byteorder-agnostic.
    - Copy only when endianness must change or fields are mixed.
    - With ``inplace=True``, mutate array when it's safe.
    """
    a = np.asarray(a)
    dt = a.dtype

    # -------- Plain (non-structured) dtype --------
    if dt.fields is None:
        # Already native or endian-agnostic ('|')
        if dt.byteorder in ("|", "="):
            return a
        # Needs endian fix
        if inplace:
            if not a.flags.writeable:
                raise ValueError("Array is not writeable; cannot convert in-place.")
            a.byteswap(inplace=True)
            a.dtype = dt.newbyteorder("=")  # type: ignore[misc]
            return a
        return a.astype(dt.newbyteorder("="), copy=False)

    # -------- Structured dtype --------
    # Collect byteorders of endian-aware fields ('<' or '>')
    field_orders = {subdt.byteorder for (subdt, *_) in dt.fields.values() if subdt.byteorder in ("<", ">")}

    if not field_orders:
        # Either all fields are native '=' or byteorder-agnostic '|'
        return a

    if len(field_orders) > 1:
        # Mixed endianness across fields → let NumPy fix per-field via astype
        # (copy)
        return a.astype(dt.newbyteorder("="), copy=True)

    # Homogeneous non-native across fields ('<' OR '>')
    if inplace:
        if not a.flags.writeable:
            raise ValueError("Array is not writeable; cannot convert in-place.")
        a.byteswap(inplace=True)
        a.dtype = dt.newbyteorder("=")  # type: ignore[misc]
        return a

    return a.astype(dt.newbyteorder("="), copy=False)


def _open_zarr(uri: str) -> tuple[Node, dict]:
    """
    Open a ZARR file and retrieve its metadata.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.

    Returns
    -------
    image_node : ome_zarr.reader.Node
        The image node of the ZARR file.
    zarr_meta : dict
        Metadata of the ZARR file.
    """
    reader = Reader(parse_url(uri))

    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]
    zarr_meta = image_node.metadata
    return image_node, zarr_meta


def zarr_to_numpy(uri: str, level: int = 3, ensure_native_endianness: bool = False) -> tuple[NDArray, dict, int]:
    """
    Convert a ZARR file to a NumPy array.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    level : int, optional
        Resolution level to read, by default 3.
    ensure_native_endianness : bool, optional
        Whether to ensure native endianness of the returned array, by default
        False.

    Returns
    -------
    arr_data : ndarray
        NumPy array of the image data.
    zarr_meta : dict
        Metadata of the ZARR file.
    level : int
        Resolution level used.
    """
    image_node, zarr_meta = _open_zarr(uri)
    arr_data = image_node.data[level].compute()
    if ensure_native_endianness:
        arr_data = ensure_native_endian(arr_data, inplace=True)
    return arr_data, zarr_meta, level


def _zarr_to_scaled(
    uri: str,
    *,
    level: int = 3,
    scale_unit: str = "millimeter",
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[Node, set[int], list[str], list[float], list[int]]:
    """
    Extract scaled coordinate information from a ZARR file.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    level : int, optional
        Resolution level to read, by default 3.
    scale_unit : str, optional
        Unit for scaling, by default "millimeter".
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    image_node : ome_zarr.reader.Node
        The image node of the ZARR file.
    rej_axes : set
        Rejected axes indices.
    spacing : list
        List of spacing values.
    size : list
        List of size values.
    original_to_subset_axes_map : dict
        Mapping from original axes to subset axes.
    """
    # Create the zarr reader
    if opened_zarr is None:
        image_node, zarr_meta = _open_zarr(uri)
    else:
        image_node, zarr_meta = opened_zarr
    scale = np.array(zarr_meta["coordinateTransformations"][level][0]["scale"])
    original_zarr_axes = zarr_meta["axes"]
    spatial_dims = set(["x", "y", "z"])
    original_to_subset_axes_map = {}  # sorted
    i = 0
    for j, ax in enumerate(original_zarr_axes):
        ax_name = ax["name"]
        if ax_name in spatial_dims:
            original_to_subset_axes_map[j] = i
            i += 1
    rej_axes = set(range(len(original_zarr_axes))) - set(original_to_subset_axes_map.keys())
    spacing = []
    size = []
    kept_zarr_axes = []
    dask_shape = image_node.data[level].shape
    for i in original_to_subset_axes_map.keys():
        kept_zarr_axes.append(original_zarr_axes[i]["name"])
        scale_factor = _unit_conversion(original_zarr_axes[i]["unit"], scale_unit)
        spacing.append(scale_factor * scale[i])
        size.append(dask_shape[i])
    return image_node, rej_axes, kept_zarr_axes, spacing, size
