"""
Module for turning ZARRs into ants images and vice versa.
"""

from __future__ import annotations

from itertools import product
from typing import Optional, Sequence

import ants
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from ome_zarr.io import parse_url
from ome_zarr.reader import Node, Reader


def direction_from_acquisition_metadata(
    acq_metadata: dict,
) -> tuple[NDArray, list[str], list[str]]:
    """
    Extracts direction, axes, and dimensions from acquisition metadata.

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
    Extracts direction, axes, and dimensions from ND metadata.

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
    Converts a unit of length to meters.

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
    Converts between two units of length.

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


def _open_zarr(uri: str) -> tuple[Node, dict]:
    """
    Opens a ZARR file and retrieves its metadata.

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


def zarr_to_numpy(uri: str, level: int = 3) -> tuple[NDArray, dict, int]:
    """
    Converts a ZARR file to a NumPy array.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    level : int, optional
        Resolution level to read, by default 3.

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
    return arr_data, zarr_meta, level


def _zarr_to_global(
    uri: str,
    *,
    level: int = 3,
    scale_unit: str = "millimeter",
    opened_zarr: Optional[tuple[Node, dict]] = None,
) -> tuple[Node, set[int], list[str], list[float], list[int]]:
    """
    Extracts global information from a ZARR file.

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
    rej_axes = set(range(len(original_zarr_axes))) - set(
        original_to_subset_axes_map.keys()
    )
    spacing = []
    size = []
    kept_zarr_axes = []
    dask_shape = image_node.data[level].shape
    for i in original_to_subset_axes_map.keys():
        kept_zarr_axes.append(original_zarr_axes[i]["name"])
        scale_factor = _unit_conversion(
            original_zarr_axes[i]["unit"], scale_unit
        )
        spacing.append(scale_factor * scale[i])
        size.append(dask_shape[i])
    return image_node, rej_axes, kept_zarr_axes, spacing, size


def _zarr_to_anatomical(
    uri: str,
    nd_metadata: dict,
    *,
    level: int = 3,
    scale_unit: str = "millimeter",
    opened_zarr: Optional[tuple[Node, dict]] = None,
) -> tuple[Node, set[int], list[str], list[float], list[int]]:
    """
    Extracts anatomical information from a ZARR file.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    nd_metadata : dict
        Neural Dynamics metadata.
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
    dirs : list
        List of direction codes.
    spacing : list
        List of spacing values.
    size : list
        List of size values.
    """
    # Get direction metadata
    _, axes, directions = direction_from_nd_metadata(nd_metadata)
    metadata_axes_to_dir = {a: d for a, d in zip(axes, directions)}
    image_node, rej_axes, zarr_axes, spacing, size = _zarr_to_global(
        uri, level=level, scale_unit=scale_unit, opened_zarr=opened_zarr
    )
    dirs = [metadata_axes_to_dir[a] for a in zarr_axes]
    return image_node, rej_axes, dirs, spacing, size


def _zarr_to_numpy_anatomical(
    uri: str,
    nd_metadata: dict,
    level: int = 3,
    scale_unit: str = "millimeter",
) -> tuple[NDArray, list[str], list[float], list[int]]:
    """
    Converts a ZARR file to a NumPy array with anatomical information.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    nd_metadata : dict
        Neural Dynamics metadata.
    level : int, optional
        Resolution level to read, by default 3.
    scale_unit : str, optional
        Unit for scaling, by default "millimeter".

    Returns
    -------
    arr_data_spatial : ndarray
        NumPy array of the image data with spatial dimensions.
    dirs : list
        List of direction codes.
    spacing : list
        List of spacing values.
    size : list
        List of size values.
    """
    image_node, rej_axes, dirs, spacing, size = _zarr_to_anatomical(
        uri, nd_metadata, level=level, scale_unit=scale_unit
    )
    arr_data = image_node.data[level].compute()
    arr_data_spatial = np.squeeze(arr_data, axis=tuple(rej_axes))
    return arr_data_spatial, dirs, spacing, size


def zarr_to_ants(
    uri: str,
    nd_metadata: dict,
    level: int = 3,
    scale_unit: str = "millimeter",
    set_origin: Optional[tuple[float, float, float]] = None,
    set_corner: Optional[str] = None,
    set_corner_lps: Optional[tuple[float, float, float]] = None,
) -> ants.ANTsImage:
    """
    Converts a ZARR file to an ANTs image.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    nd_metadata : dict
        Neural Dynamics metadata.
    level : int, optional
        Resolution level to read, by default 3.
    scale_unit : str, optional
        Unit for scaling, by default "millimeter".
    set_origin : tuple, optional
        Origin of the image, by default None. Exclusive of set_corner and
        set_corner_lps.
    set_corner : str, optional
        Which corner to use, by default None. If set, must specify both
        set_corner and set_corner_lps, exclusive of set_origin.
    set_corner_lps: tuple, optional
        Coordinates of the corner in LPS. If set, must specify both set_corner
        and set_corner_lps, exclusive of set_origin.

    Returns
    -------
    ants.ANTsImage
        ANTs image object.
    """
    (arr_data_spatial, dirs, spacing, size) = _zarr_to_numpy_anatomical(
        uri, nd_metadata, level=level, scale_unit=scale_unit
    )

    # Get direction metadata
    dir_str = "".join(dirs)
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        dir_str
    )
    dir_mat = np.array(dir_tup).reshape((3, 3))
    origin_type = _origin_args_check(set_origin, set_corner, set_corner_lps)
    if origin_type == "origin":
        assert set_origin is not None
        origin = set_origin
    elif origin_type == "corner":
        assert set_corner_lps is not None and set_corner is not None
        origin = compute_origin_for_corner(
            size, spacing, dir_tup, set_corner_lps, set_corner
        )[0]
    else:
        origin = (0.0, 0.0, 0.0)
    ants_image = ants.from_numpy(
        arr_data_spatial, spacing=spacing, direction=dir_mat, origin=origin
    )
    return ants_image


def zarr_to_sitk(
    uri: str,
    nd_metadata: dict,
    level: int = 3,
    scale_unit: str = "millimeter",
    set_origin: Optional[tuple[float, float, float]] = None,
    set_corner: Optional[str] = None,
    set_corner_lps: Optional[tuple[float, float, float]] = None,
) -> sitk.Image:
    """
    Converts a ZARR file to a SimpleITK image.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    nd_metadata : dict
        Neural Dynamics metadata.
    level : int, optional
        Resolution level to read, by default 3.
    scale_unit : str, optional
        Unit for scaling, by default "millimeter".
    set_origin : tuple, optional
        Origin of the image, by default None. Exclusive of set_corner and
        set_corner_lps.
    set_corner : str, optional
        Which corner to use, by default None. If set, must specify both
        set_corner and set_corner_lps, exclusive of set_origin.
    set_corner_lps: tuple, optional
        Coordinates of the corner in LPS. If set, must specify both set_corner
        and set_corner_lps, exclusive of set_origin.


    Returns
    -------
    sitk.Image
        SimpleITK image object.
    """
    if set_origin is None:
        origin = (0.0, 0.0, 0.0)
    else:
        raise NotImplementedError("Setting origin is not implemented yet")
    (
        arr_data_spatial,
        dirs,
        spacing,
        size,
    ) = _zarr_to_numpy_anatomical(
        uri, nd_metadata, level=level, scale_unit=scale_unit
    )
    # SimpleITK uses fortran-style arrays, not C-style, so we need to reverse
    # the order of the axes
    dir_str = "".join(reversed(dirs))
    spacing_rev = spacing[::-1]
    size_rev = size[::-1]
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        dir_str
    )
    origin_type = _origin_args_check(set_origin, set_corner, set_corner_lps)
    if origin_type == "origin":
        assert set_origin is not None
        origin = set_origin
    elif origin_type == "corner":
        assert set_corner_lps is not None and set_corner is not None
        origin = compute_origin_for_corner(
            size_rev, spacing_rev, dir_tup, set_corner_lps, set_corner
        )[0]
    else:
        origin = (0.0, 0.0, 0.0)
    sitk_image = sitk.GetImageFromArray(arr_data_spatial)
    sitk_image.SetSpacing(tuple(spacing_rev))
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(dir_tup)
    return sitk_image


def _origin_args_check(
    set_origin: Optional[tuple[float, float, float]],
    set_corner: Optional[str],
    set_corner_lps: Optional[tuple[float, float, float]],
) -> str:
    have_origin = set_origin is not None
    have_corner = set_corner is not None
    have_corner_lps = set_corner_lps is not None
    if have_origin and (have_corner or have_corner_lps):
        raise ValueError("Cannot specify both origin and corner")
    if have_corner != have_corner_lps:
        raise ValueError("Both set_corner and set_corner_lps must be set")
    if have_origin:
        return "origin"
    if have_corner:
        return "corner"
    return "none"


def zarr_to_sitk_stub(
    uri: str,
    nd_metadata: dict,
    level: int = 0,
    scale_unit: str = "millimeter",
    set_origin: Optional[tuple[float, float, float]] = None,
    set_corner: Optional[str] = None,
    set_corner_lps: Optional[tuple[float, float, float]] = None,
    opened_zarr: Optional[tuple[Node, dict]] = None,
) -> tuple[sitk.Image, tuple[int, int, int]]:
    """
    Creates a stub SimpleITK image with the same metadata as the ZARR file.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    nd_metadata : dict
        Neural Dynamics metadata.
    level : int, optional
        Resolution level to read, by default 0.
    scale_unit : str, optional
        Unit for scaling, by default "millimeter".
    set_origin : tuple, optional
        Origin of the image, by default None. Exclusive of set_corner and
        set_corner_lps.
    set_corner : str, optional
        Which corner to use, by default None. If set, must specify both
        set_corner and set_corner_lps, exclusive of set_origin.
    set_corner_lps: tuple, optional
        Coordinates of the corner in LPS. If set, must specify both set_corner
        and set_corner_lps, exclusive of set_origin.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    sitk.Image
        SimpleITK stub image object.
    tuple
        The size of the image data in each dimension of the underlying array,
        in SimpleITK order (column-major).
    """
    (
        image_node,
        rej_axes,
        dirs,
        spacing,
        size,
    ) = _zarr_to_anatomical(
        uri,
        nd_metadata,
        level=level,
        scale_unit=scale_unit,
        opened_zarr=opened_zarr,
    )
    # SimpleITK uses fortran-style arrays, not C-style, so we need to reverse
    # the order of the axes
    image_dims = len(image_node.data[level].shape)
    n_spatial = image_dims - len(rej_axes)
    dir_str = "".join(reversed(dirs))
    spacing_rev = spacing[::-1]
    size_rev = size[::-1]
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        dir_str
    )
    origin_type = _origin_args_check(set_origin, set_corner, set_corner_lps)
    if origin_type == "origin":
        assert set_origin is not None
        origin = set_origin
    elif origin_type == "corner":
        assert set_corner_lps is not None and set_corner is not None
        origin = compute_origin_for_corner(
            size_rev, spacing_rev, dir_tup, set_corner_lps, set_corner
        )[0]
    else:
        origin = (0.0, 0.0, 0.0)
    stub_image = sitk.Image([1] * n_spatial, sitk.sitkUInt8)
    stub_image.SetSpacing(tuple(spacing_rev))
    stub_image.SetOrigin(origin)
    stub_image.SetDirection(dir_tup)
    si, sj, sk = size_rev
    return stub_image, (si, sj, sk)


def _code_signs(code: str) -> np.ndarray:
    """
    Map a 3-letter code to LPS sign flips.
    code uses {L,R} x {P,A} x {S,I}. (Case-insensitive)
      LPS -> (+,+,+)
      RAS -> (-,-,+)
      LPI -> (+,+,-)
      etc.
    """
    c = code.upper()
    if len(c) != 3 or c[0] not in "LR" or c[1] not in "PA" or c[2] not in "SI":
        raise ValueError("code must be 3 letters: {L|R}{P|A}{S|I}")
    sx = +1.0 if c[0] == "L" else -1.0  # x: L=+x, R=-x in LPS
    sy = +1.0 if c[1] == "P" else -1.0  # y: P=+y, A=-y
    sz = +1.0 if c[2] == "S" else -1.0  # z: S=+z, I=-z
    return np.array([sx, sy, sz], float)


def _corner_indices(size, outer=True):
    size = np.asarray(size, float)
    lo = -0.5 if outer else 0.0
    hi = (size - 0.5) if outer else (size - 1.0)
    return np.array(
        list(product([lo, hi[0]], [lo, hi[1]], [lo, hi[2]])), float
    )


def compute_origin_for_corner(
    size: Sequence[int],
    spacing: Sequence[float],
    direction: NDArray,
    target_point: Sequence[float],
    corner_code: str = "RAS",
    target_frame: str = "LPS",
    use_outer_box: bool = False,
) -> tuple[tuple[float, float, float], NDArray, int]:
    """
    Compute the image origin such that a specified corner of the image
    aligns with a given physical point in a specified coordinate frame.

    Parameters
    ----------
    size : Iterable of numbers
        The image size along each spatial axis (e.g., [nx, ny, nz]).
    spacing : Iterable of numbers
        The voxel spacing along each axis in millimeters (e.g., [sx, sy, sz]).
    direction : array-like of numbers
        3x3 direction cosine matrix (row-major) in ITK/LPS convention.
    target_point : array-like of numbers
        Physical coordinates (in mm) of the desired corner in the target frame.
    corner_code : str, optional
        3-letter code specifying which image corner to align (e.g., "LPI",
        "RAS").  Default is "LPI".
    target_frame : str, optional
        3-letter code specifying the coordinate frame of `target_point`.
        Defaults to `corner_code`.
    use_outer_box : bool, optional
        If True, use bounding box corners (-0.5, size-0.5); if False, use voxel
        centers (0, size-1).  Default is False.

    Returns
    -------
    origin_lps : tuple of float
        The computed image origin in LPS coordinates (mm).
    chosen_corner_index : ndarray
        The continuous index (ijk) of the chosen corner.
    corner_idx_number : int
        The index (0..7) of the chosen corner.

    Notes
    -----
    This function is useful for setting the image origin so that a particular
    image corner matches a desired physical location, taking into account
    direction cosines and coordinate conventions.
    """
    if target_frame is None:
        target_frame = corner_code

    # Normalize to 3D
    size_arr = np.array(list(size) + [1, 1, 1])[:3].astype(float)
    spacing_arr = np.array(list(spacing) + [1, 1, 1])[:3].astype(float)
    D = np.asarray(direction, float).reshape(3, 3)

    # All 8 corners in continuous index space and their LPS offsets from origin
    corners_idx = _corner_indices(size_arr, outer=use_outer_box)  # (8,3)
    offsets_lps = (corners_idx * spacing_arr) @ D.T  # (8,3)

    # Pick the corner that is "most" along the requested code axes
    s_align = _code_signs(corner_code)
    vals = offsets_lps * s_align  # convert to that code's axis sense
    # lexicographic argmax: prioritize x, then y, then z in that code
    idx = np.lexsort((vals[:, 2], vals[:, 1], vals[:, 0]))[-1]
    corner_offset_lps = offsets_lps[idx]

    # Convert target point to LPS and solve: target = origin + corner_offset
    s_target = _code_signs(target_frame)
    target_lps = np.asarray(target_point, float) * s_target
    origin_lps = target_lps - corner_offset_lps

    return tuple(origin_lps), corners_idx[idx], idx
