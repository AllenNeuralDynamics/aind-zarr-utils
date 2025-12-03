"""
Module for turning ZARRs into ants images and vice versa.
"""

from __future__ import annotations

import ants  # type: ignore[import-untyped]
import numpy as np
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import fix_corner_compute_origin
from ants.core import ANTsImage  # type: ignore[import-untyped]
from numpy.typing import NDArray
from ome_zarr.io import parse_url  # type: ignore[import-untyped]
from ome_zarr.reader import Node, Reader  # type: ignore[import-untyped]


def ensure_native_endian(
    a: np.ndarray, *, inplace: bool = False
) -> np.ndarray:
    """
    Return `a` with native-endian dtype.
    - View if already native or byteorder-agnostic.
    - Copy only when endianness must change or fields are mixed.
    - With `inplace=True`, mutate array when it's safe.
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
                raise ValueError(
                    "Array is not writeable; cannot convert in-place."
                )
            a.byteswap(inplace=True)
            a.dtype = dt.newbyteorder("=")  # type: ignore[misc]
            return a
        return a.astype(dt.newbyteorder("="), copy=False)

    # -------- Structured dtype --------
    # Collect byteorders of endian-aware fields ('<' or '>')
    field_orders = {
        subdt.byteorder
        for (subdt, *_) in dt.fields.values()
        if subdt.byteorder in ("<", ">")
    }

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
            raise ValueError(
                "Array is not writeable; cannot convert in-place."
            )
        a.byteswap(inplace=True)
        a.dtype = dt.newbyteorder("=")  # type: ignore[misc]
        return a

    return a.astype(dt.newbyteorder("="), copy=False)


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


def zarr_to_numpy(
    uri: str, level: int = 3, ensure_native_endianness: bool = False
) -> tuple[NDArray, dict, int]:
    """
    Converts a ZARR file to a NumPy array.

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
    Extracts scaled coordinate information from a ZARR file.

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
    opened_zarr: tuple[Node, dict] | None = None,
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
    image_node, rej_axes, zarr_axes, spacing, size = _zarr_to_scaled(
        uri, level=level, scale_unit=scale_unit, opened_zarr=opened_zarr
    )
    dirs = [metadata_axes_to_dir[a] for a in zarr_axes]
    return image_node, rej_axes, dirs, spacing, size


def _zarr_to_numpy_anatomical(
    uri: str,
    nd_metadata: dict,
    level: int = 3,
    scale_unit: str = "millimeter",
    opened_zarr: tuple[Node, dict] | None = None,
    ensure_native_endianness: bool = False,
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
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.
    ensure_native_endianness : bool, optional
        Whether to ensure native endianness of the returned array, by default
        False.

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
        uri,
        nd_metadata,
        level=level,
        scale_unit=scale_unit,
        opened_zarr=opened_zarr,
    )
    arr_data = image_node.data[level].compute()
    arr_data_spatial = np.squeeze(arr_data, axis=tuple(rej_axes))
    if ensure_native_endianness:
        arr_data_spatial = ensure_native_endian(arr_data_spatial, inplace=True)
    return arr_data_spatial, dirs, spacing, size


def _anatomical_to_ants(
    arr_data_spatial: NDArray,
    dirs: list[str],
    spacing: list[float],
    size: list[int],
    *,
    set_origin: tuple[float, float, float] | None = None,
    set_corner: str | None = None,
    set_corner_lps: tuple[float, float, float] | None = None,
) -> ANTsImage:
    """
    Converts anatomical data to an ANTs image.

    Parameters
    ----------
    arr_data_spatial : NDArray
        NumPy array of the image data with spatial dimensions.
    dirs : list
        List of direction codes.
    spacing : list
        List of spacing values.
    size : list
        List of size values.
    set_origin : tuple, optional
        Origin of the image, by default None. Exclusive of set_corner and
        set_corner_lps.
    set_corner : str, optional
        Which corner to use, by default None. If set, must specify both
        set_corner and set_corner_lps, exclusive of set_origin.
    set_corner_lps: tuple, optional
        Coordinates of the corner in LPS. If set, must specify both set_corner
        and set_corner_lps, exclusive of set_origin.
    """
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
        origin = fix_corner_compute_origin(
            size, spacing, dir_tup, set_corner_lps, set_corner
        )[0]
    elif origin_type == "none":
        origin = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown origin_type: {origin_type}")
    ants_image = ants.from_numpy(
        arr_data_spatial, spacing=spacing, direction=dir_mat, origin=origin
    )
    return ants_image


def zarr_to_ants(
    uri: str,
    nd_metadata: dict,
    level: int = 3,
    scale_unit: str = "millimeter",
    set_origin: tuple[float, float, float] | None = None,
    set_corner: str | None = None,
    set_corner_lps: tuple[float, float, float] | None = None,
    opened_zarr: tuple[Node, dict] | None = None,
) -> ANTsImage:
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
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    ants.core.ANTsImage
        ANTs image object.
    """
    (arr_data_spatial, dirs, spacing, size) = _zarr_to_numpy_anatomical(
        uri,
        nd_metadata,
        level=level,
        scale_unit=scale_unit,
        opened_zarr=opened_zarr,
        ensure_native_endianness=True,
    )

    return _anatomical_to_ants(
        arr_data_spatial,
        dirs,
        spacing,
        size,
        set_origin=set_origin,
        set_corner=set_corner,
        set_corner_lps=set_corner_lps,
    )


def _anatomical_to_sitk(
    arr_data_spatial: np.ndarray,
    dirs: list[str],
    spacing: list[float],
    size: list[int],
    set_origin: tuple[float, float, float] | None,
    set_corner: str | None,
    set_corner_lps: tuple[float, float, float] | None,
) -> sitk.Image:
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
        origin = fix_corner_compute_origin(
            size_rev, spacing_rev, dir_tup, set_corner_lps, set_corner
        )[0]
    elif origin_type == "none":
        origin = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown origin_type: {origin_type}")
    sitk_image = sitk.GetImageFromArray(arr_data_spatial)
    sitk_image.SetSpacing(tuple(spacing_rev))
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(dir_tup)
    return sitk_image


def zarr_to_sitk(
    uri: str,
    nd_metadata: dict,
    level: int = 3,
    scale_unit: str = "millimeter",
    set_origin: tuple[float, float, float] | None = None,
    set_corner: str | None = None,
    set_corner_lps: tuple[float, float, float] | None = None,
    opened_zarr: tuple[Node, dict] | None = None,
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
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.


    Returns
    -------
    sitk.Image
        SimpleITK image object.
    """
    (
        arr_data_spatial,
        dirs,
        spacing,
        size,
    ) = _zarr_to_numpy_anatomical(
        uri,
        nd_metadata,
        level=level,
        scale_unit=scale_unit,
        opened_zarr=opened_zarr,
        ensure_native_endianness=True,
    )
    return _anatomical_to_sitk(
        arr_data_spatial,
        dirs,
        spacing,
        size,
        set_origin=set_origin,
        set_corner=set_corner,
        set_corner_lps=set_corner_lps,
    )


def _origin_args_check(
    set_origin: tuple[float, float, float] | None,
    set_corner: str | None,
    set_corner_lps: tuple[float, float, float] | None,
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
    set_origin: tuple[float, float, float] | None = None,
    set_corner: str | None = None,
    set_corner_lps: tuple[float, float, float] | None = None,
    opened_zarr: tuple[Node, dict] | None = None,
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
        origin = fix_corner_compute_origin(
            size_rev, spacing_rev, dir_tup, set_corner_lps, set_corner
        )[0]
    elif origin_type == "none":
        origin = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown origin_type: {origin_type}")
    stub_image = sitk.Image([1] * n_spatial, sitk.sitkUInt8)
    stub_image.SetSpacing(tuple(spacing_rev))
    stub_image.SetOrigin(origin)
    stub_image.SetDirection(dir_tup)
    si, sj, sk = size_rev
    return stub_image, (si, sj, sk)
