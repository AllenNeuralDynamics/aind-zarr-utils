"""Module for turning ZARRs into ants images and vice versa.

The low-level Zarr opening and metadata helpers (``_open_zarr``,
``zarr_to_numpy``, ``_zarr_to_scaled``, ``ensure_native_endian``,
``direction_from_*``, ``_unit_conversion``, ``_units_to_meter``) live in
:mod:`aind_zarr_utils.io.zarr` and :mod:`aind_zarr_utils.io.metadata` and
are re-exported here for backwards compatibility. New code should import
them from their new homes.
"""

from __future__ import annotations

import ants  # type: ignore[import-untyped]
import numpy as np
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import fix_corner_compute_origin
from ants.core import ANTsImage  # type: ignore[import-untyped]
from numpy.typing import NDArray

# These imports are re-exported below so that test fixtures and downstream
# consumers can continue to monkey-patch them at ``aind_zarr_utils.zarr``.
from ome_zarr.io import parse_url as parse_url  # type: ignore[import-untyped]
from ome_zarr.reader import Node as Node  # type: ignore[import-untyped]
from ome_zarr.reader import Reader as Reader

from aind_zarr_utils.io.metadata import (
    _unit_conversion as _unit_conversion,
)
from aind_zarr_utils.io.metadata import (
    _units_to_meter as _units_to_meter,
)
from aind_zarr_utils.io.metadata import (
    direction_from_acquisition_metadata as direction_from_acquisition_metadata,
)
from aind_zarr_utils.io.metadata import (
    direction_from_nd_metadata as direction_from_nd_metadata,
)
from aind_zarr_utils.io.zarr import (
    _open_zarr as _open_zarr,
)
from aind_zarr_utils.io.zarr import (
    _zarr_to_scaled as _zarr_to_scaled,
)
from aind_zarr_utils.io.zarr import (
    ensure_native_endian as ensure_native_endian,
)
from aind_zarr_utils.io.zarr import (
    zarr_to_numpy as zarr_to_numpy,
)


def scaled_points_to_indices(
    scaled_points: dict[str, NDArray],
    zarr_uri: str,
    *,
    scale_unit: str = "millimeter",
    opened_zarr: tuple[Node, dict] | None = None,
) -> dict[str, NDArray]:
    """
    Convert scaled (non-anatomical) coordinates to zarr indices.

    Scaled coordinates are voxel indices multiplied by voxel spacing, without
    anatomical direction information from ND metadata. This function divides
    by spacing to recover continuous indices.

    Parameters
    ----------
    scaled_points : dict[str, NDArray]
        Mapping layer name → (N, 3) array of scaled coordinates in (z, y, x)
        order. Coordinates are in the units specified by `scale_unit`.
    zarr_uri : str
        URI of the Zarr file. Used to extract voxel spacing from metadata.
    scale_unit : str, optional
        Units of the scaled coordinates. Default is "millimeter".
    opened_zarr : tuple, optional
        Pre-opened Zarr (image_node, zarr_meta). If provided, avoids
        re-opening the Zarr file.

    Returns
    -------
    dict[str, NDArray]
        Mapping layer name → (N, 3) array of continuous (floating-point)
        indices in (z, y, x) order. These can be passed to
        `indices_to_ccf_auto_metadata()` or similar functions.

    See Also
    --------
    indices_to_ccf_auto_metadata : Transform indices to CCF coordinates.
    swc_data_to_zarr_indices : Similar function for SWC coordinates.
    neuroglancer_annotations_to_scaled : Extract scaled coords from
        Neuroglancer.

    Examples
    --------
    Convert scaled coordinates to indices, then to CCF:

    >>> scaled_pts = {"layer1": np.array([[1.0, 2.0, 3.0]])}  # in mm
    >>> indices = scaled_points_to_indices(scaled_pts, zarr_uri)
    >>> ccf_coords = indices_to_ccf_auto_metadata(indices, zarr_uri)

    Notes
    -----
    - Scaled coordinates are physical distances but lack anatomical
      orientation
    - The returned indices are continuous (float), not rounded to integers
    - Uses only Zarr scale metadata, not ND acquisition metadata
    """
    # Get spacing from Zarr metadata (no anatomical info needed)
    _, _, _, spacing_raw, _ = _zarr_to_scaled(zarr_uri, level=0, scale_unit=scale_unit, opened_zarr=opened_zarr)

    spacing = np.array(spacing_raw)
    indices = {}

    for layer, pts in scaled_points.items():
        pts_arr = np.asarray(pts)
        if pts_arr.ndim != 2 or pts_arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array for layer {layer}, got shape {pts_arr.shape}")
        # Convert scaled coords to indices: indices = coords / spacing
        # Keep as float (continuous indices), don't round
        indices[layer] = pts_arr / spacing

    return indices


def _zarr_to_anatomical(
    uri: str,
    nd_metadata: dict,
    *,
    level: int = 3,
    scale_unit: str = "millimeter",
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[Node, set[int], list[str], list[float], list[int]]:
    """
    Extract anatomical information from a ZARR file.

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
    Convert a ZARR file to a NumPy array with anatomical information.

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
    Convert anatomical data to an ANTs image.

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
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(dir_str)
    dir_mat = np.array(dir_tup).reshape((3, 3))
    origin_type = _origin_args_check(set_origin, set_corner, set_corner_lps)
    if origin_type == "origin":
        assert set_origin is not None
        origin = set_origin
    elif origin_type == "corner":
        assert set_corner_lps is not None and set_corner is not None
        origin = fix_corner_compute_origin(size, spacing, dir_tup, set_corner_lps, set_corner)[0]
    elif origin_type == "none":
        origin = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown origin_type: {origin_type}")
    ants_image = ants.from_numpy(arr_data_spatial, spacing=spacing, direction=dir_mat, origin=origin)
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
    Convert a ZARR file to an ANTs image.

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
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(dir_str)
    origin_type = _origin_args_check(set_origin, set_corner, set_corner_lps)
    if origin_type == "origin":
        assert set_origin is not None
        origin = set_origin
    elif origin_type == "corner":
        assert set_corner_lps is not None and set_corner is not None
        origin = fix_corner_compute_origin(size_rev, spacing_rev, dir_tup, set_corner_lps, set_corner)[0]
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
    Convert a ZARR file to a SimpleITK image.

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
    Create a stub SimpleITK image with the same metadata as the ZARR file.

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
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(dir_str)
    origin_type = _origin_args_check(set_origin, set_corner, set_corner_lps)
    if origin_type == "origin":
        assert set_origin is not None
        origin = set_origin
    elif origin_type == "corner":
        assert set_corner_lps is not None and set_corner is not None
        origin = fix_corner_compute_origin(size_rev, spacing_rev, dir_tup, set_corner_lps, set_corner)[0]
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
