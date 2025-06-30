"""
Module for turning ZARRs into ants images and vice versa.
"""

import ants
import numpy as np
import SimpleITK as sitk
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def direction_from_acquisition_metadata(acq_metadata):
    """
    Extracts direction, axes, and dimensions from acquisition metadata.

    Parameters
    ----------
    acq_metadata : dict
        Acquisition metadata

    Returns
    -------
    dimensions : ndarray
        Sorted array of dimension names.
    axes : list
        List of axis names in lowercase.
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


def direction_from_nd_metadata(nd_metadata):
    """
    Extracts direction, axes, and dimensions from ND metadata.

    Parameters
    ----------
    nd_metadata : dict
        ND metadata

    Returns
    -------
    dimensions : ndarray
        Sorted array of dimension names.
    axes : list
        List of axis names in lowercase.
    directions : list
        List of direction codes (e.g., 'L', 'R', etc.).
    """
    return direction_from_acquisition_metadata(nd_metadata["acquisition"])


def _units_to_meter(unit):
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


def _unit_conversion(src, dst):
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


def _open_zarr(uri):
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


def zarr_to_numpy(uri, level=3):
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


def _zarr_to_anatomical(uri, nd_metadata, level=3, scale_unit="millimeter"):
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
    """
    # Get direction metadata
    _, axes, directions = direction_from_nd_metadata(nd_metadata)
    metadata_axes_to_dir = {a: d for a, d in zip(axes, directions)}
    # Create the zarr reader
    image_node, zarr_meta = _open_zarr(uri)
    scale = np.array(zarr_meta["coordinateTransformations"][level][0]["scale"])
    zarr_axes = zarr_meta["axes"]
    spatial_dims = set(["x", "y", "z"])
    original_to_subset_axes_map = {}  # sorted
    i = 0
    for j, ax in enumerate(zarr_axes):
        ax_name = ax["name"]
        if ax_name in spatial_dims:
            original_to_subset_axes_map[j] = i
            i += 1
    rej_axes = set(range(len(zarr_axes))) - set(
        original_to_subset_axes_map.keys()
    )
    dirs = []
    spacing = []
    for i in original_to_subset_axes_map.keys():
        zarr_axis = zarr_axes[i]["name"]
        dirs.append(metadata_axes_to_dir[zarr_axis])
        scale_factor = _unit_conversion(zarr_axes[i]["unit"], scale_unit)
        spacing.append(scale_factor * scale[i])
    return image_node, rej_axes, dirs, spacing


def _zarr_to_numpy_anatomical(
    uri, nd_metadata, level=3, scale_unit="millimeter"
):
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
    """
    image_node, rej_axes, dirs, spacing = _zarr_to_anatomical(
        uri, nd_metadata, level=level, scale_unit=scale_unit
    )
    arr_data = image_node.data[level].compute()
    arr_data_spatial = np.squeeze(arr_data, axis=tuple(rej_axes))
    return arr_data_spatial, dirs, spacing


def zarr_to_ants(
    uri, nd_metadata, level=3, scale_unit="millimeter", set_origin=None
):
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
        Origin of the image, by default None.

    Returns
    -------
    ants.ANTsImage
        ANTs image object.
    """
    if set_origin is None:
        origin = (0.0, 0.0, 0.0)
    else:
        raise NotImplementedError("Setting origin is not implemented yet")
    (
        arr_data_spatial,
        dirs,
        spacing,
    ) = _zarr_to_numpy_anatomical(
        uri, nd_metadata, level=level, scale_unit=scale_unit
    )

    # Get direction metadata
    dir_str = "".join(dirs)
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        dir_str
    )
    dir_mat = np.array(dir_tup).reshape((3, 3))
    ants_image = ants.from_numpy(
        arr_data_spatial, spacing=spacing, direction=dir_mat, origin=origin
    )
    return ants_image


def zarr_to_sitk(
    uri, nd_metadata, level=3, scale_unit="millimeter", set_origin=None
):
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
        Origin of the image, by default None.

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
    ) = _zarr_to_numpy_anatomical(
        uri, nd_metadata, level=level, scale_unit=scale_unit
    )
    # SimpleITK uses fortran-style arrays, not C-style, so we need to reverse
    # the order of the axes
    dir_str = "".join(reversed(dirs))
    spacing_rev = spacing[::-1]
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        dir_str
    )
    sitk_image = sitk.GetImageFromArray(arr_data_spatial)
    sitk_image.SetSpacing(tuple(spacing_rev))
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(dir_tup)
    return sitk_image


def zarr_to_sitk_stub(
    uri, nd_metadata, level=0, scale_unit="millimeter", set_origin=None
):
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
        Origin of the image, by default None.

    Returns
    -------
    sitk.Image
        SimpleITK stub image object.
    """
    if set_origin is None:
        origin = (0.0, 0.0, 0.0)
    else:
        raise NotImplementedError("Setting origin is not implemented yet")
    (
        image_node,
        rej_axes,
        dirs,
        spacing,
    ) = _zarr_to_anatomical(
        uri, nd_metadata, level=level, scale_unit=scale_unit
    )
    # SimpleITK uses fortran-style arrays, not C-style, so we need to reverse
    # the order of the axes
    image_dims = len(image_node.data[level].shape)
    n_spatial = image_dims - len(rej_axes)
    dir_str = "".join(reversed(dirs))
    spacing_rev = spacing[::-1]
    dir_tup = sitk.DICOMOrientImageFilter.GetDirectionCosinesFromOrientation(
        dir_str
    )
    stub_image = sitk.Image([1] * n_spatial, sitk.sitkUInt8)
    stub_image.SetSpacing(tuple(spacing_rev))
    stub_image.SetOrigin(origin)
    stub_image.SetDirection(dir_tup)
    return stub_image
