"""
Module for turning ZARRs into ants images and vice versa.
"""

import ants
import numpy as np
import SimpleITK as sitk
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


# I think this assumes C-style array ordering
def direction_from_acquisition_metadata(metadata):
    axes_dict = {d["dimension"]: d for d in metadata["axes"]}
    dimensions = np.sort(np.array(list(axes_dict.keys())))
    axes = []
    directions = []
    for i in dimensions:
        axes.append(axes_dict[i]["name"].lower())
        directions.append(axes_dict[i]["direction"].split("_")[-1][0].upper())
    return dimensions, axes, directions


def units_to_meter(unit):
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


def unit_conversion(src, dst):
    if src == dst:
        return 1.0
    src_meters = units_to_meter(src)
    dst_meters = units_to_meter(dst)
    return src_meters / dst_meters


def zarr_to_numpy(uri, level=3):
    reader = Reader(parse_url(uri))

    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]
    zarr_meta = image_node.metadata
    arr_data = image_node.data[level].compute()
    return arr_data, zarr_meta, level


def _zarr_to_anatomical(uri, metadata, level=3, scale_unit="millimeter"):
    # Get direction metadata
    _, axes, directions = direction_from_acquisition_metadata(metadata)
    metadata_axes_to_dir = {a: d for a, d in zip(axes, directions)}
    # Create the zarr reader
    arr_data, zarr_meta, _ = zarr_to_numpy(uri, level)
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
    arr_data_spatial = np.squeeze(arr_data, axis=tuple(rej_axes))
    dirs = []
    spacing = []
    for i in original_to_subset_axes_map.keys():
        zarr_axis = zarr_axes[i]["name"]
        dirs.append(metadata_axes_to_dir[zarr_axis])
        scale_factor = unit_conversion(zarr_axes[i]["unit"], scale_unit)
        spacing.append(scale_factor * scale[i])
    return arr_data_spatial, dirs, spacing


def zarr_to_ants(
    uri, metadata, level=3, scale_unit="millimeter", set_origin=None
):
    if set_origin is None:
        origin = (0.0, 0.0, 0.0)
    else:
        raise NotImplementedError("Setting origin is not implemented yet")
    (
        arr_data_spatial,
        dirs,
        spacing,
    ) = _zarr_to_anatomical(uri, metadata, level=level, scale_unit=scale_unit)

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
    uri, metadata, level=3, scale_unit="millimeter", set_origin=None
):
    if set_origin is None:
        origin = (0.0, 0.0, 0.0)
    else:
        raise NotImplementedError("Setting origin is not implemented yet")
    (
        arr_data_spatial,
        dirs,
        spacing,
    ) = _zarr_to_anatomical(uri, metadata, level=level, scale_unit=scale_unit)
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
