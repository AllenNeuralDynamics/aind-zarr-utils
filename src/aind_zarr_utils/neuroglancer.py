"""
Module for reading Neuroglancer annotation layers.
"""

from __future__ import annotations

import re
from typing import TypeVar

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from aind_zarr_utils.annotations import annotation_indices_to_anatomical
from aind_zarr_utils.zarr import zarr_to_sitk_stub

T = TypeVar("T", int, float)


def neuroglancer_annotations_to_indices(
    data: dict,
    layer_names: str | list[str] | None = None,
    return_description: bool = True,
) -> tuple[dict[str, NDArray], dict[str, NDArray] | None]:
    """
    Reads annotation layers from a Neuroglancer JSON file and returns points in
    voxel indices.

    This function reads the annotation layers from a Neuroglancer JSON file and
    returns the points in voxel indices. Optionally, it returns the
    descriptions of the annotations if they exist.

    Notes
    -----
    The points in the Neuroglancer annotation layers are assumed to be in the
    order z, y, x, t. Only the indices for z, y, x are returned, in that order.

    Parameters
    ----------
    data: dict
        Parsed Neuroglancer JSON data.
    layer_names : str or list of str or None, optional
        Names of annotation layers to extract. If None, auto-detects all
        annotation layers. Default is None.
    return_description : bool, optional
        If True, returns annotation descriptions alongside points. Default is
        True.

    Returns
    -------
    annotations : dict
        Dictionary where keys are annotation layer names and values are (N, 3)
        numpy arrays of continuous (floating-point) voxel indices in (z, y, x)
        order. Values may represent sub-voxel positions.
    descriptions : dict or None
        Dictionary of annotation descriptions for each layer. Returned only if
        `return_description` is True, otherwise None.
    """

    layers = data.get("layers", [])
    layer_names = _resolve_layer_names(
        layers, layer_names, layer_type="annotation"
    )
    annotations, descriptions = _process_annotation_layers(
        layers,
        layer_names,
        return_description=return_description,
    )

    return annotations, descriptions


def neuroglancer_annotations_to_anatomical(
    neuroglancer_data: dict,
    zarr_uri: str,
    metadata: dict,
    layer_names: str | list[str] | None = None,
    return_description: bool = True,
    scale_unit: str = "millimeter",
    set_origin: tuple[T, T, T] | None = None,
    set_corner: str | None = None,
    set_corner_lps: tuple[float, float, float] | None = None,
    stub_image: sitk.Image | None = None,
) -> tuple[dict[str, NDArray], dict[str, NDArray] | None]:
    """
    Transforms Neuroglancer annotations to anatomical points.

    Notes
    -----
    This function assumes that none of the layers have had their transform
    altered. If any has, this will likely return incorrect values.

    The points in the Neuroglancer annotation layers are assumed to be in the
    order z, y, x, t. Only the indices for z, y, x are returned, in that order.

    Parameters
    ----------
    neuroglancer_data : dict
        Parsed Neuroglancer JSON data containing annotation layers.
    zarr_uri : str
        URI for the Zarr dataset.
    metadata : dict
        Neural Dynamics metadata for the Zarr dataset, which has acquisition
        metadata
    layer_names : str or list of str or None, optional
        Names of annotation layers to extract. If None, auto-detects all
        annotation layers. Default is None.
    return_description : bool, optional
        If True, returns annotation descriptions alongside points. Default is
        True.
    scale_unit : str, optional
        Unit to scale the anatomical points. Default is "millimeter".
    set_origin : tuple, optional
        Origin of the image, by default None. Exclusive of set_corner and
        set_corner_lps.
    set_corner : str, optional
        Which corner to use, by default None. If set, must specify both
        set_corner and set_corner_lps, exclusive of set_origin.
    set_corner_lps: tuple, optional
        Coordinates of the corner in LPS. If set, must specify both set_corner
        and set_corner_lps, exclusive of set_origin.
    stub_image : sitk.Image, optional
        Pre-created SimpleITK stub image to use for coordinate transformations.
        If provided, zarr_uri, metadata, scale_unit, set_origin, set_corner,
        and set_corner_lps parameters are ignored for stub creation.

    Returns
    -------
    anatomical_points : dict
        Dictionary where keys are annotation names and values are anatomical
        points. The points are in LPS (Left, Posterior, Superior) order,
        which is the standard for medical imaging.
    descriptions : dict
        Dictionary where keys are annotation names and values are lists of
        descriptions.
    """
    if stub_image is None:
        stub_img, _ = zarr_to_sitk_stub(
            zarr_uri,
            metadata,
            scale_unit=scale_unit,
            set_origin=set_origin,
            set_corner=set_corner,
            set_corner_lps=set_corner_lps,
        )
    else:
        stub_img = stub_image
    annotations, descriptions = neuroglancer_annotations_to_indices(
        neuroglancer_data,
        layer_names=layer_names,
        return_description=return_description,
    )
    annotation_points = annotation_indices_to_anatomical(
        stub_img,
        annotations,
    )
    return annotation_points, descriptions


def neuroglancer_annotations_to_scaled(
    data: dict,
    layer_names: str | list[str] | None = None,
    return_description: bool = True,
) -> tuple[dict[str, NDArray], list[str], dict[str, NDArray] | None]:
    """
    Reads annotation layers from a Neuroglancer JSON file and returns points in
    neuroglancer scaled coordinates.

    This function reads the annotation layers from a Neuroglancer JSON file and
    returns the points in scaled coordinates. The points are scaled by the
    voxel spacing found in the JSON file and are in units described by the
    `units` return value. Optionally, it returns the descriptions of the
    annotations if they exist.

    Notes
    -----
    This function assumes that none of the layers have had their transform
    altered. If any has, this will likely return incorrect values.

    The scaled coordinates are voxel indices multiplied by voxel spacing found
    in the JSON file, with units described by the `units` return value. They do
    NOT take into account how the brain was imaged, or what orientation the
    brain is in. For that, use `neuroglancer_annotations_to_anatomical`
    instead.

    The points in the Neuroglancer annotation layers are assumed to be in the
    order z, y, x, t. Only the spatial dimensions z, y, x are returned, in that
    order.

    Parameters
    ----------
    data : dict
        Parsed Neuroglancer JSON data.
    layer_names : str or list of str or None, optional
        Names of annotation layers to extract. If None, auto-detects all
        annotation layers. Default is None.
    return_description : bool, optional
        If True, returns annotation descriptions alongside points. Default is
        True.

    Returns
    -------
    annotations : dict
        Dictionary of annotation coordinates, scaled by the values in the
        dimension information, for each layer. The coordinates are in units
        described by
        the `units` return value.
    units : list of str
        Units of each dimension (e.g., ['m', 'm', 'm']).
    descriptions : dict or None
        Dictionary of annotation descriptions for each layer. Returned only if
        `return_description` is True, otherwise None. If `return_description`
        is True and there is no description for a point, its value will be
        None.
    """
    spacing, units = _extract_spacing(data["dimensions"])

    layers = data.get("layers", [])
    layer_names = _resolve_layer_names(
        layers, layer_names, layer_type="annotation"
    )
    annotations, descriptions = _process_annotation_layers(
        layers,
        layer_names,
        spacing=spacing,
        return_description=return_description,
    )

    return annotations, units, descriptions


def _extract_spacing(dimension_data: dict) -> tuple[NDArray, list[str]]:
    """
    Extracts voxel spacing from the Neuroglancer file.

    Parameters
    ----------
    dimension_data : dict
        Neuroglancer JSON dimension data.

    Returns
    -------
    spacing : numpy.ndarray
        Voxel spacing in each dimension.
    units : list of str
        Units of each dimension (e.g., ['m', 'm', 'm']).

    Raises
    ------
    ValueError
        If the required dimensions ('z', 'y', 'x') are not present in the file.
    """
    keep_order = ["z", "y", "x"]
    dimension_set = set(dimension_data.keys())
    missing = set(keep_order) - dimension_set
    if missing:
        raise ValueError(
            "Neuroglancer file must contain z, y, and x dimensions, "
            f"but missing: {missing}."
        )
    spacing = []
    units = []
    for dim in keep_order:
        space, unit = dimension_data[dim]
        spacing.append(space)
        units.append(unit)
    return np.array(spacing, dtype=float), units


def _resolve_layer_names(
    layers: list[dict],
    layer_names: str | list[str] | None,
    layer_type: str,
) -> list[str]:
    """
    Resolves layer names based on user input or auto-detects layers of the
    given type.

    Parameters
    ----------
    layers : list of dict
        Neuroglancer JSON layers.
    layer_names : str or list of str or None
        User-specified layer names or None to auto-detect.
    layer_type : str
        Type of layer to extract ('annotation' or 'probe').

    Returns
    -------
    list of str
        List of resolved layer names.

    Raises
    ------
    ValueError
        If the input `layer_names` is invalid.
    """
    if isinstance(layer_names, str):
        return [layer_names]
    if layer_names is None:
        return [
            layer["name"] for layer in layers if layer["type"] == layer_type
        ]
    if isinstance(layer_names, list):
        return layer_names
    raise ValueError(
        "Invalid input for layer_names. Expected a string, "
        "list of strings, or None."
    )


def _process_annotation_layers(
    layers: list[dict],
    layer_names: list[str],
    spacing: NDArray | None = None,
    return_description: bool = True,
) -> tuple[dict[str, NDArray], dict[str, NDArray] | None]:
    """
    Processes annotation layers to extract points and descriptions.

    Parameters
    ----------
    layers : list of dict
        Neuroglancer JSON layers.
    layer_names : list of str
        Names of annotation layers to extract.
    spacing : numpy.ndarray or None, optional
        Voxel spacing for scaling. If None, no scaling is done. Default is
        None.
    return_description : bool, optional
        Whether to extract descriptions alongside points. Default is True.

    Returns
    -------
    annotations : dict
        Annotation points for each layer.
    descriptions : dict or None
        Annotation descriptions for each layer, or None if not requested.
    """
    annotations = {}
    descriptions: dict[str, NDArray] | None = (
        {} if return_description else None
    )
    for layer_name in layer_names:
        layer = _get_layer_by_name(layers, layer_name)
        points, layer_descriptions = _process_layer_and_descriptions(
            layer,
            spacing=spacing,
            return_description=return_description,
        )
        annotations[layer_name] = points
        if descriptions is not None and layer_descriptions is not None:
            descriptions[layer_name] = layer_descriptions

    return annotations, descriptions


def _get_layer_by_name(layers: list[dict], name: str) -> dict:
    """
    Retrieves a layer by its name.

    Parameters
    ----------
    layers : list of dict
        Neuroglancer JSON layers.
    name : str
        Layer name to retrieve.

    Returns
    -------
    dict
        Layer data.

    Raises
    ------
    ValueError
        If the layer is not found.
    """
    for layer in layers:
        if layer["name"] == name:
            return layer
    raise ValueError(f'Layer "{name}" not found in the Neuroglancer file.')


def _process_layer_and_descriptions(
    layer: dict,
    spacing: NDArray | None = None,
    return_description: bool = True,
) -> tuple[NDArray, NDArray | None]:
    """
    Processes layer points and descriptions.

    Parameters
    ----------
    layer : dict
        Layer data.
    spacing : numpy.ndarray or None, optional
        Voxel spacing for scaling. If None, no scaling is done. Default is
        None.
    return_description : bool, optional
        Whether to extract descriptions. Default is True.

    Returns
    -------
    points : numpy.ndarray
        Scaled and reordered points.
    descriptions : numpy.ndarray or None
        Descriptions, or None if not requested.

    Raises
    ------
    ValueError
        If the annotation points do not have 4 dimensions (z, y, x, t).
    """
    points = []
    annotations = layer.get("annotations", [])
    for annotation in annotations:
        point_arr = np.array(annotation.get("point", []), dtype=float)
        if point_arr.shape[0] != 4:
            raise ValueError(
                "Annotation points expected to have 4 dimensions "
                f"(z, y, x, t), but {point_arr.shape[0]} found."
            )
        points.append(point_arr[:3])  # Keep only the first three dimensions
    points_arr = np.stack(points) if points else np.empty((0, 3), dtype=float)
    if spacing is not None:
        points_arr = points_arr * spacing

    if return_description:
        descriptions = [
            annotation.get("description", None) for annotation in annotations
        ]
        return points_arr, np.array(descriptions, dtype=object)
    return points_arr, None


def _sanitize_source_url(source: str) -> str:
    """
    Sanitizes a Neuroglancer source URL by removing 'zarr://' or 'zarr:/'.

    Parameters
    ----------
    source : str
        Original source URL.

    Returns
    -------
    str
        Sanitized source URL.
    """
    source = re.sub(r"^zarr:/+", "", source)
    source = re.sub(r"\|zarr2:$", "", source)
    return source


def get_image_sources(
    data: dict, remove_zarr_protocol: bool = False
) -> dict[str, str | None]:
    """
    Reads image source URL(s) from a Neuroglancer JSON file.

    Parameters
    ----------
    data: dict
        Parsed Neuroglancer JSON data.
    remove_zarr_protocol : bool, optional
        If True, removes 'zarr://' or 'zarr:/' prefixes from the source URLs.
        Default is False.

    Returns
    -------
    image_sources : dict
        Dictionary mapping image layer names to their source URLs.
    """
    image_sources = {}
    for layer in data.get("layers", []):
        if layer.get("type") == "image" and "name" in layer:
            this_source = layer.get("source", None)
            source_str = None
            if this_source is not None:
                if isinstance(this_source, str):
                    source_str = this_source
                elif isinstance(this_source, dict):
                    source_str = this_source.get("url")
            if remove_zarr_protocol and source_str is not None:
                source_str = _sanitize_source_url(source_str)
            image_sources[layer["name"]] = source_str
    return image_sources
