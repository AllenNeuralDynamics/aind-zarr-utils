"""
Module for reading Neuroglancer annotation layers.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from aind_zarr_utils.annotations import annotation_indices_to_anatomical
from aind_zarr_utils.zarr import zarr_to_sitk_stub


def neuroglancer_annotations_to_indices(
    data: dict,
    layer_names: Optional[Union[str, list[str]]] = None,
    return_description: bool = True,
) -> tuple[dict[str, NDArray], Optional[dict[str, list[str]]]]:
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
        Dictionary of annotation coordinates for each layer.
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
    layer_names: Optional[Union[str, list[str]]] = None,
    return_description: bool = True,
    scale_unit: str = "millimeter",
    set_origin: Optional[tuple] = None,
) -> tuple[dict[str, NDArray], Optional[dict[str, list[str]]]]:
    """
    Transforms Neuroglancer annotations to physical points in the image space.

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
        Unit to scale the physical points. Default is "millimeter".
    set_origin : tuple or None, optional
        If provided, sets the origin of the image to this value. Default is
        None.

    Returns
    -------
    physical_points : dict
        Dictionary where keys are annotation names and values are physical
        points. The points are in LPS (Left, Posterior, Superior) order,
        which is the standard for medical imaging.
    descriptions : dict
        Dictionary where keys are annotation names and values are lists of
        descriptions.
    """
    stub_img = zarr_to_sitk_stub(
        zarr_uri, metadata, scale_unit=scale_unit, set_origin=set_origin
    )
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


def neuroglancer_annotations_to_global(
    data: dict,
    layer_names: Optional[Union[str, list[str]]] = None,
    return_description: bool = True,
) -> tuple[dict[str, NDArray], list[str], Optional[dict[str, list[str]]]]:
    """
    Reads annotation layers from a Neuroglancer JSON file and returns points in
    neuroglancer global coordinates.

    This function reads the annotation layers from a Neuroglancer JSON file and
    returns the points in physical coordinates. The points are scaled by the
    voxel spacing found in the JSON file and are in units described by the
    `units` return value. Optionally, it returns the descriptions of the
    annotations if they exist.

    Notes
    -----
    This function assumes that none of the layers have had their transform
    altered. If any has, this will likely return incorrect values.

    The physical coordinates are scaled by the voxel spacing found in the
    JSON file and are in units described by the `units` return value. They do
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


def _resolve_layer_names(layers, layer_names, layer_type):
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
    layers,
    layer_names,
    spacing=None,
    return_description=True,
):
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
    descriptions = {} if return_description else None
    for layer_name in layer_names:
        layer = _get_layer_by_name(layers, layer_name)
        points, layer_descriptions = _process_layer_and_descriptions(
            layer,
            spacing=spacing,
            return_description=return_description,
        )
        annotations[layer_name] = points
        if return_description:
            descriptions[layer_name] = layer_descriptions

    return annotations, descriptions


def _get_layer_by_name(layers, name):
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
    layer,
    spacing=None,
    return_description=True,
):
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
    points = np.stack(points) if points else np.empty((0, 3), dtype=float)
    if spacing is not None:
        points = points * spacing

    if return_description:
        descriptions = [
            annotation.get("description", None) for annotation in annotations
        ]
        return points, np.array(descriptions, dtype=object)
    return points, None


def get_image_sources(data):
    """
    Reads image source URL(s) from a Neuroglancer JSON file.

    Parameters
    ----------
    data: dict
        Parsed Neuroglancer JSON data.

    Returns
    -------
    image_sources : dict
        Dictionary mapping image layer names to their source URLs.
    """
    image_sources = {}
    for layer in data.get("layers", []):
        if layer.get("type") == "image" and "name" in layer:
            image_sources[layer["name"]] = layer.get("source", None)
    return image_sources
