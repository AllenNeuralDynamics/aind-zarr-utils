"""Reading Neuroglancer annotation state.

These helpers parse the JSON produced by Neuroglancer's *Save state* and
return its annotation layers as ``(N, 3)`` numpy arrays in the package's
canonical ``(z, y, x)`` order, optionally together with the per-point
descriptions.

A layer's points are stored in the order given by its **own**
``source.transform.outputDimensions``, which is captured when the layer is
created and frozen thereafter. Neuroglancer matches layer space to global
space by dimension *name*, so one state can legitimately hold layers with
different orders, and reordering the global ``dimensions`` block does not
touch any stored point. Every function here therefore permutes by name using
the layer's own space, falling back to the global block only for a layer that
declares no transform. Two flavours are supported:

* :func:`neuroglancer_annotations_to_indices` — raw voxel indices.
* :func:`neuroglancer_annotations_to_scaled` — voxel indices multiplied
  by the per-axis spacing recorded in the Neuroglancer state.
* :func:`neuroglancer_annotations_to_anatomical` — the indices, lifted
  into LPS millimeters via a Zarr-derived stub image.

The image-source URL extractor :func:`get_image_sources` is also here.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from aind_zarr_utils.annotations import annotation_indices_to_anatomical
from aind_zarr_utils.zarr import zarr_to_sitk_stub

T = TypeVar("T", int, float)

#: Order this package returns spatial points in, regardless of a layer's own.
CANONICAL_SPATIAL_ORDER = ("z", "y", "x")


@dataclass(frozen=True)
class NamedPoints:
    """Points bound to the coordinate space that gives them meaning.

    ``points[:, i]`` is the coordinate along ``names[i]``. An array on its own
    is ambiguous -- layers in one state may declare different orders -- so the
    two always travel together.
    """

    names: tuple[str, ...]
    scales: NDArray
    units: tuple[str, ...]
    points: NDArray
    descriptions: NDArray | None = None

    def to_canonical(self) -> NDArray:
        """Return the points permuted into ``CANONICAL_SPATIAL_ORDER``."""
        return self.points[:, list(_canonical_perm(self.names))]

    def canonical_scales(self) -> NDArray:
        """Return the scales permuted into ``CANONICAL_SPATIAL_ORDER``."""
        return np.asarray(self.scales)[list(_canonical_perm(self.names))]

    def canonical_units(self) -> tuple[str, ...]:
        """Return the units permuted into ``CANONICAL_SPATIAL_ORDER``."""
        return tuple(self.units[i] for i in _canonical_perm(self.names))


def _canonical_perm(names: tuple[str, ...]) -> tuple[int, ...]:
    """Return the indices selecting z, y, x -- in that order -- out of *names*."""
    missing = set(CANONICAL_SPATIAL_ORDER) - set(names)
    if missing:
        raise ValueError(
            f"Neuroglancer coordinate space must contain z, y, and x dimensions, "
            f"but missing: {sorted(missing)}. Found: {list(names)}."
        )
    return tuple(names.index(n) for n in CANONICAL_SPATIAL_ORDER)


@dataclass(frozen=True)
class _Space:
    """A Neuroglancer coordinate space: ordered names with scales and units."""

    names: tuple[str, ...]
    scales: NDArray
    units: tuple[str, ...]

    @property
    def rank(self) -> int:
        """Number of dimensions in the space."""
        return len(self.names)


def _coordinate_space(dimension_data: dict) -> _Space | None:
    """Parse a Neuroglancer dimensions mapping, preserving its declared order.

    Returns None for anything that is not a plain ``name -> (scale, unit)``
    mapping -- notably coordinate-array dimensions, which carry no scale.
    Units are kept verbatim rather than normalised to SI; callers compare them
    for equality, so converting would only invent agreement that is not there.
    """
    if not isinstance(dimension_data, dict) or not dimension_data:
        return None
    names, scales, units = [], [], []
    for name, value in dimension_data.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        scale, unit = value
        if not isinstance(scale, (int, float)) or not isinstance(unit, str):
            return None
        names.append(name)
        scales.append(float(scale))
        units.append(unit)
    return _Space(tuple(names), np.asarray(scales, dtype=float), tuple(units))


def _global_space(data: dict) -> _Space | None:
    """Return the state's global coordinate space, if it declares a usable one."""
    return _coordinate_space(data.get("dimensions", {}))


def _layer_output_space(layer: dict) -> _Space | None:
    """Return the space a layer's own points live in.

    Reads the layer's ``source.transform.outputDimensions``. None means the
    layer declares none, and the caller keeps the legacy positional z, y, x
    reading -- every layer of a real saved state declares one, so this is the
    hand-built-state path. :func:`_warn_on_assumed_order` covers the case where
    that assumption is visibly wrong.

    Raises
    ------
    ValueError
        If the transform carries a ``matrix`` (an affine this reader does not
        apply), or the layer's sources disagree about their output space.
    """
    sources = layer.get("source")
    if sources is None:
        return None
    if not isinstance(sources, list):
        sources = [sources]

    spaces = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        transform = source.get("transform")
        if not isinstance(transform, dict):
            continue
        if transform.get("matrix") is not None:
            raise ValueError(
                f"Annotation layer {layer.get('name')!r} carries a transform matrix. "
                "This reader only permutes and scales by named dimensions, so "
                "applying it would silently drop the affine."
            )
        output_dimensions = transform.get("outputDimensions")
        if not output_dimensions:
            continue
        parsed = _coordinate_space(output_dimensions)
        if parsed is None:
            raise ValueError(
                f"Annotation layer {layer.get('name')!r} has an outputDimensions "
                "this reader cannot interpret as name -> (scale, unit): "
                f"{output_dimensions!r}."
            )
        spaces.append(parsed)

    if not spaces:
        return None
    first = spaces[0]
    for other in spaces[1:]:
        if other.names != first.names or not np.allclose(other.scales, first.scales):
            raise ValueError(
                f"Annotation layer {layer.get('name')!r} has sources with differing output spaces; cannot choose one."
            )
    return first


def annotation_layer_spaces(data: dict) -> dict[str, tuple[str, ...]]:
    """Map each annotation layer to the dimension order it declares.

    More than one distinct order in a state is legal but rare, and means any
    reader that assumes a fixed order is wrong for some of the layers.
    """
    spaces = {}
    for layer in data.get("layers", []):
        if layer.get("type") != "annotation":
            continue
        space = _layer_output_space(layer)
        spaces[layer["name"]] = tuple(space.names) if space is not None else ()
    return spaces


def neuroglancer_annotations_to_named(
    data: dict,
    layer_names: str | list[str] | None = None,
    return_description: bool = True,
) -> dict[str, NamedPoints]:
    """Read annotation layers without imposing an order on them.

    Each layer keeps its own declared dimension order and scales; use
    :meth:`NamedPoints.to_canonical` to permute. This is the honest primitive
    -- the other readers are this plus a permutation.

    Parameters
    ----------
    data : dict
        Parsed Neuroglancer JSON data.
    layer_names : str or list of str or None, optional
        Names of annotation layers to extract. If None, auto-detects all
        annotation layers. Default is None.
    return_description : bool, optional
        If True, attaches annotation descriptions. Default is True.

    Returns
    -------
    dict
        Layer name to :class:`NamedPoints`.
    """
    layers = data.get("layers", [])
    selected = _resolve_layer_names(layers, layer_names, layer_type="annotation")

    result = {}
    for layer_name in selected:
        layer = _get_layer_by_name(layers, layer_name)
        space = _layer_output_space(layer)
        if space is None:
            raise ValueError(
                f"Annotation layer {layer_name!r} declares no coordinate space; "
                "its points cannot be labelled without assuming an order."
            )
        points, descriptions = _read_layer_points(
            layer,
            rank=space.rank,
            return_description=return_description,
        )
        result[layer_name] = NamedPoints(
            names=tuple(space.names),
            scales=np.asarray(space.scales),
            units=tuple(space.units),
            points=points,
            descriptions=descriptions,
        )
    return result


def _warn_on_assumed_order(layer_name: str, global_space: _Space | None) -> None:
    """Warn when a transform-less layer is read as z, y, x against a state that isn't.

    The silent-and-wrong combination: nothing on the layer says which order its
    points are in, and the only order the state does declare is not the one we
    are about to assume.
    """
    if global_space is None:
        return
    spatial = tuple(n for n in global_space.names if n in CANONICAL_SPATIAL_ORDER)
    if spatial and spatial != CANONICAL_SPATIAL_ORDER:
        warnings.warn(
            f"Annotation layer {layer_name!r} declares no coordinate space, so its "
            f"points are read as {list(CANONICAL_SPATIAL_ORDER)}, but the state's "
            f"global dimensions are {list(global_space.names)}. If the points follow "
            "the global order, they are being mislabelled.",
            stacklevel=4,
        )


def _warn_on_mixed_spaces(spaces: dict[str, tuple[str, ...]]) -> None:
    """Warn when one state mixes dimension orders across its layers."""
    distinct = {names for names in spaces.values() if names}
    if len(distinct) > 1:
        listed = ", ".join(f"{name}={list(names)}" for name, names in sorted(spaces.items()))
        warnings.warn(
            "Neuroglancer state mixes annotation dimension orders: "
            f"{sorted(list(d) for d in distinct)}. Points are permuted per layer, "
            f"but a reader that assumes one order is wrong for some. Layers: {listed}",
            stacklevel=3,
        )


def neuroglancer_annotations_to_indices(
    data: dict,
    layer_names: str | list[str] | None = None,
    return_description: bool = True,
) -> tuple[dict[str, NDArray], dict[str, NDArray] | None]:
    """Read annotation layers from a Neuroglancer JSON file as voxel indices.

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
    layer_names = _resolve_layer_names(layers, layer_names, layer_type="annotation")
    annotations, descriptions = _process_annotation_layers(
        layers,
        layer_names,
        return_description=return_description,
        global_space=_global_space(data),
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
    """Transform Neuroglancer annotations into anatomical points.

    Notes
    -----
    Points are permuted into z, y, x by name using each layer's own
    ``outputDimensions``; see the module docstring.

    They are then treated as voxel indices in the stub's grid, which holds
    while a layer's scale equals the volume's voxel size -- true for every
    annotation produced against a multiscale root, since Neuroglancer keeps
    coordinates in base-resolution units. A layer sourced from one pyramid
    level instead would need rescaling by ``layerScale / stubSpacing``, which
    is what Neuroglancer itself applies (``baseInputScale / baseOutputScale``,
    ``render_coordinate_transform.ts``). Nothing here checks that, because a
    SimpleITK image carries no unit to check against.

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
    """Read annotation layers from a Neuroglancer JSON file as scaled coordinates.

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
    layer_names = _resolve_layer_names(layers, layer_names, layer_type="annotation")
    global_space = _global_space(data)
    _check_units_agree(layers, layer_names, global_space, units)
    annotations, descriptions = _process_annotation_layers(
        layers,
        layer_names,
        spacing=spacing,
        return_description=return_description,
        global_space=global_space,
    )

    return annotations, units, descriptions


def _check_units_agree(
    layers: list[dict],
    layer_names: list[str],
    global_space: _Space | None,
    units: list[str],
) -> None:
    """Refuse to report one unit list when the layers disagree with it.

    The scaled reader returns a single ``units`` for every layer, which is only
    honest while they all match.
    """
    for layer_name in layer_names:
        space = _layer_output_space(_get_layer_by_name(layers, layer_name))
        if space is None:
            continue
        perm = _canonical_perm(tuple(space.names))
        layer_units = [space.units[i] for i in perm]
        if layer_units != list(units):
            raise ValueError(
                f"Annotation layer {layer_name!r} is in {layer_units}, but the state's "
                f"global dimensions are {list(units)}. Use "
                "neuroglancer_annotations_to_named to keep each layer's own units."
            )


def _extract_spacing(dimension_data: dict) -> tuple[NDArray, list[str]]:
    """Extract voxel spacing from the Neuroglancer file.

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
        raise ValueError(f"Neuroglancer file must contain z, y, and x dimensions, but missing: {missing}.")
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
    """Resolve layer names from user input or auto-detect layers of the given type.

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
        return [layer["name"] for layer in layers if layer["type"] == layer_type]
    if isinstance(layer_names, list):
        return layer_names
    raise ValueError("Invalid input for layer_names. Expected a string, list of strings, or None.")


def _process_annotation_layers(
    layers: list[dict],
    layer_names: list[str],
    spacing: NDArray | None = None,
    return_description: bool = True,
    global_space: _Space | None = None,
) -> tuple[dict[str, NDArray], dict[str, NDArray] | None]:
    """Process annotation layers to extract points and descriptions.

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
    descriptions: dict[str, NDArray] | None = {} if return_description else None
    seen_spaces: dict[str, tuple[str, ...]] = {}
    for layer_name in layer_names:
        layer = _get_layer_by_name(layers, layer_name)
        space = _layer_output_space(layer)
        if space is None:
            _warn_on_assumed_order(layer_name, global_space)
        seen_spaces[layer_name] = tuple(space.names) if space is not None else ()
        # Scale by the layer's own values, in canonical order, not the global
        # block's -- a pasted-in layer can carry another volume's spacing.
        layer_spacing = spacing
        if spacing is not None and space is not None:
            layer_spacing = np.asarray(space.scales)[list(_canonical_perm(tuple(space.names)))]
        points, layer_descriptions = _process_layer_and_descriptions(
            layer,
            spacing=layer_spacing,
            return_description=return_description,
            space=space,
        )
        annotations[layer_name] = points
        if descriptions is not None and layer_descriptions is not None:
            descriptions[layer_name] = layer_descriptions

    _warn_on_mixed_spaces(seen_spaces)
    return annotations, descriptions


def _get_layer_by_name(layers: list[dict], name: str) -> dict:
    """Retrieve a layer by its name.

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
    space: _Space | None = None,
) -> tuple[NDArray, NDArray | None]:
    """Process layer points and descriptions.

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

    space : neuroglancer CoordinateSpace, optional
        The layer's own space; points are permuted from its order into
        ``CANONICAL_SPATIAL_ORDER``. If None, the points are assumed to already
        be in z, y, x, t order.

    Raises
    ------
    ValueError
        If a point's length does not match the declared dimension count.
    """
    rank = 4 if space is None else space.rank
    perm = (0, 1, 2) if space is None else _canonical_perm(tuple(space.names))
    points_arr, descriptions = _read_layer_points(
        layer,
        rank=rank,
        return_description=return_description,
    )
    points_arr = points_arr[:, list(perm)]
    if spacing is not None:
        points_arr = points_arr * spacing
    return points_arr, descriptions


def _read_layer_points(
    layer: dict,
    rank: int,
    return_description: bool = True,
) -> tuple[NDArray, NDArray | None]:
    """Read a layer's raw points, in the layer's own dimension order.

    *rank* is the layer's declared dimension count; a point of any other length
    means the state disagrees with itself, which is worth stopping for.
    """
    points = []
    annotations = layer.get("annotations", [])
    for annotation in annotations:
        point_arr = np.array(annotation.get("point", []), dtype=float)
        if point_arr.shape[0] != rank:
            raise ValueError(f"Annotation points expected to have {rank} dimensions, but {point_arr.shape[0]} found.")
        points.append(point_arr)
    points_arr = np.stack(points) if points else np.empty((0, rank), dtype=float)

    if return_description:
        descriptions = [annotation.get("description", None) for annotation in annotations]
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


def get_image_sources(data: dict, remove_zarr_protocol: bool = False) -> dict[str, str | None]:
    """Read image source URL(s) from a Neuroglancer JSON file.

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
