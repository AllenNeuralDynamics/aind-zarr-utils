"""Read declared NGFF scale/translation geometry and bind explicit interpretations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from aind_metadata_reader import Document, Interpretation, Source
from aind_s3_cache.uri_utils import as_pathlike, as_string
from numpy.typing import NDArray
from ome_zarr.reader import Node  # type: ignore[import-untyped]


def _vector(value: Any, dimensions: int) -> NDArray:
    """Require finite numeric vectors before using them in physical geometry."""
    if not isinstance(value, list) or len(value) != dimensions or any(type(x) not in (int, float) for x in value):
        raise ValueError("NGFF transform must contain one numeric value per axis")
    vector = np.asarray(value, dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("NGFF transform contains nonfinite values")
    return vector


def _compose(transforms: list[dict], dimensions: int) -> tuple[NDArray, NDArray]:
    """Compose diagonal transforms in their declared order, including translated scales."""
    scale = np.ones(dimensions)
    translation = np.zeros(dimensions)
    for transform in transforms:
        kind = transform.get("type")
        # Historical flattened metadata accepted an untyped scale dictionary.
        if kind is None and set(transform) == {"scale"}:
            kind = "scale"
        if kind == "identity":
            continue
        if kind == "scale":
            factor = _vector(transform.get("scale"), dimensions)
            if (factor <= 0).any():
                raise ValueError("NGFF scales must be positive")
            scale *= factor
            translation *= factor
        elif kind == "translation":
            translation += _vector(transform.get("translation"), dimensions)
        else:
            raise ValueError(f"Unsupported NGFF coordinate transform {kind!r}")
    return scale, translation


def level_transform(metadata: dict, level: int) -> tuple[NDArray, NDArray]:
    """Compose dataset then image transforms; missing translation means zero, not block averaging.

    https://ngff.openmicroscopy.org/0.4/#multiscale-md specifies the transform order.
    This interprets geometry without asserting full-schema validity.
    """
    dimensions = len(metadata["axes"])
    multiscales = metadata.get("multiscales")
    if multiscales is not None:
        image = multiscales[0]
        dataset = image["datasets"][level]["coordinateTransformations"]
        transforms = [*dataset, *image.get("coordinateTransformations", [])]
    else:
        transforms = metadata["coordinateTransformations"][level]
    return _compose(transforms, dimensions)


def attributes_document(uri: str, opened_zarr: tuple[Node, dict]) -> Document:
    """Capture the opened NGFF attributes with their locator; parsed attributes are not original bytes."""
    node, metadata = opened_zarr
    attributes = getattr(getattr(node, "zarr", None), "root_attrs", None)
    if not isinstance(attributes, dict) or "multiscales" not in attributes:
        if "multiscales" not in metadata:
            raise ValueError("Opened Zarr does not expose the original multiscales attributes")
        attributes = {"multiscales": metadata["multiscales"]}
    kind, bucket, path = as_pathlike(uri)
    root = next((p for p in (path, *path.parents) if ".zarr" in p.suffixes), None)
    if root is None:
        raise ValueError("Cannot locate the Zarr root for attribute provenance")
    location = getattr(node, "zarr", None)
    storage_format = getattr(getattr(location, "fmt", None), "zarr_format", 2)
    filename = "zarr.json" if storage_format == 3 else ".zattrs"
    source = Source(kind="mapping", locator=as_string(kind, bucket, root / filename))
    return Document(attributes, provenance=source)


def interpreted_zarr(uri: str, opened_zarr: tuple[Node, dict], interpretation: Interpretation) -> tuple[Node, dict]:
    """Bind an interpretation to the same source capture, without changing pixels or stored attributes."""
    original = attributes_document(uri, opened_zarr)
    if original.snapshot.snapshot_id != interpretation.original.snapshot.snapshot_id:
        raise ValueError("Geometry interpretation belongs to a different Zarr attribute capture")
    node, metadata = opened_zarr
    attributes = interpretation.derived.raw
    if attributes == interpretation.original.raw:
        return opened_zarr
    images = attributes.get("multiscales")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Geometry interpretation must identify one multiscale image")
    result = deepcopy(metadata)
    result["multiscales"] = images
    result["axes"] = images[0]["axes"]
    result["coordinateTransformations"] = [d["coordinateTransformations"] for d in images[0]["datasets"]]
    return node, result
