"""URI/path helpers for AIND SmartSPIM asset hierarchies.

The functions here understand the directory layout of a SmartSPIM dataset on
S3 (or any other URI scheme supported by ``aind_s3_cache.uri_utils``) and
provide a small toolkit for walking between the *acquisition Zarr*, the
*asset root*, and the *alignment channel Zarr*. They are intentionally free
of any pixel- or numpy-level concerns.
"""

from __future__ import annotations

from pathlib import PurePath, PurePosixPath
from typing import Any

from aind_s3_cache.json_utils import get_json
from aind_s3_cache.uri_utils import as_pathlike, as_string


def _asset_from_zarr_pathlike(zarr_path: PurePath) -> PurePath:
    """
    Return the asset (dataset) root directory for a given Zarr path.

    Parameters
    ----------
    zarr_path : Path
        A concrete filesystem path pointing somewhere inside a ``*.zarr``
        (or ``*.ome.zarr``) hierarchy.

    Returns
    -------
    Path
        The directory two levels above the provided Zarr path. For AIND
        SmartSPIM assets this corresponds to the asset root that contains
        processing outputs.
    """
    return zarr_path.parents[2]


def _asset_from_zarr_any(zarr_uri: str) -> str:
    """
    Return the asset root URI (string form) for an arbitrary Zarr URI.

    Parameters
    ----------
    zarr_uri : str
        URI or path-like string to a location inside a Zarr store.

    Returns
    -------
    str
        Asset root expressed in the same URI style as the input.
    """
    kind, bucket, p = as_pathlike(zarr_uri)
    return as_string(kind, bucket, _asset_from_zarr_pathlike(p))


def _zarr_base_name_pathlike(p: PurePath) -> str | None:
    """
    Infer the logical base name for a Zarr / OME-Zarr hierarchy.

    The base name is the directory name with all ``.ome`` / ``.zarr``
    suffixes removed. If no ancestor contains ``".zarr"`` in its suffixes,
    ``None`` is returned.

    Parameters
    ----------
    p : PurePath
        Path located at or within a Zarr hierarchy.

    Returns
    -------
    str or None
        Base stem without zarr/ome extensions, or ``None`` if not found.
    """
    # Walk up until we find a *.zarr (or *.ome.zarr) segment.
    z = next((a for a in (p, *p.parents) if ".zarr" in a.suffixes), None)
    if not z:
        return None

    # Strip all suffixes on that segment.
    q = z
    for _ in z.suffixes:
        q = q.with_suffix("")
    return q.name


def _zarr_base_name_any(base: str) -> str | None:
    """
    Wrap :func:`_zarr_base_name_pathlike` for any URI style.

    Parameters
    ----------
    base : str
        URI or path pointing at / inside a Zarr hierarchy.

    Returns
    -------
    str or None
        Base name without suffixes, or ``None`` if not detected.
    """
    _, _, p = as_pathlike(base)
    return _zarr_base_name_pathlike(p)


def alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
    asset_uri: str | None = None,
    a_zarr_uri: str | None = None,
    **kwargs: Any,
) -> tuple[str, dict, dict]:
    """
    Return the alignment uris for a given Zarr path.

    Parameters
    ----------
    asset_uri : str, optional
        Base URI for the asset containing the Zarr and metadata files. If
        ``None``, the asset root is inferred from ``a_zarr_uri``.
    a_zarr_uri : str, optional
        URI of an acquisition Zarr within the asset. If ``None``, the asset
        root is taken from ``asset_uri``.
    **kwargs : Any
        Forwarded keyword arguments accepted by :func:`get_json`. Common keys
        include:
        - ``s3_client`` : S3Client | None
        - ``anonymous`` : bool

    Returns
    -------
    tuple
        ``(zarr_uri, metadata, processing_data)`` where ``zarr_uri`` is the
        inferred alignment Zarr URI, ``metadata`` is the parsed
        ``metadata.nd.json`` content, and ``processing_data`` is the parsed
        ``processing.json`` content.
    """
    # Local import to avoid an ``io.paths`` ↔ ``io.processing`` cycle: the
    # processing helpers themselves depend on ``_zarr_base_name_pathlike``
    # defined above.
    from aind_zarr_utils.io.processing import (
        image_atlas_alignment_path_relative_from_processing,
    )

    if asset_uri is None:
        if a_zarr_uri is None:
            raise ValueError("Must provide either a_zarr_uri or asset_uri")
        uri_type, bucket, a_zarr_pathlike = as_pathlike(a_zarr_uri)
        asset_pathlike = _asset_from_zarr_pathlike(a_zarr_pathlike)
    else:
        uri_type, bucket, asset_pathlike = as_pathlike(asset_uri)
    metadata_pathlike = asset_pathlike / "metadata.nd.json"
    processing_pathlike = asset_pathlike / "processing.json"
    metadata_uri = as_string(uri_type, bucket, metadata_pathlike)
    processing_uri = as_string(uri_type, bucket, processing_pathlike)
    metadata = get_json(metadata_uri, **kwargs)
    processing_data = get_json(processing_uri, **kwargs)
    alignment_rel_path = image_atlas_alignment_path_relative_from_processing(processing_data)
    if alignment_rel_path is None:
        raise ValueError("Could not determine image atlas alignment path from processing data")
    channel = PurePosixPath(alignment_rel_path).stem
    zarr_pathlike = asset_pathlike / f"image_tile_fusing/OMEZarr/{channel}.zarr"
    zarr_uri = as_string(uri_type, bucket, zarr_pathlike)
    return zarr_uri, metadata, processing_data
