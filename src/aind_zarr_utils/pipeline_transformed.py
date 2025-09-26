"""
Utilities to reconstruct a pipeline's spatial domain for LS → CCF mappings
and to apply ANTs transform chains to points/annotations.

The goal is to produce a SimpleITK *stub* image (no pixels) whose header
(origin, spacing, direction) matches what the SmartSPIM processing pipeline
would have produced for a given acquisition. This lets you convert Zarr
voxel indices to the *same* anatomical coordinates that the transforms were
trained in, and then compose the appropriate ANTs transforms to reach CCF.

Notes
-----
- All world coordinates are **ITK LPS** and **millimeters**.
- SimpleITK direction matrices are 3×3 row-major tuples; **columns** are
  the world directions of index axes (i, j, k).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import SimpleITK as sitk
from aind_registration_utils.ants import apply_ants_transforms_to_point_arr
from aind_s3_cache.json_utils import get_json
from aind_s3_cache.s3_cache import (
    get_local_path_for_resource,
)
from aind_s3_cache.uri_utils import as_pathlike, as_string, join_any
from numpy.typing import NDArray
from packaging.version import Version

from aind_zarr_utils.annotations import annotation_indices_to_anatomical
from aind_zarr_utils.neuroglancer import (
    get_image_sources,
    neuroglancer_annotations_to_indices,
)
from aind_zarr_utils.pipeline_domain_selector import (
    Header,
    OverlaySelector,
    apply_overlays,
    estimate_pipeline_multiscale,
    get_selector,
)
from aind_zarr_utils.zarr import (
    _open_zarr,
    _unit_conversion,
    _zarr_to_global,
    zarr_to_sitk_stub,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from ome_zarr.reader import Node

T = TypeVar("T", int, float)


@dataclass(slots=True, frozen=True)
class TransformChain:
    """
    A pair of forward/reverse ANTs transform chains plus inversion flags.

    Parameters
    ----------
    fixed : str
        Name of the fixed space (e.g., ``"template"`` or ``"ccf"``).
    moving : str
        Name of the moving space (e.g., ``"individual"`` or ``"template"``).
    forward_chain : list[str]
        Paths (relative) for forward mapping ``moving → fixed``.
    forward_chain_invert : list[bool]
        Per-transform flags indicating inversion when applying forward map.
    reverse_chain : list[str]
        Paths (relative) for reverse mapping ``fixed → moving``.
    reverse_chain_invert : list[bool]
        Per-transform flags indicating inversion for reverse map.

    Notes
    -----
    - Order matters: ANTs expects displacement fields/affines in the
      sequence they were produced (usually warp then affine).
    """

    fixed: str
    moving: str
    forward_chain: list[str]
    forward_chain_invert: list[bool]
    reverse_chain: list[str]
    reverse_chain_invert: list[bool]


@dataclass(slots=True, frozen=True)
class TemplatePaths:
    """
    Base URI for a transform set and its associated :class:`TransformChain`.

    Parameters
    ----------
    base : str
        Base URI/prefix containing transform files.
    chain : TransformChain
        Transform chain definition rooted at ``base``.
    """

    base: str
    chain: TransformChain


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
    Wrapper around :func:`_zarr_base_name_pathlike` for any URI style.

    Parameters
    ----------
    base : str
        URI or path pointing at / inside a Zarr hierarchy.

    Returns
    -------
    str or None
        Base name without suffixes, or ``None`` if not detected.
    """
    kind, bucket, p = as_pathlike(base)
    return _zarr_base_name_pathlike(p)


_PIPELINE_TEMPLATE_TRANSFORM_CHAINS: dict[str, TransformChain] = {
    "SmartSPIM-template_2024-05-16_11-26-14": TransformChain(
        fixed="ccf",
        moving="template",
        forward_chain=[
            "spim_template_to_ccf_syn_1Warp_25.nii.gz",
            "spim_template_to_ccf_syn_0GenericAffine_25.mat",
        ],
        forward_chain_invert=[False, False],
        reverse_chain=[
            "spim_template_to_ccf_syn_0GenericAffine_25.mat",
            "spim_template_to_ccf_syn_1InverseWarp_25.nii.gz",
        ],
        reverse_chain_invert=[True, False],
    )
}

_PIPELINE_TEMPLATE_TRANSFORMS: dict[str, TemplatePaths] = {
    "SmartSPIM-template_2024-05-16_11-26-14": TemplatePaths(
        base="s3://aind-open-data/SmartSPIM-template_2024-05-16_11-26-14/",
        chain=_PIPELINE_TEMPLATE_TRANSFORM_CHAINS[
            "SmartSPIM-template_2024-05-16_11-26-14"
        ],
    )
}

_PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS: dict[int, TransformChain] = {
    3: TransformChain(
        fixed="template",
        moving="individual",
        forward_chain=[
            "ls_to_template_SyN_1Warp.nii.gz",
            "ls_to_template_SyN_0GenericAffine.mat",
        ],
        forward_chain_invert=[False, False],
        reverse_chain=[
            "ls_to_template_SyN_0GenericAffine.mat",
            "ls_to_template_SyN_1InverseWarp.nii.gz",
        ],
        reverse_chain_invert=[True, False],
    )
}


def _get_processing_pipeline_data(
    processing_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Return validated processing pipeline metadata.

    Parameters
    ----------
    processing_data : dict
        Top-level metadata (e.g. contents of ``processing.json``) expected
        to contain a ``processing_pipeline`` key with a semantic version.

    Returns
    -------
    dict
        The nested ``processing_pipeline`` dictionary.

    Raises
    ------
    ValueError
        If the pipeline version is missing or the major version is not 3.
    """
    ver_str = processing_data.get("processing_pipeline", {}).get(
        "pipeline_version", None
    )
    if not ver_str:
        raise ValueError("Missing pipeline version")
    pipeline_ver = int(ver_str.split(".")[0])
    if pipeline_ver != 3:
        raise ValueError(f"Unsupported pipeline version: {pipeline_ver}")
    pipeline: dict[str, Any] = processing_data.get("processing_pipeline", {})
    return pipeline


def _get_zarr_import_process(
    processing_data: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Locate the *Image importing* process block.

    Parameters
    ----------
    processing_data : dict
        Processing metadata supplying ``data_processes`` list.

    Returns
    -------
    dict or None
        Matching process dict or ``None`` if not present.
    """
    pipeline = _get_processing_pipeline_data(processing_data)
    want_name = "Image importing"
    proc = next(
        (p for p in pipeline["data_processes"] if p.get("name") == want_name),
        None,
    )
    return proc


def _get_image_atlas_alignment_process(
    processing_data: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Locate the *Image atlas alignment* process for SmartSPIM → CCF.

    The process is uniquely identified by name plus a notes string describing
    the LS → template → CCF chain.

    Parameters
    ----------
    processing_data : dict
        Processing metadata.

    Returns
    -------
    dict or None
        Matching process dict or ``None`` if not found.
    """
    pipeline = _get_processing_pipeline_data(processing_data)
    want_name = "Image atlas alignment"
    want_notes = (
        "Template based registration: LS -> template -> Allen CCFv3 Atlas"
    )

    proc = next(
        (
            p
            for p in pipeline["data_processes"]
            if p.get("name") == want_name and p.get("notes") == want_notes
        ),
        None,
    )
    return proc


def image_atlas_alignment_path_relative_from_processing(
    processing_data: dict[str, Any],
) -> str | None:
    """
    Return relative path to atlas alignment outputs for a processing run.

    The relative path (if determinable) has the form::

        image_atlas_alignment/<channel>/

    where ``<channel>`` is derived from the base name of the input LS Zarr.

    Parameters
    ----------
    processing_data : dict
        Processing metadata.

    Returns
    -------
    str or None
        Relative path or ``None`` if the required process / channel can't
        be resolved.
    """
    proc = _get_image_atlas_alignment_process(processing_data)
    input_zarr = proc.get("input_location") if proc else None
    channel = (
        _zarr_base_name_pathlike(PurePosixPath(input_zarr))
        if input_zarr
        else None
    )
    rel_path = f"image_atlas_alignment/{channel}/" if channel else None

    return rel_path


def mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> sitk.Image:
    """
    Construct a SimpleITK stub matching pipeline spatial corrections.

    This fabricates a *minimal* image (no pixel data read) that reflects
    the spatial domain (spacing, direction, origin) the SmartSPIM pipeline
    would have produced after applying registered overlays and multiscale
    logic.

    Parameters
    ----------
    zarr_uri : str
        URI of the raw Zarr store.
    metadata : dict
        ND metadata (instrument + acquisition) used by overlays.
    processing_data : dict
        Processing metadata containing version / process list.
    overlay_selector : OverlaySelector, optional
        Selector used to obtain overlay sequence; defaults to the global
        selector.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    sitk.Image
        Stub image with corrected spatial metadata.

    Raises
    ------
    ValueError
        If the needed import process / version is absent.
    """
    proc = _get_zarr_import_process(processing_data)
    if not proc:
        raise ValueError(
            "Could not find zarr import process in processing data"
        )

    pipeline_version = proc.get("code_version")
    if not pipeline_version:
        raise ValueError("Pipeline version not found in zarr import process")
    if opened_zarr is None:
        image_node, zarr_meta = _open_zarr(zarr_uri)
    else:
        image_node, zarr_meta = opened_zarr
    multiscale_no = estimate_pipeline_multiscale(
        zarr_meta, Version(pipeline_version)
    )

    stub_img, size_ijk = zarr_to_sitk_stub(
        zarr_uri,
        metadata,
        opened_zarr=(image_node, zarr_meta),
    )

    # Convert stub to Header for domain corrections.
    base_header = Header.from_sitk(stub_img, size_ijk)

    # Select and apply overlays based on pipeline version and metadata.
    overlays = overlay_selector.select(version=pipeline_version, meta=metadata)
    corrected_header, applied = apply_overlays(
        base_header, overlays, metadata, multiscale_no
    )

    # Return corrected stub image.
    return corrected_header.as_sitk()


def pipeline_transforms(
    zarr_uri: str,
    processing_data: dict[str, Any],
    *,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
) -> tuple[TemplatePaths, TemplatePaths]:
    """
    Return individual→template and template→CCF transform path data.

    Parameters
    ----------
    zarr_uri : str
        URI to an LS acquisition Zarr.
    processing_data : dict
        Processing metadata.
    template_used : str, optional
        Key identifying which template transform set to apply.
    template_base : str, optional
        Base path for the template transforms. If ``None``, the default from
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS` is used. Defaults to ``None``.

    Returns
    -------
    (TemplatePaths, TemplatePaths)
        First element: individual→template chain.
        Second element: template→CCF chain.

    Raises
    ------
    ValueError
        If the alignment path cannot be inferred from processing metadata.
    """
    uri_type, bucket, zarr_pathlike = as_pathlike(zarr_uri)
    asset_pathlike = _asset_from_zarr_pathlike(zarr_pathlike)
    alignment_rel_path = image_atlas_alignment_path_relative_from_processing(
        processing_data
    )
    if alignment_rel_path is None:
        raise ValueError(
            "Could not determine image atlas alignment path from "
            "processing data"
        )
    alignment_path = as_string(
        uri_type,
        bucket,
        asset_pathlike / alignment_rel_path,
    )
    individual_ants_paths = TemplatePaths(
        alignment_path,
        _PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS[3],
    )
    if template_base:
        template_ants_paths = TemplatePaths(
            template_base, _PIPELINE_TEMPLATE_TRANSFORM_CHAINS[template_used]
        )
    else:
        template_ants_paths = _PIPELINE_TEMPLATE_TRANSFORMS[template_used]
    return individual_ants_paths, template_ants_paths


def _pipeline_image_transforms_local_paths(
    individual_ants_paths: TemplatePaths,
    template_ants_paths: TemplatePaths,
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = False,
    cache_dir: str | os.PathLike | None = None,
) -> tuple[list[str], list[bool]]:
    img_transforms_individual_is_inverted = (
        individual_ants_paths.chain.forward_chain_invert
    )
    img_transforms_template_is_inverted = (
        template_ants_paths.chain.forward_chain_invert
    )

    img_transforms_individual_paths = [
        get_local_path_for_resource(
            join_any(individual_ants_paths.base, p),
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
        ).path
        for p in individual_ants_paths.chain.forward_chain
    ]
    img_transforms_template_paths = [
        get_local_path_for_resource(
            join_any(template_ants_paths.base, p),
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
        ).path
        for p in template_ants_paths.chain.forward_chain
    ]

    img_transform_paths = (
        img_transforms_template_paths + img_transforms_individual_paths
    )
    img_transform_paths_str = [str(p) for p in img_transform_paths]
    img_transform_is_inverted = (
        img_transforms_template_is_inverted
        + img_transforms_individual_is_inverted
    )
    return img_transform_paths_str, img_transform_is_inverted


def pipeline_image_transforms_local_paths(
    zarr_uri: str,
    processing_data: dict[str, Any],
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = False,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
) -> tuple[list[str], list[bool]]:
    """
    Resolve local filesystem paths to the image transform chain files.

    Download (or locate in cache) all ANTs transform components needed to
    map individual LS acquisition images into CCF space.

    Parameters
    ----------
    zarr_uri : str
        Acquisition Zarr URI.
    processing_data : dict
        Processing metadata.
    s3_client : S3Client, optional
        Boto3 S3 client (typed) for authenticated access.
    anonymous : bool, optional
        Use unsigned S3 access if ``True``.
    cache_dir : str or PathLike, optional
        Directory to cache downloaded resources.
    template_used : str, optional
        Template transform key (see
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS`).
    template_base : str, optional
        Base path for the template transforms. If ``None``, the default from
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS` will be used. Defaults to
        ``None``.

    Returns
    -------
    list[str]
        Paths to image transform files in the application order (forward
        chains).
    list[bool]
        Flags indicating whether each transform should be inverted.
    """
    individual_ants_paths, template_ants_paths = pipeline_transforms(
        zarr_uri,
        processing_data,
        template_used=template_used,
        template_base=template_base,
    )
    return _pipeline_image_transforms_local_paths(
        individual_ants_paths,
        template_ants_paths,
        s3_client=s3_client,
        anonymous=anonymous,
        cache_dir=cache_dir,
    )


def _pipeline_point_transforms_local_paths(
    individual_ants_paths: TemplatePaths,
    template_ants_paths: TemplatePaths,
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = False,
    cache_dir: str | os.PathLike | None = None,
) -> tuple[list[str], list[bool]]:
    pt_transforms_individual_is_inverted = (
        individual_ants_paths.chain.reverse_chain_invert
    )
    pt_transforms_template_is_inverted = (
        template_ants_paths.chain.reverse_chain_invert
    )

    pt_transforms_individual_paths = [
        get_local_path_for_resource(
            join_any(individual_ants_paths.base, p),
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
        ).path
        for p in individual_ants_paths.chain.reverse_chain
    ]
    pt_transforms_template_paths = [
        get_local_path_for_resource(
            join_any(template_ants_paths.base, p),
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
        ).path
        for p in template_ants_paths.chain.reverse_chain
    ]

    pt_transform_paths = (
        pt_transforms_individual_paths + pt_transforms_template_paths
    )
    pt_transform_paths_str = [str(p) for p in pt_transform_paths]
    pt_transform_is_inverted = (
        pt_transforms_individual_is_inverted
        + pt_transforms_template_is_inverted
    )
    return pt_transform_paths_str, pt_transform_is_inverted


def pipeline_point_transforms_local_paths(
    zarr_uri: str,
    processing_data: dict[str, Any],
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = False,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
) -> tuple[list[str], list[bool]]:
    """
    Resolve local filesystem paths to the point transform chain files.

    Download (or locate in cache) all ANTs transform components needed to
    map individual LS acquisition points into CCF space.

    Parameters
    ----------
    zarr_uri : str
        Acquisition Zarr URI.
    processing_data : dict
        Processing metadata.
    s3_client : S3Client, optional
        Boto3 S3 client (typed) for authenticated access.
    anonymous : bool, optional
        Use unsigned S3 access if ``True``.
    cache_dir : str or PathLike, optional
        Directory to cache downloaded resources.
    template_used : str, optional
        Template transform key (see
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS`).
    template_base : str, optional
        Base path for the template transforms. If ``None``, the default from
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS` will be used. Defaults to
        ``None``.

    Returns
    -------
    list[str]
        Paths to transform files in the application order (reverse chains).
    list[bool]
        Flags indicating whether each transform should be inverted.
    """
    individual_ants_paths, template_ants_paths = pipeline_transforms(
        zarr_uri,
        processing_data,
        template_used=template_used,
        template_base=template_base,
    )
    return _pipeline_point_transforms_local_paths(
        individual_ants_paths,
        template_ants_paths,
        s3_client=s3_client,
        anonymous=anonymous,
        cache_dir=cache_dir,
    )


def pipeline_transforms_local_paths(
    zarr_uri: str,
    processing_data: dict[str, Any],
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = False,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
) -> tuple[list[str], list[bool]]:
    """
    Resolve local filesystem paths to the transform chain files.

    Download (or locate in cache) all ANTs transform components needed to
    map individual LS acquisition images and points to CCF space.

    The "image" and "points" transforms are inverses of each other, so if you
    need to map points from ccf → individual LS space, use the "image"
    transform.

    Parameters
    ----------
    zarr_uri : str
        Acquisition Zarr URI.
    processing_data : dict
        Processing metadata.
    s3_client : S3Client, optional
        Boto3 S3 client (typed) for authenticated access.
    anonymous : bool, optional
        Use unsigned S3 access if ``True``.
    cache_dir : str or PathLike, optional
        Directory to cache downloaded resources.
    template_used : str, optional
        Template transform key (see
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS`).
    template_base : str, optional
        Base path for the template transforms. If ``None``, the default from
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS` will be used. Defaults to
        ``None``.

    Returns
    -------
    list[str]
        Paths to point transform files in the application order (reverse
        chains).
    list[bool]
        Flags indicating whether each point transform should be inverted.
    list[str]
        Paths to image transform files in the application order (forward
        chains).
    list[bool]
        Flags indicating whether each image transform should be inverted.
    """
    individual_ants_paths, template_ants_paths = pipeline_transforms(
        zarr_uri,
        processing_data,
        template_used=template_used,
        template_base=template_base,
    )
    pt_transform_paths_str, pt_transform_is_inverted = (
        _pipeline_point_transforms_local_paths(
            individual_ants_paths,
            template_ants_paths,
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
        )
    )
    img_transform_paths_str, img_transform_is_inverted = (
        _pipeline_image_transforms_local_paths(
            individual_ants_paths,
            template_ants_paths,
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
        )
    )
    return (
        pt_transform_paths_str,
        pt_transform_is_inverted,
        img_transform_paths_str,
        img_transform_is_inverted,
    )


def indices_to_ccf(
    annotation_indices: dict[str, NDArray],
    zarr_uri: str,
    metadata: dict[str, Any],
    processing_data: dict,
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = False,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
    opened_zarr: tuple[Node, dict] | None = None,
) -> dict[str, NDArray]:
    """
    Convert voxel indices (LS space) directly into CCF coordinates.

    Parameters
    ----------
    annotation_indices : dict[str, NDArray]
        Mapping layer name → (N, 3) index array (z, y, x order expected by
        downstream conversion routine).
    zarr_uri : str
        LS acquisition Zarr.
    metadata : dict
        ND metadata needed for spatial corrections.
    processing_data : dict
        Processing metadata.
    s3_client : S3Client, optional
        S3 client.
    anonymous : bool, optional
        Use unsigned access.
    cache_dir : str or PathLike, optional
        Resource cache directory.
    template_used : str, optional
        Template transform key.
    template_base : str, optional
        Base path for the template transforms. If ``None``, the default from
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS` will be used. Defaults to
        ``None``.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    dict[str, NDArray]
        Mapping layer → (N, 3) array of physical CCF coordinates.
    """
    pipeline_stub = mimic_pipeline_zarr_to_anatomical_stub(
        zarr_uri, metadata, processing_data, opened_zarr=opened_zarr
    )
    annotation_points = annotation_indices_to_anatomical(
        pipeline_stub,
        annotation_indices,
    )
    pt_transform_paths_str, pt_transform_is_inverted = (
        pipeline_point_transforms_local_paths(
            zarr_uri,
            processing_data,
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
            template_used=template_used,
            template_base=template_base,
        )
    )
    annotation_points_ccf: dict[str, NDArray] = {}
    for layer, pts in annotation_points.items():
        annotation_points_ccf[layer] = apply_ants_transforms_to_point_arr(
            pts,
            transform_list=pt_transform_paths_str,
            whichtoinvert=pt_transform_is_inverted,
        )
    return annotation_points_ccf


def neuroglancer_to_ccf(
    neuroglancer_data: dict,
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    layer_names: str | list[str] | None = None,
    return_description: bool = True,
    s3_client: S3Client | None = None,
    anonymous: bool = False,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[dict[str, NDArray], dict[str, NDArray] | None]:
    """
    Convert Neuroglancer annotation JSON into CCF coordinates.

    Parameters
    ----------
    neuroglancer_data : dict
        Parsed Neuroglancer state JSON.
    zarr_uri : str
        LS acquisition Zarr.
    metadata : dict
        ND metadata.
    processing_data : dict
        Processing metadata.
    layer_names : str | list[str] | None, optional
        Subset of annotation layer names to include; all if ``None``.
    return_description : bool, optional
        Whether to include description lists in the second return value.
    s3_client : S3Client, optional
        S3 client.
    anonymous : bool, optional
        Use unsigned S3 access if ``True``.
    cache_dir : str or PathLike, optional
        Cache directory for transform downloads.
    template_used : str, optional
        Template transform key.
    template_base : str, optional
        Base path for the template transforms. If ``None``, the default from
        :data:`_PIPELINE_TEMPLATE_TRANSFORMS` will be used. Defaults to
        ``None``.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    tuple
        ``(annotation_points_ccf, descriptions)`` where ``descriptions`` is
        ``None`` if ``return_description`` is ``False``.
    """
    # Create pipeline-corrected stub image for coordinate transformations.
    annotation_indices, descriptions = neuroglancer_annotations_to_indices(
        neuroglancer_data,
        layer_names=layer_names,
        return_description=return_description,
    )
    annotation_points_ccf = indices_to_ccf(
        annotation_indices,
        zarr_uri,
        metadata,
        processing_data,
        s3_client=s3_client,
        anonymous=anonymous,
        cache_dir=cache_dir,
        template_used=template_used,
        template_base=template_base,
        opened_zarr=opened_zarr,
    )
    return annotation_points_ccf, descriptions


def alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
    asset_uri: str | None = None, a_zarr_uri: str | None = None
) -> tuple[str, dict, dict]:
    """
    Return the alignment uris for a given Zarr path.

    """
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
    metadata = get_json(metadata_uri)
    processing_data = get_json(processing_uri)
    alignment_rel_path = image_atlas_alignment_path_relative_from_processing(
        processing_data
    )
    if alignment_rel_path is None:
        raise ValueError(
            "Could not determine image atlas alignment path from "
            "processing data"
        )
    channel = PurePosixPath(alignment_rel_path).stem
    zarr_pathlike = (
        asset_pathlike / f"image_tile_fusing/OMEZarr/{channel}.zarr"
    )
    zarr_uri = as_string(uri_type, bucket, zarr_pathlike)
    return zarr_uri, metadata, processing_data


def neuroglancer_to_ccf_auto_metadata(
    neuroglancer_data: dict,
    asset_uri: str | None = None,
    **kwargs: Any,
) -> tuple[dict[str, NDArray], dict[str, NDArray] | None]:
    """Resolve pipeline metadata files then convert annotations to CCF.

    This is a convenience wrapper that infers the acquisition (LS) Zarr URI
    from a Neuroglancer state (``image_sources``), loads the accompanying
    ``metadata.nd.json`` and ``processing.json`` files located at the asset
    root, and then delegates to :func:`neuroglancer_to_ccf`.

    Parameters
    ----------
    neuroglancer_data : dict
        Parsed Neuroglancer state JSON containing an ``image_sources``
        section referencing at least one LS Zarr.
    asset_uri : str, optional
        Base URI for the asset containing the Zarr and metadata files. If
        ``None``, the asset root is inferred from the Zarr URI in
        ``neuroglancer_data``.
    **kwargs : Any
        Forwarded keyword arguments accepted by :func:`neuroglancer_to_ccf`.
        Common keys include:

        - ``layer_names`` : str | list[str] | None
        - ``return_description`` : bool
        - ``s3_client`` : S3Client | None
        - ``anonymous`` : bool
        - ``cache_dir`` : str | os.PathLike | None
        - ``template_used`` : str

    Returns
    -------
    tuple
        ``(annotation_points_ccf, descriptions)`` where
        ``annotation_points_ccf`` is a mapping ``layer -> (N,3) NDArray`` of
        CCF coordinates and ``descriptions`` is a mapping ``layer -> list`` of
        point descriptions or ``None`` if descriptions were not requested.

    Raises
    ------
    ValueError
        If no image sources can be found in ``neuroglancer_data``.
    """
    if asset_uri is None:
        image_sources = get_image_sources(neuroglancer_data)
        # Get first image source in dict
        a_zarr_uri = next(iter(image_sources.values()), None)
        if a_zarr_uri is None:
            raise ValueError("No image sources found in neuroglancer data")
        zarr_uri, metadata, processing_data = (
            alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
                a_zarr_uri=a_zarr_uri
            )
        )
    else:
        zarr_uri, metadata, processing_data = (
            alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
                asset_uri=asset_uri
            )
        )
    return neuroglancer_to_ccf(
        neuroglancer_data,
        zarr_uri=zarr_uri,
        metadata=metadata,
        processing_data=processing_data,
        **kwargs,
    )


def swc_data_to_zarr_indices(
    swc_point_dict: dict[str, NDArray],
    zarr_uri: str,
    swc_point_order: str = "zyx",
    swc_point_units: str = "micrometer",
    opened_zarr: tuple[Node, dict] | None = None,
) -> dict[str, NDArray]:
    """Convert SWC coordinates to zarr indices.

    Parameters
    ----------
    swc_point_dict : dict[str, NDArray]
        Dictionary containing SWC points for a set of neurons. Keys are
        neuron IDs and values are (N, 3) arrays of SWC point coordinates.
    zarr_uri : str
        URI of the LS acquisition Zarr.
    processing_data : dict
        Processing metadata with pipeline version and process list.
    swc_point_order : str, optional
        Order of the zarr coordinates in the input arrays. Default is 'zyx'.
    swc_point_units : str, optional
        Units of the input coordinates. Default is 'microns'.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    dict[str, NDArray]
        Mapping neuron ID → (N, 3) array of zarr indices.
    """
    unit_scale = _unit_conversion(swc_point_units, "millimeter")
    swc_point_order_lower = swc_point_order.lower()
    swc_to_zarr_axis_order = [swc_point_order_lower.index(ax) for ax in "zyx"]
    _, _, _, spacing_raw, _ = _zarr_to_global(
        zarr_uri, level=0, opened_zarr=opened_zarr
    )

    spacing = np.array(spacing_raw)
    swc_zarr_indices = {}
    for k, pts in swc_point_dict.items():
        pts_arr = np.asarray(pts)
        if pts_arr.ndim != 2 or pts_arr.shape[1] != 3:
            raise ValueError(
                f"Expected (N, 3) array for key {k}, got shape {pts_arr.shape}"
            )
        swc_zarr_indices[k] = np.round(
            (unit_scale * pts_arr[:, swc_to_zarr_axis_order]) / spacing
        ).astype(int)
    return swc_zarr_indices


def swc_data_to_ccf(
    swc_point_dict: dict[str, NDArray],
    alignment_zarr_uri: str,
    metadata: dict[str, Any],
    processing_data: dict[str, Any],
    *,
    swc_point_order: str = "zyx",
    swc_point_units: str = "micrometer",
    opened_zarr: tuple[Node, dict] | None = None,
    **kwargs: Any,
) -> dict[str, NDArray]:
    """Convert SWC annotations to CCF coordinates.

    Converts SWC coordinates to zarr indices and then converts these indices to
    CCF coordinates. This function requires the Zarr URI and metadata to be
    provided explicitly.

    Parameters
    ----------
    swc_point_dict : dict[str, NDArray]
        Dictionary containing SWC points for a set of neurons. Keys are
        neuron IDs and values are (N, 3) arrays of SWC point coordinates.
    alignment_zarr_uri : str
        URI of the LS acquisition Zarr.
    metadata : dict
        ND metadata with acquisition information.
    processing_data : dict
        Processing metadata with pipeline version and process list.
    swc_point_order : str, optional
        Order of the zarr coordinates in the input arrays. Default is 'zyx'.
    swc_point_units : str, optional
        Units of the input coordinates. Default is 'microns'.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.
    **kwargs : Any
        Forwarded keyword arguments accepted by :func:`indices_to_ccf`.

    Returns
    -------
    dict[str, NDArray]
        Mapping neuron ID → (N, 3) array of physical CCF coordinates in LPS.
    """
    if opened_zarr is None:
        an_open_zarr = _open_zarr(alignment_zarr_uri)
    else:
        an_open_zarr = opened_zarr

    swc_zarr_indices = swc_data_to_zarr_indices(
        swc_point_dict,
        alignment_zarr_uri,
        swc_point_order=swc_point_order,
        swc_point_units=swc_point_units,
        opened_zarr=an_open_zarr,
    )
    swc_pts_ccf = indices_to_ccf(
        swc_zarr_indices,
        alignment_zarr_uri,
        metadata,
        processing_data,
        opened_zarr=an_open_zarr,
        **kwargs,
    )
    return swc_pts_ccf


def swc_data_to_ccf_auto_metadata(
    swc_point_dict: dict[str, NDArray],
    asset_uri: str,
    swc_point_order: str = "zyx",
    swc_point_units: str = "micrometer",
    **kwargs: Any,
) -> dict[str, NDArray]:
    """Resolve pipeline metadata files then convert SWC annotations to CCF.

    This is a convenience wrapper that infers the location of and loads the
    accompanying ``metadata.nd.json`` and ``processing.json`` files located at
    the asset root, and then delegates to :func:`swc_data_to_ccf`.

    Parameters
    ----------
    swc_point_dict : dict[str, NDArray]
        Dictionary containing SWC points for a set of neurons. Keys are
        neuron IDs and values are (N, 3) arrays of SWC point coordinates.
    asset_uri : str
        Base URI for the asset containing the Zarr and metadata files.
    swc_point_order : str, optional
        Order of the zarr coordinates in the input arrays. Default is 'zyx'.
    swc_point_units : str, optional
        Units of the input coordinates. Default is 'microns'.
    **kwargs : Any
        Forwarded keyword arguments accepted by :func:`indices_to_ccf`.

    Returns
    -------
    dict[str, NDArray]
        Mapping neuron ID → (N, 3) array of physical CCF coordinates in LPS.
    """
    zarr_uri, metadata, processing_data = (
        alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            asset_uri=asset_uri
        )
    )
    return swc_data_to_ccf(
        swc_point_dict,
        zarr_uri,
        metadata,
        processing_data,
        swc_point_order=swc_point_order,
        swc_point_units=swc_point_units,
        **kwargs,
    )
