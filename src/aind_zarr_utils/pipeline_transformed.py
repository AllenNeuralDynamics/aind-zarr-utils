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
from aind_anatomical_utils.anatomical_volume import (
    AnatomicalHeader,
    fix_corner_compute_origin,
)
from aind_anatomical_utils.coordinate_systems import _OPPOSITE_AXES
from aind_registration_utils.ants import (
    apply_ants_transforms_to_point_arr,
)
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
    OverlaySelector,
    apply_overlays,
    estimate_pipeline_multiscale,
    get_selector,
)
from aind_zarr_utils.zarr import (
    _open_zarr,
    _unit_conversion,
    _zarr_to_scaled,
    zarr_to_ants,
    zarr_to_sitk,
    zarr_to_sitk_stub,
)

if TYPE_CHECKING:
    from ants.core import ANTsImage  # type: ignore[import-untyped]
    from mypy_boto3_s3 import S3Client
    from ome_zarr.reader import Node  # type: ignore[import-untyped]

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
    if pipeline_ver not in set((3, 4)):
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


def _pipeline_anatomical_check_args(
    zarr_uri: str,
    processing_data: dict[str, Any],
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[dict[str, Any], str, Node, dict, int | None]:
    """
    Validate and extract needed metadata for pipeline anatomical header.

    Parameters
    ----------
    zarr_uri : str
        URI of the raw Zarr store.
    processing_data : dict
        Processing metadata containing version / process list.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    import_process : dict
        The zarr import process metadata.
    pipeline_version : str
        The pipeline version string.
    image_node : Node
        The root node of the opened Zarr image.
    zarr_meta : dict
        Metadata from the Zarr store.
    multiscale_no : int or None
        Estimated multiscale number, if determinable.
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
    return proc, pipeline_version, image_node, zarr_meta, multiscale_no


def _apply_pipeline_overlays_to_header(
    base_header: AnatomicalHeader,
    pipeline_version: str,
    metadata: dict,
    multiscale_no: int | None,
    *,
    overlay_selector: OverlaySelector = get_selector(),
) -> tuple[AnatomicalHeader, list[str]]:
    """
    Select and apply pipeline overlays to a base anatomical header.

    Parameters
    ----------
    base_header : AnatomicalHeader
        The base anatomical header to modify.
    pipeline_version : str
        The pipeline version string.
    metadata : dict
        ND metadata (instrument + acquisition) used by overlays.
    multiscale_no : int or None
        Estimated multiscale number, if determinable.
    overlay_selector : OverlaySelector, optional
        Selector used to obtain overlay sequence; defaults to the global
        selector.

    Returns
    -------
    AnatomicalHeader
        Corrected anatomical header with overlays applied.
    list[str]
        List of applied overlay names.
    """
    overlays = overlay_selector.select(version=pipeline_version, meta=metadata)
    return apply_overlays(base_header, overlays, metadata, multiscale_no or 3)


def _mimic_pipeline_anatomical_header(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[AnatomicalHeader, list[str], AnatomicalHeader]:
    """
    Construct an AnatomicalHeader matching pipeline spatial corrections.

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
    AnatomicalHeader
        Corrected anatomical header with overlays applied.
    list[str]
        List of applied overlay names.
    AnatomicalHeader
        Base anatomical header before overlays were applied.
    """
    # Validate and extract needed metadata.
    _, pipeline_version, image_node, zarr_meta, multiscale_no = (
        _pipeline_anatomical_check_args(
            zarr_uri, processing_data, opened_zarr=opened_zarr
        )
    )

    stub_img, size_ijk = zarr_to_sitk_stub(
        zarr_uri,
        metadata,
        opened_zarr=(image_node, zarr_meta),
    )

    # Convert stub to AnatomicalHeader for domain corrections.
    base_header = AnatomicalHeader.from_sitk(stub_img, size_ijk)

    # Select and apply overlays based on pipeline version and metadata.
    header, applied = _apply_pipeline_overlays_to_header(
        base_header,
        pipeline_version,
        metadata,
        multiscale_no,
        overlay_selector=overlay_selector,
    )
    return header, applied, base_header


def base_and_pipeline_anatomical_stub(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[sitk.Image, sitk.Image, tuple[int, int, int]]:
    """
    Return both the base and pipeline-corrected anatomical SimpleITK stubs.

    This convenience helper builds two lightweight (no pixel data) SimpleITK
    images representing (1) the uncorrected spatial header derived directly
    from the Zarr metadata and (2) the header after applying all pipeline
    overlays appropriate for the supplied processing metadata. It also returns
    the native voxel size (IJK dimensions) of the acquisition.

    Parameters
    ----------
    zarr_uri : str
        URI of the raw acquisition Zarr store used to derive the base header.
    metadata : dict
        ND (instrument/acquisition) metadata consulted by overlay predicates.
    processing_data : dict
        Processing metadata containing the pipeline version and process list
        used to select overlays.
    overlay_selector : OverlaySelector, optional
        Selector that resolves the ordered list of overlays to apply based
        on ``pipeline_version`` and acquisition metadata. Defaults to the
        global selector from
        :func:`~aind_zarr_utils.pipeline_domain_selector.get_selector`.
    opened_zarr : tuple[Node, dict] | None, optional
        Pre-opened ``(image_node, zarr_meta)`` tuple. If provided, avoids an
        additional Zarr open; if ``None`` the Zarr is opened internally.

    Returns
    -------
    base_stub : sitk.Image
        SimpleITK stub image whose header reflects the original (uncorrected)
        spatial metadata.
    pipeline_stub : sitk.Image
        SimpleITK stub image whose header reflects all pipeline overlay
        corrections (origin, spacing, direction).
    native_size : tuple[int, int, int]
        The voxel dimensions (I, J, K) of the acquisition in index space.

    Notes
    -----
    - Both returned images contain no pixel buffer; they are produced via
      ``AnatomicalHeader.as_sitk_stub()`` for header-only operations.
    - Use :func:`mimic_pipeline_zarr_to_anatomical_stub` if you only need the
      corrected stub.
    - Coordinates follow ITK LPS convention and spacing is in millimeters.
    """

    corrected_header, _, base_header = _mimic_pipeline_anatomical_header(
        zarr_uri,
        metadata,
        processing_data,
        overlay_selector=overlay_selector,
        opened_zarr=opened_zarr,
    )
    stub_img = corrected_header.as_sitk_stub()
    native_size = corrected_header.size_ijk
    return base_header.as_sitk_stub(), stub_img, native_size


def mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[sitk.Image, tuple[int, int, int]]:
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
    tuple
        The size of the image in IJK coordinates.

    Raises
    ------
    ValueError
        If the needed import process / version is absent.
    """
    corrected_header, _, _ = _mimic_pipeline_anatomical_header(
        zarr_uri,
        metadata,
        processing_data,
        overlay_selector=overlay_selector,
        opened_zarr=opened_zarr,
    )
    stub_img = corrected_header.as_sitk_stub()
    native_size = corrected_header.size_ijk
    return stub_img, native_size


def apply_pipeline_overlays_to_sitk(
    img: sitk.Image,
    zarr_uri: str,
    processing_data: dict,
    metadata: dict,
    level: int = 3,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> None:
    """
    Apply pipeline spatial overlays to a SimpleITK image header in-place.

    This function modifies the spatial metadata (spacing, origin, direction)
    of a SimpleITK image to match the corrections that would have been applied
    by the SmartSPIM processing pipeline. The correction approach differs
    depending on the pyramid level:

    - **Level 0**: Overlays are applied directly to the image header using
      the base header and overlay selection logic.
    - **Level > 0**: The level 0 corrected header is computed first, then
      spacing is scaled by ``2**level`` and the origin/direction are applied
      from the corrected header.

    Parameters
    ----------
    img : sitk.Image
        The SimpleITK image whose header will be modified **in-place**.
    zarr_uri : str
        URI of the raw Zarr store. Used to derive pipeline version and
        metadata needed for overlay application.
    processing_data : dict
        Processing metadata containing pipeline version and process list.
        Used to derive parameters for overlay application.
    metadata : dict
        ND metadata dictionary containing instrument and acquisition parameters
        required by overlay selection and application logic.
    level : int, optional
        Pyramid level of the image. Must be non-negative. Default is 3.
    overlay_selector : OverlaySelector, optional
        Selector used to obtain the overlay sequence based on pipeline version
        and metadata. Defaults to the global selector from
        :func:`~aind_zarr_utils.pipeline_domain_selector.get_selector`.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    None
        The function modifies ``img`` in-place and returns nothing.

    See Also
    --------
    apply_pipeline_overlays_to_ants : Equivalent function for ANTs images.
    mimic_pipeline_zarr_to_sitk : Create a new SimpleITK image with pipeline
        corrections applied.

    Notes
    -----
    - All spatial coordinates are in **ITK LPS** convention and
      **millimeters**.
    - For ``level > 0``, the function internally calls
      :func:`_mimic_pipeline_anatomical_header` to obtain the corrected
      level 0 header, then scales the spacing by ``2**level``.
    - The image pixel data is not modified, only the spatial metadata in
      the header.
    - **This function mutates the input image in-place.** If you need to
      preserve the original image, create a copy first.

    Examples
    --------
    Apply overlays to a level 0 image:

    ```python
    from aind_zarr_utils.pipeline_transformed import (
        apply_pipeline_overlays_to_sitk,
    )
    from aind_zarr_utils.zarr import zarr_to_sitk

    # Load image and metadata
    zarr_uri = "s3://bucket/dataset.zarr"
    metadata = {...}  # ND metadata
    processing_data = {...}  # Processing metadata

    # Create base image
    img = zarr_to_sitk(zarr_uri, metadata, level=0)

    # Apply overlays in-place
    apply_pipeline_overlays_to_sitk(
        img,
        zarr_uri,
        processing_data,
        metadata,
        level=0,
    )
    # img is now modified with pipeline corrections
    ```
    """
    # Derive pipeline-specific parameters from zarr_uri and processing_data
    _, pipeline_version, image_node, zarr_meta, multiscale_no = (
        _pipeline_anatomical_check_args(
            zarr_uri, processing_data, opened_zarr=opened_zarr
        )
    )

    if level == 0:
        # Overlays only work at level 0, so in this case we can work with them
        # directly.

        # Convert stub to AnatomicalHeader for domain corrections.
        base_header = AnatomicalHeader.from_sitk(img)

        # Select and apply overlays based on pipeline version and metadata.
        overlays = overlay_selector.select(
            version=pipeline_version, meta=metadata
        )
        corrected_header, _ = apply_overlays(
            base_header, overlays, metadata, multiscale_no or 3
        )
        corrected_header.update_sitk(img)
    elif level > 0:
        # For levels > 0, we need to correct for the downsampling factor.
        # First let's get the level 0 stub with the applied overlays.
        corrected_header, _, _ = _mimic_pipeline_anatomical_header(
            zarr_uri,
            metadata,
            processing_data,
            overlay_selector=overlay_selector,
            opened_zarr=(image_node, zarr_meta),
        )
        spacing_level_scale = 2**level
        spacing_scaled = tuple(
            s * spacing_level_scale for s in corrected_header.spacing
        )
        img.SetSpacing(spacing_scaled)
        img.SetOrigin(corrected_header.origin)
        img.SetDirection(corrected_header.direction_tuple())


def mimic_pipeline_zarr_to_sitk(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    level: int = 3,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> sitk.Image:
    """
    Construct a SimpleITK image matching pipeline spatial corrections.

    This fabricates a SimpleITK image that reflects the spatial domain
    (spacing, direction, origin) the SmartSPIM pipeline would have produced
    after applying registered overlays and multiscale logic.

    Returns
    -------
    ants.core.ANTsImage
        A new ANTs image instance reflecting the spatial domain.
    """
    if level < 0:
        raise ValueError("Level must be non-negative")
    _, pipeline_version, image_node, zarr_meta, multiscale_no = (
        _pipeline_anatomical_check_args(
            zarr_uri, processing_data, opened_zarr=opened_zarr
        )
    )

    img = zarr_to_sitk(
        zarr_uri,
        metadata,
        level=level,
        opened_zarr=(image_node, zarr_meta),
    )
    apply_pipeline_overlays_to_sitk(
        img,
        zarr_uri,
        processing_data,
        metadata,
        level,
        overlay_selector=overlay_selector,
        opened_zarr=(image_node, zarr_meta),
    )
    return img


def base_and_pipeline_zarr_to_sitk(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    level: int = 3,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Construct both base and pipeline-corrected ANTs images from Zarr.

    This fabricates an ANTs image that reflects the spatial domain (spacing,
    direction, origin) the SmartSPIM pipeline would have produced after
    applying registered overlays and multiscale logic.

    Returns
    -------
    base_img : ants.core.ANTsImage
        The uncorrected ANTs image from the Zarr at the requested level.
    pipeline_img : ants.core.ANTsImage
        A new ANTs image instance reflecting the spatial domain.
    """
    if level < 0:
        raise ValueError("Level must be non-negative")
    _, pipeline_version, image_node, zarr_meta, multiscale_no = (
        _pipeline_anatomical_check_args(
            zarr_uri, processing_data, opened_zarr=opened_zarr
        )
    )

    base_img = zarr_to_sitk(
        zarr_uri,
        metadata,
        level=level,
        opened_zarr=(image_node, zarr_meta),
    )

    pipeline_img = sitk.Image(base_img)
    apply_pipeline_overlays_to_sitk(
        pipeline_img,
        zarr_uri,
        processing_data,
        metadata,
        level,
        overlay_selector=overlay_selector,
        opened_zarr=(image_node, zarr_meta),
    )
    return base_img, pipeline_img


def apply_pipeline_overlays_to_ants(
    img: ANTsImage,
    zarr_uri: str,
    processing_data: dict,
    metadata: dict,
    level: int = 3,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> None:
    """
    Apply pipeline spatial overlays to an ANTs image header in-place.

    This function modifies the spatial metadata (spacing, origin, direction)
    of an ANTs image to match the corrections that would have been applied
    by the SmartSPIM processing pipeline. The correction approach differs
    depending on the pyramid level:

    - **Level 0**: Overlays are applied directly to the image header using
      the base header and overlay selection logic.
    - **Level > 0**: The level 0 corrected header is computed first (in
      SimpleITK convention), then spacing is scaled by ``2**level`` and
      coordinate system conversions are applied to account for the ANTs vs
      SimpleITK array ordering differences.

    Parameters
    ----------
    img : ANTsImage
        The ANTs image whose header will be modified **in-place**.
    zarr_uri : str
        URI of the raw Zarr store. Used to derive pipeline version and
        metadata needed for overlay application.
    processing_data : dict
        Processing metadata containing pipeline version and process list.
        Used to derive parameters for overlay application.
    metadata : dict
        ND metadata dictionary containing instrument and acquisition parameters
        required by overlay selection and application logic.
    level : int, optional
        Pyramid level of the image. Must be non-negative. Default is 3.
    overlay_selector : OverlaySelector, optional
        Selector used to obtain the overlay sequence based on pipeline version
        and metadata. Defaults to the global selector from
        :func:`~aind_zarr_utils.pipeline_domain_selector.get_selector`.
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    None
        The function modifies ``img`` in-place and returns nothing.

    See Also
    --------
    apply_pipeline_overlays_to_sitk : Equivalent function for SimpleITK
        images.
    mimic_pipeline_zarr_to_ants : Create a new ANTs image with pipeline
        corrections applied.

    Notes
    -----
    - All spatial coordinates are in **ITK LPS** convention and
      **millimeters**.
    - For ``level > 0``, the function internally calls
      :func:`_mimic_pipeline_anatomical_header` to obtain the corrected
      level 0 header in SimpleITK convention, then applies coordinate
      transformations to account for ANTs vs SimpleITK array ordering
      differences.
    - **ANTs vs SimpleITK ordering**: ANTs and SimpleITK interpret numpy
      arrays differently. For the same physical volume, their underlying
      array data are transposed relative to each other. This function handles
      the necessary conversions:

      - Spacing must be reversed
      - Origin must be recomputed for the opposite corner of the volume
      - Direction matrix should remain the same for known pipeline issues

    - The image pixel data is not modified, only the spatial metadata in
      the header.
    - **This function mutates the input image in-place.** If you need to
      preserve the original image, create a copy first.

    Examples
    --------
    Apply overlays to a level 0 ANTs image:

    ```python
    from aind_zarr_utils.pipeline_transformed import (
        apply_pipeline_overlays_to_ants,
    )
    from aind_zarr_utils.zarr import zarr_to_ants

    # Load image and metadata
    zarr_uri = "s3://bucket/dataset.zarr"
    metadata = {...}  # ND metadata
    processing_data = {...}  # Processing metadata

    # Create base image
    img = zarr_to_ants(zarr_uri, metadata, level=0)

    # Apply overlays in-place
    apply_pipeline_overlays_to_ants(
        img,
        zarr_uri,
        processing_data,
        metadata,
        level=0,
    )
    # img is now modified with pipeline corrections
    ```
    """
    # Derive pipeline-specific parameters from zarr_uri and processing_data
    _, pipeline_version, image_node, zarr_meta, multiscale_no = (
        _pipeline_anatomical_check_args(
            zarr_uri, processing_data, opened_zarr=opened_zarr
        )
    )

    if level == 0:
        # Overlays only work at level 0, so in this case we can work with them
        # directly.

        # Convert stub to AnatomicalHeader for domain corrections.
        base_header = AnatomicalHeader.from_ants(img)

        # Select and apply overlays based on pipeline version and metadata.
        overlays = overlay_selector.select(
            version=pipeline_version, meta=metadata
        )
        corrected_header, _ = apply_overlays(
            base_header, overlays, metadata, multiscale_no or 3
        )
        corrected_header.update_ants(img)
    elif level > 0:
        # For levels > 0, we need to correct for the downsampling factor.
        # First let's get the level 0 stub with the applied overlays.
        corrected_header, _, _ = _mimic_pipeline_anatomical_header(
            zarr_uri,
            metadata,
            processing_data,
            overlay_selector=overlay_selector,
            opened_zarr=(image_node, zarr_meta),
        )
        spacing_level_scale = 2**level

        # The corrected header above is calculated based on SimpleITK ordering
        # of the zarr data, which is the reverse of ANTs ordering due to how
        # these libraries accept numpy arrays of data. Even though these images
        # have the same physical interpretation, their underlying array data
        # are transposed. So, to apply the SimpleITK-based header, we need to
        # reverse the spacing tuple.
        spacing_rev_scaled = tuple(
            s * spacing_level_scale for s in reversed(corrected_header.spacing)
        )
        img.set_spacing(spacing_rev_scaled)
        # The origin is also wrong, because it is a different corner of the
        # volume.
        header_origin_code = (
            sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
                corrected_header.direction_tuple()
            )
        )
        header_origin_corner_code = "".join(
            _OPPOSITE_AXES[d] for d in header_origin_code
        )
        ants_origin, _, _ = fix_corner_compute_origin(
            img.shape,
            spacing_rev_scaled,
            img.direction,  # This should be the same
            target_point=corrected_header.origin,
            corner_code=header_origin_corner_code,
        )
        img.set_origin(ants_origin)
        # The direction matrix should be the same for known pipeline issues. If
        # not, fix


def base_and_pipeline_zarr_to_ants(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    level: int = 3,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[ANTsImage, ANTsImage]:
    """
    Construct both base and pipeline-corrected ANTs images from Zarr.

    This fabricates an ANTs image that reflects the spatial domain (spacing,
    direction, origin) the SmartSPIM pipeline would have produced after
    applying registered overlays and multiscale logic.

    Returns
    -------
    base_img : ants.core.ANTsImage
        The uncorrected ANTs image from the Zarr at the requested level.
    pipeline_img : ants.core.ANTsImage
        A new ANTs image instance reflecting the spatial domain.
    """
    if level < 0:
        raise ValueError("Level must be non-negative")
    _, pipeline_version, image_node, zarr_meta, multiscale_no = (
        _pipeline_anatomical_check_args(
            zarr_uri, processing_data, opened_zarr=opened_zarr
        )
    )

    base_img = zarr_to_ants(
        zarr_uri,
        metadata,
        level=level,
        opened_zarr=(image_node, zarr_meta),
    )

    pipeline_img = base_img.clone()
    apply_pipeline_overlays_to_ants(
        pipeline_img,
        zarr_uri,
        processing_data,
        metadata,
        level,
        overlay_selector=overlay_selector,
        opened_zarr=(image_node, zarr_meta),
    )
    return base_img, pipeline_img


def mimic_pipeline_zarr_to_ants(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    level: int = 3,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> ANTsImage:
    """
    Construct an ANTs image matching pipeline spatial corrections.

    This fabricates an ANTs image that reflects the spatial domain (spacing,
    direction, origin) the SmartSPIM pipeline would have produced after
    applying registered overlays and multiscale logic.

    Returns
    -------
    ants.core.ANTsImage
        A new ANTs image instance reflecting the spatial domain.
    """
    if level < 0:
        raise ValueError("Level must be non-negative")
    _, pipeline_version, image_node, zarr_meta, multiscale_no = (
        _pipeline_anatomical_check_args(
            zarr_uri, processing_data, opened_zarr=opened_zarr
        )
    )

    img = zarr_to_ants(
        zarr_uri,
        metadata,
        level=level,
        opened_zarr=(image_node, zarr_meta),
    )
    apply_pipeline_overlays_to_ants(
        img,
        zarr_uri,
        processing_data,
        metadata,
        level,
        overlay_selector=overlay_selector,
        opened_zarr=(image_node, zarr_meta),
    )

    return img


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
    anonymous: bool = True,
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
    anonymous: bool = True,
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
    anonymous: bool = True,
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
    anonymous: bool = True,
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
    anonymous: bool = True,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
) -> tuple[list[str], list[bool], list[str], list[bool]]:
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
    anonymous: bool = True,
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
        downstream conversion routine). Index arrays can contain continuous
        (floating-point) values for sub-voxel precision.
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
        Mapping layer → (N, 3) array of anatomical CCF coordinates in LPS.
    """
    pipeline_stub, _ = mimic_pipeline_zarr_to_anatomical_stub(
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
    anonymous: bool = True,
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


def ccf_to_indices(
    ccf_points: dict[str, NDArray],
    alignment_zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | None = None,
    opened_zarr: tuple[Node, dict] | None = None,
    scale_unit: str = "millimeter",
) -> dict[str, NDArray]:
    """
    Transform points from CCF space to continuous zarr indices.

    This function applies the inverse of the registration pipeline, converting
    points from Allen CCF space back to the continuous (sub-index) indices of
    the zarr dataset, accounting for pipeline-specific domain corrections.

    The transformation chain is:
    1. CCF → Pipeline anatomical (via ANTs image transforms)
    2. Pipeline anatomical → continuous indices (via pipeline stub)

    Parameters
    ----------
    ccf_points : dict[str, NDArray]
        Mapping layer name → (N, 3) array of CCF coordinates in LPS order.
    alignment_zarr_uri : str
        URI of the alignment zarr (channel used for registration).
    metadata : dict
        Neural Dynamics metadata (metadata.nd.json).
    processing_data : dict
        Processing pipeline metadata (processing.json).
    template_used : str, optional
        Template identifier. Default is
        "SmartSPIM-template_2024-05-16_11-26-14".
    template_base : str, optional
        Base path for template transforms. If None, uses default S3 location.
    opened_zarr : tuple, optional
        Pre-opened zarr (image_node, zarr_meta). Avoids re-opening.
    scale_unit : str, optional
        Unit for anatomical coordinates. Default is "millimeter".

    Returns
    -------
    dict[str, NDArray]
        Mapping layer name → (N, 3) array of continuous indices in z,y,x order
        used by Neuroglancer

    See Also
    --------
    indices_to_ccf : Forward transform from indices to CCF.
    ccf_to_anatomical_auto_metadata : Convenience wrapper with auto metadata.

    Notes
    -----
    - ANTs transforms output in pipeline anatomical space
    - Converts from pipeline space to LS space via shared index space
    - Returns coordinates in zarr z,y,x order (used by Neuroglancer)
    - Output coordinates match what Neuroglancer uses

    Examples
    --------
    >>> ccf_pts = {"layer1": np.array([[5000, 6000, 7000]])}  # CCF coords
    >>> anatomical_pts = ccf_to_anatomical(
    ...     ccf_pts, alignment_zarr_uri, metadata, processing_data
    ... )
    """
    # Get image transform chains (for transforming points from CCF to pipeline)
    img_transform_paths_str, img_transform_is_inverted = (
        pipeline_image_transforms_local_paths(
            alignment_zarr_uri,
            processing_data,
            template_used=template_used,
            template_base=template_base,
        )
    )

    # Concatenate all points from all layers for batch processing
    layer_names = []
    layer_sizes = []
    all_ccf_points = []

    for layer, pts in ccf_points.items():
        layer_names.append(layer)
        layer_sizes.append(len(pts))
        all_ccf_points.append(pts)

    # Stack all points into single array
    all_ccf_points_arr = np.vstack(all_ccf_points)

    # Single ANTs call for all points (reduces overhead)
    all_pipeline_points_arr = apply_ants_transforms_to_point_arr(
        all_ccf_points_arr,
        transform_list=img_transform_paths_str,
        whichtoinvert=img_transform_is_inverted,
    )

    # Re-segregate transformed points back into layers
    pipeline_anatomical_points = {}
    start_idx = 0
    for layer_name, size in zip(layer_names, layer_sizes):
        end_idx = start_idx + size
        pipeline_anatomical_points[layer_name] = all_pipeline_points_arr[
            start_idx:end_idx
        ]
        start_idx = end_idx

    # Get both stubs at once using helper function
    pipeline_stub, _ = mimic_pipeline_zarr_to_anatomical_stub(
        alignment_zarr_uri,
        metadata,
        processing_data,
        opened_zarr=opened_zarr,
    )

    # Convert pipeline anatomical → indices → LS anatomical
    ls_indices = {}

    for layer, pipeline_pts in pipeline_anatomical_points.items():
        # Convert each point: Pipeline anatomical → indices → LS anatomical
        ls_indices_layer = []

        for point_lps in pipeline_pts:
            # Pipeline anatomical → continuous indices
            # Both ANTs and SimpleITK use LPS points - no conversion needed
            continuous_idx = (
                pipeline_stub.TransformPhysicalPointToContinuousIndex(
                    point_lps.astype(np.float64)
                )
            )
            ls_indices_layer.append(np.array(continuous_idx))

        ls_indices[layer] = np.array(ls_indices_layer)

    return ls_indices


def ccf_to_indices_auto_metadata(
    ccf_points: dict[str, NDArray],
    zarr_uri: str,
    *,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    **kwargs: Any,
) -> dict[str, NDArray]:
    """
    Resolve pipeline metadata then convert CCF to indices.

    Convenience wrapper that infers and loads metadata.nd.json and
    processing.json from the asset root, then delegates to ccf_to_indices.
    Parameters
    ----------
    ccf_points : dict[str, NDArray]
        Mapping layer name → (N, 3) array of CCF coordinates in LPS order.
    zarr_uri : str
        URI of the acquisition zarr. Asset root will be inferred.
    template_used : str, optional
        Template identifier. Default is
        "SmartSPIM-template_2024-05-16_11-26-14".
    **kwargs : Any
        Forwarded to ccf_to_indices.

    Returns
    -------
    dict[str, NDArray]
        Mapping layer name → (N, 3) array of continuous indices in z,y,x order
        (used by Neuroglancer).

    See Also
    --------
    ccf_to_indices : Main function with explicit metadata parameters.

    Examples
    --------
    >>> ccf_pts = {"layer1": np.array([[5000, 6000, 7000]])}
    >>> indices_pts = ccf_to_indices_auto_metadata(ccf_pts, zarr_uri)
    """
    alignment_zarr_uri, metadata, processing_data = (
        alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            a_zarr_uri=zarr_uri
        )
    )
    return ccf_to_indices(
        ccf_points,
        alignment_zarr_uri,
        metadata,
        processing_data,
        template_used=template_used,
        **kwargs,
    )


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
        image_sources = get_image_sources(
            neuroglancer_data, remove_zarr_protocol=True
        )
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
        Mapping neuron ID → (N, 3) array of integer zarr indices (rounded from
        continuous coordinates).
    """
    unit_scale = _unit_conversion(swc_point_units, "millimeter")
    swc_point_order_lower = swc_point_order.lower()
    swc_to_zarr_axis_order = [swc_point_order_lower.index(ax) for ax in "zyx"]
    _, _, _, spacing_raw, _ = _zarr_to_scaled(
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
        Mapping neuron ID → (N, 3) array of anatomical CCF coordinates in LPS.
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
        Mapping neuron ID → (N, 3) array of anatomical CCF coordinates in LPS.
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


def indices_to_ccf_auto_metadata(
    annotation_indices: dict[str, NDArray],
    zarr_uri: str,
    **kwargs: Any,
) -> dict[str, NDArray]:
    """Resolve pipeline metadata files then convert indices to CCF.

    This is a convenience wrapper that infers the location of and loads the
    accompanying ``metadata.nd.json`` and ``processing.json`` files from the
    asset root (inferred from the Zarr URI), and then delegates to
    :func:`indices_to_ccf`.

    Parameters
    ----------
    annotation_indices : dict[str, NDArray]
        Mapping layer name → (N, 3) index array (z, y, x order expected by
        downstream conversion routine). Index arrays can contain continuous
        (floating-point) values for sub-voxel precision.
    zarr_uri : str
        URI of the acquisition Zarr file that the indices reference. The asset
        root will be inferred from this URI to locate metadata files.
    **kwargs : Any
        Forwarded keyword arguments accepted by :func:`indices_to_ccf`.
        Common keys include:

        - ``s3_client`` : S3Client | None
        - ``anonymous`` : bool
        - ``cache_dir`` : str | os.PathLike | None
        - ``template_used`` : str
        - ``template_base`` : str | None
        - ``opened_zarr`` : tuple[Node, dict] | None

    Returns
    -------
    dict[str, NDArray]
        Mapping layer → (N, 3) array of anatomical CCF coordinates in LPS.

    See Also
    --------
    indices_to_ccf : The underlying function that performs the transformation.
    neuroglancer_to_ccf_auto_metadata : Similar wrapper for Neuroglancer data.
    swc_data_to_ccf_auto_metadata : Similar wrapper for SWC data.

    Examples
    --------
    Convert annotation indices to CCF coordinates:

    >>> indices = {
    ...     "layer1": np.array([[100, 200, 300], [150, 250, 350]]),
    ...     "layer2": np.array([[50, 100, 150]])
    ... }
    >>> ccf_coords = indices_to_ccf_auto_metadata(
    ...     indices,
    ...     zarr_uri="s3://aind-open-data/dataset_123/image.zarr"
    ... )
    """
    alignment_zarr_uri, metadata, processing_data = (
        alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            a_zarr_uri=zarr_uri
        )
    )
    return indices_to_ccf(
        annotation_indices,
        alignment_zarr_uri,
        metadata,
        processing_data,
        **kwargs,
    )
