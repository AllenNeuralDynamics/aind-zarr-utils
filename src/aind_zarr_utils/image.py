"""Backend-agnostic pipeline-overlay application for SimpleITK / ANTs images.

This module is the single home for the math that takes a *pipeline-corrected*
:class:`~aind_anatomical_utils.anatomical_volume.AnatomicalHeader` (SimpleITK
convention) and projects it onto either a SimpleITK or ANTs image at an
arbitrary pyramid level.

The SmartSPIM CCF registration placed the pyramid level it registered, not
level 0, so the overlays run on that level's grid: anchoring a corner of the
level-0 grid instead misses by the level's half-voxel offset plus whatever the
pyramid truncated at that end. Every other level, level 0 included, is derived
from the registered level by :func:`_rescale_level`, which keeps each level's
voxels over their level-0 sources.

ANTs and SimpleITK transpose numpy arrays relative to each other;
:func:`_to_ants_convention` projects a SimpleITK-convention header onto an
ANTs image.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import (
    AnatomicalHeader,
    fix_corner_compute_origin,
)
from aind_anatomical_utils.coordinate_systems import _OPPOSITE_AXES
from packaging.version import Version

from aind_zarr_utils.domain.overlays import _PIPELINE_MULTISCALE_FACTOR, estimate_pipeline_multiscale
from aind_zarr_utils.domain.selector import (
    OverlaySelector,
    apply_overlays,
    get_selector,
)
from aind_zarr_utils.io.processing import _get_zarr_import_process
from aind_zarr_utils.io.zarr import _open_zarr
from aind_zarr_utils.zarr import zarr_to_sitk_stub

if TYPE_CHECKING:
    from ants.core import ANTsImage  # type: ignore[import-untyped]
    from numpy.typing import NDArray
    from ome_zarr.reader import Node  # type: ignore[import-untyped]


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
    zarr_import_version : str
        The zarr import process version string (from Image importing process).
    image_node : Node
        The root node of the opened Zarr image.
    zarr_meta : dict
        Metadata from the Zarr store.
    multiscale_no : int or None
        Estimated multiscale number, if determinable.
    """
    proc = _get_zarr_import_process(processing_data)
    if not proc:
        raise ValueError("Could not find zarr import process in processing data")

    zarr_import_version = proc.get("code_version")
    if not zarr_import_version:
        raise ValueError("Zarr import version not found in zarr import process")
    if opened_zarr is None:
        image_node, zarr_meta = _open_zarr(zarr_uri)
    else:
        image_node, zarr_meta = opened_zarr
    multiscale_no = estimate_pipeline_multiscale(zarr_meta, Version(zarr_import_version))
    return proc, zarr_import_version, image_node, zarr_meta, multiscale_no


def _apply_pipeline_overlays_to_header(
    base_header: AnatomicalHeader,
    zarr_import_version: str,
    metadata: dict,
    multiscale_no: int,
    *,
    overlay_selector: OverlaySelector = get_selector(),
) -> tuple[AnatomicalHeader, list[str]]:
    """
    Select and apply pipeline overlays to a base anatomical header.

    Parameters
    ----------
    base_header : AnatomicalHeader
        The base anatomical header to modify.
    zarr_import_version : str
        The zarr import process version string (code_version from
        Image importing process).
    metadata : dict
        ND metadata (instrument + acquisition) used by overlays.
    multiscale_no : int
        Pyramid level the registration used, relative to ``base_header``'s
        level; 0 when ``base_header`` is that level.
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
    overlays = overlay_selector.select(version=zarr_import_version, meta=metadata)
    return apply_overlays(
        base_header,
        overlays,
        metadata,
        multiscale_no,
        zarr_import_version=zarr_import_version,
    )


def _rescale_level(header: AnatomicalHeader, factor: float, size_ijk: tuple[int, int, int]) -> AnatomicalHeader:
    """Header of the pyramid level whose spacing is ``factor`` times ``header``'s, in the same frame.

    A coarse voxel is the mean of the fine voxels beneath it, so the first centers of two levels sit half a voxel
    of each apart; ``factor < 1`` goes towards level 0.

    Parameters
    ----------
    header : AnatomicalHeader
        A level's header, SimpleITK convention.
    factor : float
        Spacing ratio of the target level to ``header``'s level.
    size_ijk : tuple of int
        Size of the target level.

    Returns
    -------
    AnatomicalHeader
        The target level's header.
    """
    spacing = np.asarray(header.spacing, dtype=float)
    target_spacing = spacing * factor
    origin = np.asarray(header.origin, dtype=float) + header.direction @ ((target_spacing - spacing) / 2)
    ox, oy, oz = (float(v) for v in origin)
    sx, sy, sz = (float(v) for v in target_spacing)
    return replace(header, origin=(ox, oy, oz), spacing=(sx, sy, sz), size_ijk=size_ijk)


def _build_pipeline_header(
    zarr_uri: str,
    metadata: dict,
    processing_data: dict,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[AnatomicalHeader, list[str], AnatomicalHeader]:
    """
    Construct an AnatomicalHeader matching pipeline spatial corrections.

    The returned ``corrected`` header describes level 0 in the frame the
    registration placed its level in, and is in **SimpleITK convention**:
    its ``spacing`` is in SITK index order (= reversed compared to ANTs),
    and its ``direction`` matrix's columns correspond to SITK index axes.
    Derive other levels with :func:`_rescale_level`, and use
    :func:`_to_ants_convention` to project onto an ANTs image.

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
    corrected : AnatomicalHeader
        Corrected level-0 anatomical header (SITK convention).
    applied : list[str]
        Names of overlays that actually changed the header.
    base : AnatomicalHeader
        Base anatomical header before overlays were applied.
    """
    _, zarr_import_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
    )

    registration_level = 3 if multiscale_no is None else multiscale_no
    stub_img, size_ijk = zarr_to_sitk_stub(
        zarr_uri,
        metadata,
        opened_zarr=(image_node, zarr_meta),
    )
    base_header = AnatomicalHeader.from_sitk(stub_img, size_ijk)
    registered_stub, registered_size = zarr_to_sitk_stub(
        zarr_uri,
        metadata,
        level=registration_level,
        opened_zarr=(image_node, zarr_meta),
    )
    registered, applied = _apply_pipeline_overlays_to_header(
        AnatomicalHeader.from_sitk(registered_stub, registered_size),
        zarr_import_version,
        metadata,
        0,
        overlay_selector=overlay_selector,
    )
    header = _rescale_level(registered, _PIPELINE_MULTISCALE_FACTOR**-registration_level, size_ijk)
    return header, applied, base_header


def _to_ants_convention(
    corrected_header_sitk: AnatomicalHeader,
    ants_shape: tuple[int, ...],
    ants_direction: NDArray,
) -> tuple[tuple[float, ...], tuple[float, float, float]]:
    """Project a SITK-convention header onto the spacing/origin of an ANTs image of the same level.

    Parameters
    ----------
    corrected_header_sitk : AnatomicalHeader
        Header in SimpleITK convention, at the destination image's level.
    ants_shape : tuple of int
        Shape of the destination ANTs image (in ANTs index order).
    ants_direction : numpy.ndarray
        Direction matrix of the destination ANTs image. For the default
        pipeline rule set this matches ``corrected_header_sitk.direction``;
        the parameter is forwarded verbatim to
        :func:`fix_corner_compute_origin`.

    Returns
    -------
    spacing : tuple of float
        Spacing in ANTs index order (= reversed of ``corrected_header_sitk.spacing``).
    origin : tuple of float
        ANTs image origin (LPS) recomputed so that the anatomical corner
        opposite to the SITK origin lands at ``corrected_header_sitk.origin``.
    """
    # ANTs and SimpleITK transpose numpy arrays relative to each other; the
    # SITK-convention spacing must be reversed before being applied to ANTs.
    spacing_rev = tuple(reversed(corrected_header_sitk.spacing))

    # The SITK origin sits at the anatomical corner described by the
    # orientation code derived from the SITK direction matrix. The ANTs
    # image's index (0,0,0) corresponds to the *opposite* anatomical corner
    # (the array storage is transposed). Recompute the origin so that the
    # opposite corner of the ANTs image lands at the SITK origin.
    header_origin_code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
        corrected_header_sitk.direction_tuple()
    )
    header_origin_corner_code = "".join(_OPPOSITE_AXES[d] for d in header_origin_code)
    ants_origin, _, _ = fix_corner_compute_origin(
        ants_shape,
        spacing_rev,
        ants_direction,
        target_point=corrected_header_sitk.origin,
        corner_code=header_origin_corner_code,
    )
    return spacing_rev, ants_origin


def apply_pipeline_overlays(
    img: sitk.Image | ANTsImage,
    zarr_uri: str,
    processing_data: dict,
    metadata: dict,
    level: int = 3,
    *,
    overlay_selector: OverlaySelector = get_selector(),
    opened_zarr: tuple[Node, dict] | None = None,
) -> None:
    """Apply pipeline overlay corrections in-place to ``img``.

    The corrected level-0 header from :func:`_build_pipeline_header` is
    rescaled to ``level`` by :func:`_rescale_level` and written to ``img``;
    an ANTs image goes through :func:`_to_ants_convention` first.

    Parameters
    ----------
    img : sitk.Image or ants.core.ANTsImage
        Image whose spatial header will be modified in place.
    zarr_uri : str
        URI of the raw Zarr store.
    processing_data : dict
        Processing metadata.
    metadata : dict
        ND (instrument/acquisition) metadata.
    level : int, optional
        Pyramid level of ``img``. Default is 3.
    overlay_selector : OverlaySelector, optional
        Selector for overlay rules. Defaults to the global cached selector.
    opened_zarr : tuple, optional
        Pre-opened (image_node, zarr_meta) to avoid re-opening the Zarr.

    Returns
    -------
    None
        ``img``'s spatial metadata is mutated in place.
    """
    # Defer the ANTs import — antspyx is heavy.
    from ants.core import ANTsImage

    # Negative levels are a no-op; the orchestration functions reject them at their own boundary.
    if level < 0:
        return
    if not isinstance(img, (sitk.Image, ANTsImage)):
        raise TypeError(f"Unsupported image type: {type(img).__name__}")
    corrected, _, _ = _build_pipeline_header(
        zarr_uri,
        metadata,
        processing_data,
        overlay_selector=overlay_selector,
        opened_zarr=opened_zarr,
    )
    if isinstance(img, sitk.Image):
        si, sj, sk = img.GetSize()
        _rescale_level(corrected, _PIPELINE_MULTISCALE_FACTOR**level, (si, sj, sk)).update_sitk(img)
    else:
        ai, aj, ak = img.shape
        header = _rescale_level(corrected, _PIPELINE_MULTISCALE_FACTOR**level, (ak, aj, ai))
        new_spacing, new_origin = _to_ants_convention(header, img.shape, img.direction)
        img.set_spacing(new_spacing)
        img.set_origin(new_origin)
        # Direction is intentionally left unchanged: for the default rule set the overlays only modify origin and
        # spacing, so the ANTs direction already matches the corrected header.
