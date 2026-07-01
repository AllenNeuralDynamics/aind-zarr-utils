"""Backend-agnostic pipeline-overlay application for SimpleITK / ANTs images.

This module is the single home for the math that takes a *pipeline-corrected*
:class:`~aind_anatomical_utils.anatomical_volume.AnatomicalHeader` (built in
SimpleITK convention from a Zarr stub at level 0) and projects it onto either
a SimpleITK or ANTs image at an arbitrary pyramid level.

The two backends agree at level 0 — there it's a straightforward
``AnatomicalHeader.from_<backend>(img) → apply_overlays → update_<backend>(img)``
loop. They diverge at level > 0:

* **SimpleITK**: spacing scaled by ``2**level``; origin and direction copied
  through verbatim.
* **ANTs**: spacing reversed (because ANTs and SimpleITK transpose numpy
  arrays relative to each other) **then** scaled by ``2**level``; origin
  recomputed via :func:`~aind_anatomical_utils.anatomical_volume.fix_corner_compute_origin`
  so that the *opposite* anatomical corner of the ANTs image lands at the
  level-0 SimpleITK origin; direction left untouched (and assumed to match
  the corrected header's direction — true for the default rule set, where
  overlays only modify origin and spacing).

The level > 0 ANTs math lives in :func:`_to_ants_convention`, a small pure
helper that's directly testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import (
    AnatomicalHeader,
    fix_corner_compute_origin,
)
from aind_anatomical_utils.coordinate_systems import _OPPOSITE_AXES
from packaging.version import Version

from aind_zarr_utils.domain.overlays import estimate_pipeline_multiscale
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
    zarr_import_version : str
        The zarr import process version string (code_version from
        Image importing process).
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
    overlays = overlay_selector.select(version=zarr_import_version, meta=metadata)
    return apply_overlays(
        base_header,
        overlays,
        metadata,
        multiscale_no or 3,
        zarr_import_version=zarr_import_version,
    )


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

    The returned ``corrected`` header is in **SimpleITK convention**: its
    ``spacing`` is in SITK index order (= reversed compared to ANTs), and
    its ``direction`` matrix's columns correspond to SITK index axes. Use
    :func:`_to_ants_convention` to project it onto an ANTs image.

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
        Corrected anatomical header with overlays applied (SITK convention).
    applied : list[str]
        Names of overlays that actually changed the header.
    base : AnatomicalHeader
        Base anatomical header before overlays were applied.
    """
    _, zarr_import_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
    )

    stub_img, size_ijk = zarr_to_sitk_stub(
        zarr_uri,
        metadata,
        opened_zarr=(image_node, zarr_meta),
    )

    base_header = AnatomicalHeader.from_sitk(stub_img, size_ijk)
    header, applied = _apply_pipeline_overlays_to_header(
        base_header,
        zarr_import_version,
        metadata,
        multiscale_no,
        overlay_selector=overlay_selector,
    )
    return header, applied, base_header


def _to_ants_convention(
    corrected_header_sitk: AnatomicalHeader,
    level: int,
    ants_shape: tuple[int, ...],
    ants_direction: NDArray,
) -> tuple[tuple[float, ...], tuple[float, float, float]]:
    """Project a SITK-convention level-0 header onto ANTs spacing/origin at ``level``.

    Parameters
    ----------
    corrected_header_sitk : AnatomicalHeader
        Header in SimpleITK convention (built from a level-0 Zarr stub).
    level : int
        Pyramid level of the destination ANTs image.
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
        Spacing in ANTs index order (= reversed of ``corrected_header_sitk.spacing``)
        scaled by ``2**level``.
    origin : tuple of float
        ANTs image origin (LPS) recomputed so that the anatomical corner
        opposite to the SITK origin lands at ``corrected_header_sitk.origin``.
    """
    spacing_level_scale = 2**level
    # ANTs and SimpleITK transpose numpy arrays relative to each other; the
    # SITK-convention spacing must be reversed before being applied to ANTs.
    spacing_rev_scaled = tuple(s * spacing_level_scale for s in reversed(corrected_header_sitk.spacing))

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
        spacing_rev_scaled,
        ants_direction,
        target_point=corrected_header_sitk.origin,
        corner_code=header_origin_corner_code,
    )
    return spacing_rev_scaled, ants_origin


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

    Type-dispatches on ``img`` to handle SimpleITK and ANTs:

    * **SimpleITK + level=0** — build a base header from ``img`` via
      ``AnatomicalHeader.from_sitk``, run overlays, write back via
      ``update_sitk``.
    * **SimpleITK + level>0** — fetch the level-0 corrected header via
      :func:`_build_pipeline_header`, scale spacing by ``2**level``, copy
      origin/direction through.
    * **ANTs + level=0** — same as SITK but using ``from_ants`` /
      ``update_ants``.
    * **ANTs + level>0** — fetch the level-0 SITK-convention corrected
      header, then project to ANTs via :func:`_to_ants_convention`.

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

    _, zarr_import_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
    )

    if level == 0:
        if isinstance(img, sitk.Image):
            base_header = AnatomicalHeader.from_sitk(img)
        elif isinstance(img, ANTsImage):
            base_header = AnatomicalHeader.from_ants(img)
        else:
            raise TypeError(f"Unsupported image type: {type(img).__name__}")

        overlays = overlay_selector.select(version=zarr_import_version, meta=metadata)
        corrected, _ = apply_overlays(
            base_header,
            overlays,
            metadata,
            multiscale_no or 3,
            zarr_import_version=zarr_import_version,
        )

        if isinstance(img, sitk.Image):
            corrected.update_sitk(img)
        else:
            corrected.update_ants(img)
    elif level > 0:
        corrected, _, _ = _build_pipeline_header(
            zarr_uri,
            metadata,
            processing_data,
            overlay_selector=overlay_selector,
            opened_zarr=(image_node, zarr_meta),
        )
        if isinstance(img, sitk.Image):
            spacing_level_scale = 2**level
            spacing_scaled = tuple(s * spacing_level_scale for s in corrected.spacing)
            img.SetSpacing(spacing_scaled)
            img.SetOrigin(corrected.origin)
            img.SetDirection(corrected.direction_tuple())
        elif isinstance(img, ANTsImage):
            new_spacing, new_origin = _to_ants_convention(corrected, level, img.shape, img.direction)
            img.set_spacing(new_spacing)
            img.set_origin(new_origin)
            # Direction is intentionally left unchanged: for the default
            # rule set the overlays only modify origin/spacing, so the
            # ANTs direction already matches the corrected header.
        else:
            raise TypeError(f"Unsupported image type: {type(img).__name__}")
    # NOTE: level < 0 is silently a no-op, matching the historical behaviour
    # of apply_pipeline_overlays_to_sitk / _to_ants. The orchestration
    # functions reject negative levels at their own boundary.
