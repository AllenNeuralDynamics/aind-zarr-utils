"""Reconstruct a pipeline's spatial domain for LS → CCF mappings and apply ANTs transform chains to points/annotations.

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
from numpy.typing import NDArray
from packaging.version import Version

from aind_zarr_utils.annotations import annotation_indices_to_anatomical
from aind_zarr_utils.formats.swc import swc_data_to_indices
from aind_zarr_utils.io.metadata import _unit_conversion as _unit_conversion  # noqa: F401  # legacy re-export

# Re-exported from io/* so existing test patches (and downstream imports)
# continue to find these names at ``aind_zarr_utils.pipeline_transformed.*``.
from aind_zarr_utils.io.paths import (
    _asset_from_zarr_any as _asset_from_zarr_any,
)
from aind_zarr_utils.io.paths import (
    _asset_from_zarr_pathlike as _asset_from_zarr_pathlike,
)
from aind_zarr_utils.io.paths import (
    _zarr_base_name_any as _zarr_base_name_any,
)
from aind_zarr_utils.io.paths import (
    _zarr_base_name_pathlike as _zarr_base_name_pathlike,
)
from aind_zarr_utils.io.paths import (
    alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike as alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike,  # noqa: E501
)
from aind_zarr_utils.io.processing import (
    _get_image_atlas_alignment_process as _get_image_atlas_alignment_process,
)
from aind_zarr_utils.io.processing import (
    _get_processing_pipeline_data as _get_processing_pipeline_data,
)
from aind_zarr_utils.io.processing import (
    _get_zarr_import_process as _get_zarr_import_process,
)
from aind_zarr_utils.io.processing import (
    image_atlas_alignment_path_relative_from_processing as image_atlas_alignment_path_relative_from_processing,
)
from aind_zarr_utils.io.transforms import (
    _PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS as _PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS,
)
from aind_zarr_utils.io.transforms import (
    _PIPELINE_TEMPLATE_TRANSFORM_CHAINS as _PIPELINE_TEMPLATE_TRANSFORM_CHAINS,
)
from aind_zarr_utils.io.transforms import (
    _PIPELINE_TEMPLATE_TRANSFORMS as _PIPELINE_TEMPLATE_TRANSFORMS,
)
from aind_zarr_utils.io.transforms import (
    TemplatePaths as TemplatePaths,
)
from aind_zarr_utils.io.transforms import (
    TransformChain as TransformChain,
)
from aind_zarr_utils.io.transforms import (
    _pipeline_image_transforms_local_paths as _pipeline_image_transforms_local_paths,
)
from aind_zarr_utils.io.transforms import (
    _pipeline_point_transforms_local_paths as _pipeline_point_transforms_local_paths,
)
from aind_zarr_utils.io.transforms import (
    pipeline_image_transforms_local_paths as pipeline_image_transforms_local_paths,
)
from aind_zarr_utils.io.transforms import (
    pipeline_point_transforms_local_paths as pipeline_point_transforms_local_paths,
)
from aind_zarr_utils.io.transforms import (
    pipeline_transforms as pipeline_transforms,
)
from aind_zarr_utils.io.transforms import (
    pipeline_transforms_local_paths as pipeline_transforms_local_paths,
)
from aind_zarr_utils.io.zarr import _open_zarr, _zarr_to_scaled
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
    zarr_to_ants,
    zarr_to_sitk,
    zarr_to_sitk_stub,
)

if TYPE_CHECKING:
    from ants.core import ANTsImage  # type: ignore[import-untyped]
    from mypy_boto3_s3 import S3Client
    from ome_zarr.reader import Node  # type: ignore[import-untyped]

T = TypeVar("T", int, float)


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
    _, zarr_import_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
    )

    stub_img, size_ijk = zarr_to_sitk_stub(
        zarr_uri,
        metadata,
        opened_zarr=(image_node, zarr_meta),
    )

    # Convert stub to AnatomicalHeader for domain corrections.
    base_header = AnatomicalHeader.from_sitk(stub_img, size_ijk)

    # Select and apply overlays based on zarr import version and metadata.
    header, applied = _apply_pipeline_overlays_to_header(
        base_header,
        zarr_import_version,
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
    _, zarr_import_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
    )

    if level == 0:
        # Overlays only work at level 0, so in this case we can work with them
        # directly.

        # Convert stub to AnatomicalHeader for domain corrections.
        base_header = AnatomicalHeader.from_sitk(img)

        # Select and apply overlays based on zarr import version and metadata.
        overlays = overlay_selector.select(version=zarr_import_version, meta=metadata)
        corrected_header, _ = apply_overlays(
            base_header,
            overlays,
            metadata,
            multiscale_no or 3,
            zarr_import_version=zarr_import_version,
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
        spacing_scaled = tuple(s * spacing_level_scale for s in corrected_header.spacing)
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
    _, pipeline_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
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
    _, pipeline_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
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
    _, zarr_import_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
    )

    if level == 0:
        # Overlays only work at level 0, so in this case we can work with them
        # directly.

        # Convert stub to AnatomicalHeader for domain corrections.
        base_header = AnatomicalHeader.from_ants(img)

        # Select and apply overlays based on zarr import version and metadata.
        overlays = overlay_selector.select(version=zarr_import_version, meta=metadata)
        corrected_header, _ = apply_overlays(
            base_header,
            overlays,
            metadata,
            multiscale_no or 3,
            zarr_import_version=zarr_import_version,
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
        spacing_rev_scaled = tuple(s * spacing_level_scale for s in reversed(corrected_header.spacing))
        img.set_spacing(spacing_rev_scaled)
        # The origin is also wrong, because it is a different corner of the
        # volume.
        header_origin_code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            corrected_header.direction_tuple()
        )
        header_origin_corner_code = "".join(_OPPOSITE_AXES[d] for d in header_origin_code)
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
    _, pipeline_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
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
    _, pipeline_version, image_node, zarr_meta, multiscale_no = _pipeline_anatomical_check_args(
        zarr_uri, processing_data, opened_zarr=opened_zarr
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
    template_base: str | os.PathLike | None = None,
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
    template_base : str or PathLike, optional
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
    pt_transform_paths_str, pt_transform_is_inverted = pipeline_point_transforms_local_paths(
        zarr_uri,
        processing_data,
        s3_client=s3_client,
        anonymous=anonymous,
        cache_dir=cache_dir,
        template_used=template_used,
        template_base=template_base,
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
    template_base: str | os.PathLike | None = None,
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
    template_base : str or PathLike, optional
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
    template_base: str | os.PathLike | None = None,
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
    template_base : str or PathLike, optional
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
    img_transform_paths_str, img_transform_is_inverted = pipeline_image_transforms_local_paths(
        alignment_zarr_uri,
        processing_data,
        template_used=template_used,
        template_base=template_base,
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
        pipeline_anatomical_points[layer_name] = all_pipeline_points_arr[start_idx:end_idx]
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
            # Convert to numpy ordering `...[::-1]`
            continuous_idx = (pipeline_stub.TransformPhysicalPointToContinuousIndex(point_lps.astype(np.float64)))[::-1]
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
    alignment_zarr_uri, metadata, processing_data = alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
        a_zarr_uri=zarr_uri
    )
    return ccf_to_indices(
        ccf_points,
        alignment_zarr_uri,
        metadata,
        processing_data,
        template_used=template_used,
        **kwargs,
    )


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
        image_sources = get_image_sources(neuroglancer_data, remove_zarr_protocol=True)
        # Get first image source in dict
        a_zarr_uri = next(iter(image_sources.values()), None)
        if a_zarr_uri is None:
            raise ValueError("No image sources found in neuroglancer data")
        zarr_uri, metadata, processing_data = alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            a_zarr_uri=a_zarr_uri
        )
    else:
        zarr_uri, metadata, processing_data = alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            asset_uri=asset_uri
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
    _, _, _, spacing_raw, _ = _zarr_to_scaled(zarr_uri, level=0, opened_zarr=opened_zarr)
    return swc_data_to_indices(
        swc_point_dict,
        spacing_raw,
        swc_point_order=swc_point_order,
        swc_point_units=swc_point_units,
    )


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
    zarr_uri, metadata, processing_data = alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
        asset_uri=asset_uri
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
        - ``template_base`` : str | os.PathLike | None
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
    alignment_zarr_uri, metadata, processing_data = alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
        a_zarr_uri=zarr_uri
    )
    return indices_to_ccf(
        annotation_indices,
        alignment_zarr_uri,
        metadata,
        processing_data,
        **kwargs,
    )
