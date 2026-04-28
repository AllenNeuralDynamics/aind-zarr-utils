"""ANTs transform-chain plumbing for SmartSPIM ↔ CCF mappings.

A SmartSPIM acquisition is registered to CCF in two stages:

* *individual → template* — produced per-acquisition by the SmartSPIM
  pipeline; located alongside ``processing.json`` under
  ``image_atlas_alignment/<channel>/``.
* *template → CCF* — a fixed set of transforms shared across all SmartSPIM
  data, hosted on ``s3://aind-open-data``.

This module describes those chains as :class:`TransformChain` /
:class:`TemplatePaths` value objects, and resolves the constituent ANTs
``.nii.gz`` / ``.mat`` files to local paths via the ``aind_s3_cache`` cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aind_s3_cache.s3_cache import (
    get_local_path_for_resource,
)
from aind_s3_cache.uri_utils import as_pathlike, as_string, join_any

from aind_zarr_utils.io.paths import _asset_from_zarr_pathlike
from aind_zarr_utils.io.processing import (
    image_atlas_alignment_path_relative_from_processing,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


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
        chain=_PIPELINE_TEMPLATE_TRANSFORM_CHAINS["SmartSPIM-template_2024-05-16_11-26-14"],
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


def pipeline_transforms(
    zarr_uri: str,
    processing_data: dict[str, Any],
    *,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | os.PathLike | None = None,
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
    template_base : str or PathLike, optional
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
    alignment_rel_path = image_atlas_alignment_path_relative_from_processing(processing_data)
    if alignment_rel_path is None:
        raise ValueError("Could not determine image atlas alignment path from processing data")
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
            str(template_base),
            _PIPELINE_TEMPLATE_TRANSFORM_CHAINS[template_used],
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
    img_transforms_individual_is_inverted = individual_ants_paths.chain.forward_chain_invert
    img_transforms_template_is_inverted = template_ants_paths.chain.forward_chain_invert

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

    img_transform_paths = img_transforms_template_paths + img_transforms_individual_paths
    img_transform_paths_str = [str(p) for p in img_transform_paths]
    img_transform_is_inverted = img_transforms_template_is_inverted + img_transforms_individual_is_inverted
    return img_transform_paths_str, img_transform_is_inverted


def pipeline_image_transforms_local_paths(
    zarr_uri: str,
    processing_data: dict[str, Any],
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = True,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | os.PathLike | None = None,
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
    template_base : str or PathLike, optional
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
    pt_transforms_individual_is_inverted = individual_ants_paths.chain.reverse_chain_invert
    pt_transforms_template_is_inverted = template_ants_paths.chain.reverse_chain_invert

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

    pt_transform_paths = pt_transforms_individual_paths + pt_transforms_template_paths
    pt_transform_paths_str = [str(p) for p in pt_transform_paths]
    pt_transform_is_inverted = pt_transforms_individual_is_inverted + pt_transforms_template_is_inverted
    return pt_transform_paths_str, pt_transform_is_inverted


def pipeline_point_transforms_local_paths(
    zarr_uri: str,
    processing_data: dict[str, Any],
    *,
    s3_client: S3Client | None = None,
    anonymous: bool = True,
    cache_dir: str | os.PathLike | None = None,
    template_used: str = "SmartSPIM-template_2024-05-16_11-26-14",
    template_base: str | os.PathLike | None = None,
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
    template_base : str or PathLike, optional
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
    template_base: str | os.PathLike | None = None,
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
    template_base : str or PathLike, optional
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
    pt_transform_paths_str, pt_transform_is_inverted = _pipeline_point_transforms_local_paths(
        individual_ants_paths,
        template_ants_paths,
        s3_client=s3_client,
        anonymous=anonymous,
        cache_dir=cache_dir,
    )
    img_transform_paths_str, img_transform_is_inverted = _pipeline_image_transforms_local_paths(
        individual_ants_paths,
        template_ants_paths,
        s3_client=s3_client,
        anonymous=anonymous,
        cache_dir=cache_dir,
    )
    return (
        pt_transform_paths_str,
        pt_transform_is_inverted,
        img_transform_paths_str,
        img_transform_is_inverted,
    )
