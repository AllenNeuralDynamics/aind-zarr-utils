"""``Asset`` façade — a single object that owns an opened Zarr plus its metadata.

This module is the user-facing entry point introduced by the API redesign.
An :class:`Asset` instance carries:

* the URI of the *alignment-channel* Zarr,
* the parsed ``metadata.nd.json`` (ND/acquisition metadata),
* the parsed ``processing.json``,
* an opened ``ome-zarr`` ``Node`` (lazily acquired, cached for the lifetime
  of the asset),
* per-asset configuration: overlay selector, S3 client, anonymous flag,
  cache directory, and template selection.

The methods on :class:`Asset` (``image``, ``stub``, ``apply_overlays``) and
the lazy ``transforms`` property delegate to the existing free functions in
:mod:`aind_zarr_utils.zarr`, :mod:`aind_zarr_utils.image`, and
:mod:`aind_zarr_utils.io.transforms`, threading ``opened_zarr=self.opened_zarr``
everywhere so that no S3 resource is read twice in a logical workflow.

Three eager constructors handle the common discovery patterns:

* :meth:`Asset.from_zarr` — pass any acquisition-Zarr URI; the alignment
  channel is auto-resolved from the asset root.
* :meth:`Asset.from_root` — pass the asset root directly.
* :meth:`Asset.from_neuroglancer` — pull a Zarr URI from the
  ``image_sources`` of a Neuroglancer state and resolve from there.

The bare constructor ``Asset(zarr_uri, metadata, processing)`` is the
no-I/O path: pass already-loaded values explicitly. The Zarr is still
opened lazily on first access.

Notes
-----
:class:`Asset` is **not thread-safe**. ``ome-zarr``'s ``Reader`` holds an
``fsspec`` filesystem object that is fork-unsafe; do not pass an opened
:class:`Asset` across processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import SimpleITK as sitk

from aind_zarr_utils.domain.selector import OverlaySelector, get_selector
from aind_zarr_utils.formats.neuroglancer import get_image_sources
from aind_zarr_utils.image import apply_pipeline_overlays
from aind_zarr_utils.io.paths import (
    alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike,
)
from aind_zarr_utils.io.transforms import pipeline_transforms_local_paths
from aind_zarr_utils.io.zarr import _open_zarr
from aind_zarr_utils.origin import Origin
from aind_zarr_utils.pipeline_transformed import (
    mimic_pipeline_zarr_to_anatomical_stub,
)
from aind_zarr_utils.zarr import zarr_to_ants, zarr_to_sitk, zarr_to_sitk_stub

if TYPE_CHECKING:
    import os

    from ants.core import ANTsImage  # type: ignore[import-untyped]
    from mypy_boto3_s3 import S3Client
    from ome_zarr.reader import Node  # type: ignore[import-untyped]


_DEFAULT_TEMPLATE = "SmartSPIM-template_2024-05-16_11-26-14"


@dataclass(frozen=True, slots=True)
class TransformPaths:
    """Local filesystem paths for the pipeline's ANTs transform chains.

    Both the *image* (forward) and *point* (reverse) chain files are
    materialised together so that round-tripping points through CCF reuses
    the same downloaded files.

    Attributes
    ----------
    point_paths : list[str]
        Paths to the transform files in the order ANTs expects when warping
        *points* (CCF → individual: image-direction inverse).
    point_invert : list[bool]
        Whether each entry in ``point_paths`` should be inverted by ANTs.
    image_paths : list[str]
        Paths to the transform files in the order ANTs expects when warping
        *images* (individual → CCF: forward direction).
    image_invert : list[bool]
        Whether each entry in ``image_paths`` should be inverted by ANTs.
    """

    point_paths: list[str]
    point_invert: list[bool]
    image_paths: list[str]
    image_invert: list[bool]


@dataclass
class Asset:
    """A SmartSPIM acquisition with eager metadata and lazy Zarr/transform I/O.

    Parameters
    ----------
    zarr_uri : str
        URI of the alignment-channel Zarr. Use the classmethods
        :meth:`from_zarr` / :meth:`from_root` / :meth:`from_neuroglancer`
        to discover this from a different starting point.
    metadata : dict
        Parsed ``metadata.nd.json``.
    processing : dict
        Parsed ``processing.json``.
    overlay_selector : OverlaySelector, optional
        Pipeline-overlay selector. Defaults to the cached singleton
        returned by :func:`~aind_zarr_utils.domain.selector.get_selector`;
        the factory call avoids the import-time binding trap of using
        ``= get_selector()`` as a default-arg expression.
    s3_client, anonymous, cache_dir : optional
        S3 access knobs forwarded to
        :func:`~aind_zarr_utils.io.transforms.pipeline_transforms_local_paths`.
    template_used : str
        Key naming the template→CCF transform set to use. Defaults to
        ``"SmartSPIM-template_2024-05-16_11-26-14"``.
    template_base : str or os.PathLike, optional
        Override base path for the template transforms (e.g. a local
        mirror). If ``None``, the canonical S3 location is used.
    """

    zarr_uri: str
    metadata: dict[str, Any]
    processing: dict[str, Any]
    overlay_selector: OverlaySelector = field(default_factory=get_selector)
    s3_client: S3Client | None = None
    anonymous: bool = True
    cache_dir: str | os.PathLike | None = None
    template_used: str = _DEFAULT_TEMPLATE
    template_base: str | os.PathLike | None = None

    # Lazily-populated cache of the opened Zarr ``(Node, metadata)`` tuple.
    # Set via the :attr:`opened_zarr` property or by the eager ``from_*``
    # classmethods, which pre-open the Zarr to fail fast on bad URIs.
    _opened_zarr: tuple[Node, dict] | None = field(default=None, init=False, repr=False, compare=False)

    # Lazily-populated cache of the resolved transform-chain local paths.
    _transforms: TransformPaths | None = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------ ctors

    @classmethod
    def from_zarr(
        cls,
        zarr_uri: str,
        *,
        anonymous: bool = True,
        s3_client: S3Client | None = None,
        cache_dir: str | os.PathLike | None = None,
        overlay_selector: OverlaySelector | None = None,
        template_used: str = _DEFAULT_TEMPLATE,
        template_base: str | os.PathLike | None = None,
    ) -> Asset:
        """Discover the asset's metadata + alignment Zarr from any acquisition Zarr URI.

        Walks up two directory levels from ``zarr_uri`` to find the asset
        root, loads ``metadata.nd.json`` and ``processing.json``, infers the
        alignment channel from the processing metadata, and pre-opens the
        alignment Zarr (eager). The ``zarr_uri`` you pass need not itself
        be the alignment channel.
        """
        z_uri, metadata, processing = alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            a_zarr_uri=zarr_uri,
            anonymous=anonymous,
            s3_client=s3_client,
        )
        return cls._eagerly(
            z_uri,
            metadata,
            processing,
            overlay_selector=overlay_selector,
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
            template_used=template_used,
            template_base=template_base,
        )

    @classmethod
    def from_root(
        cls,
        asset_uri: str,
        *,
        anonymous: bool = True,
        s3_client: S3Client | None = None,
        cache_dir: str | os.PathLike | None = None,
        overlay_selector: OverlaySelector | None = None,
        template_used: str = _DEFAULT_TEMPLATE,
        template_base: str | os.PathLike | None = None,
    ) -> Asset:
        """Discover the asset's metadata + alignment Zarr from the asset root URI."""
        z_uri, metadata, processing = alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            asset_uri=asset_uri,
            anonymous=anonymous,
            s3_client=s3_client,
        )
        return cls._eagerly(
            z_uri,
            metadata,
            processing,
            overlay_selector=overlay_selector,
            s3_client=s3_client,
            anonymous=anonymous,
            cache_dir=cache_dir,
            template_used=template_used,
            template_base=template_base,
        )

    @classmethod
    def from_neuroglancer(
        cls,
        ng_state: dict[str, Any],
        *,
        asset_uri: str | None = None,
        anonymous: bool = True,
        s3_client: S3Client | None = None,
        cache_dir: str | os.PathLike | None = None,
        overlay_selector: OverlaySelector | None = None,
        template_used: str = _DEFAULT_TEMPLATE,
        template_base: str | os.PathLike | None = None,
    ) -> Asset:
        """Discover the asset from a Neuroglancer state.

        If ``asset_uri`` is given, behaves like :meth:`from_root`. Otherwise
        the first ``image_sources`` URL in ``ng_state`` is used as the
        starting point (passed to :meth:`from_zarr`).
        """
        if asset_uri is not None:
            return cls.from_root(
                asset_uri,
                anonymous=anonymous,
                s3_client=s3_client,
                cache_dir=cache_dir,
                overlay_selector=overlay_selector,
                template_used=template_used,
                template_base=template_base,
            )
        sources = get_image_sources(ng_state, remove_zarr_protocol=True)
        a_zarr_uri = next(iter(sources.values()), None)
        if a_zarr_uri is None:
            raise ValueError("No image sources found in neuroglancer data")
        return cls.from_zarr(
            a_zarr_uri,
            anonymous=anonymous,
            s3_client=s3_client,
            cache_dir=cache_dir,
            overlay_selector=overlay_selector,
            template_used=template_used,
            template_base=template_base,
        )

    @classmethod
    def _eagerly(
        cls,
        zarr_uri: str,
        metadata: dict[str, Any],
        processing: dict[str, Any],
        *,
        overlay_selector: OverlaySelector | None,
        s3_client: S3Client | None,
        anonymous: bool,
        cache_dir: str | os.PathLike | None,
        template_used: str,
        template_base: str | os.PathLike | None,
    ) -> Asset:
        """Internal: build an :class:`Asset` and pre-open its Zarr."""
        kwargs: dict[str, Any] = {
            "zarr_uri": zarr_uri,
            "metadata": metadata,
            "processing": processing,
            "s3_client": s3_client,
            "anonymous": anonymous,
            "cache_dir": cache_dir,
            "template_used": template_used,
            "template_base": template_base,
        }
        if overlay_selector is not None:
            kwargs["overlay_selector"] = overlay_selector
        asset = cls(**kwargs)
        asset._opened_zarr = _open_zarr(zarr_uri)
        return asset

    # --------------------------------------------------------------- accessors

    @property
    def opened_zarr(self) -> tuple[Node, dict]:
        """The opened ome-zarr ``(Node, metadata)`` tuple, opened on first access."""
        if self._opened_zarr is None:
            self._opened_zarr = _open_zarr(self.zarr_uri)
        return self._opened_zarr

    @property
    def transforms(self) -> TransformPaths:
        """The pipeline's ANTs transform chains as local file paths.

        First access downloads (or locates in the ``aind_s3_cache`` cache)
        all required transform files; the result is cached for the lifetime
        of the asset.
        """
        if self._transforms is None:
            pt_paths, pt_invert, img_paths, img_invert = pipeline_transforms_local_paths(
                self.zarr_uri,
                self.processing,
                s3_client=self.s3_client,
                anonymous=self.anonymous,
                cache_dir=self.cache_dir,
                template_used=self.template_used,
                template_base=self.template_base,
            )
            self._transforms = TransformPaths(
                point_paths=pt_paths,
                point_invert=pt_invert,
                image_paths=img_paths,
                image_invert=img_invert,
            )
        return self._transforms

    # ----------------------------------------------------------------- builders

    def image(
        self,
        *,
        level: int = 3,
        library: Literal["sitk", "ants"] = "sitk",
        pipeline: bool = False,
        origin: Origin | None = None,
        scale_unit: str = "millimeter",
    ) -> sitk.Image | ANTsImage:
        """Build a full image from the asset's Zarr at the given pyramid level.

        Parameters
        ----------
        level : int, optional
            Resolution level (0 = full resolution; higher = coarser).
            Default ``3``.
        library : {"sitk", "ants"}, optional
            Backend. Default ``"sitk"``.
        pipeline : bool, optional
            If ``True``, apply pipeline overlay corrections in-place after
            constructing the image. Default ``False``.
        origin : Origin, optional
            Origin specification. Defaults to :meth:`Origin.default`.
            Only consulted when ``pipeline=False`` (the pipeline path
            uses the corrected header's origin).
        scale_unit : str, optional
            Output spacing unit. Default ``"millimeter"``.

        Returns
        -------
        sitk.Image or ants.core.ANTsImage
            The constructed image, with pipeline corrections applied if
            ``pipeline=True``.
        """
        legacy_origin = (origin or Origin.default())._legacy_kwargs()
        if library == "sitk":
            img: sitk.Image | ANTsImage = zarr_to_sitk(
                self.zarr_uri,
                self.metadata,
                level=level,
                scale_unit=scale_unit,
                opened_zarr=self.opened_zarr,
                **legacy_origin,
            )
        elif library == "ants":
            img = zarr_to_ants(
                self.zarr_uri,
                self.metadata,
                level=level,
                scale_unit=scale_unit,
                opened_zarr=self.opened_zarr,
                **legacy_origin,
            )
        else:
            raise ValueError(f"Unsupported library: {library!r} (expected 'sitk' or 'ants')")
        if pipeline:
            self.apply_overlays(img, level=level)
        return img

    def stub(
        self,
        *,
        pipeline: bool = False,
        origin: Origin | None = None,
        scale_unit: str = "millimeter",
        level: int = 0,
    ) -> tuple[sitk.Image, tuple[int, int, int]]:
        """Build a header-only SimpleITK stub for the asset's Zarr.

        Returns
        -------
        stub : sitk.Image
            A 1×1×1 SimpleITK image whose spatial header matches what a
            full-resolution image would have, optionally with pipeline
            corrections applied.
        size_ijk : tuple of three ints
            The level-0 native voxel dimensions of the acquisition, in
            SimpleITK index order.

        Notes
        -----
        Only SimpleITK is currently supported for stubs (mirrors the
        existing ``zarr_to_sitk_stub`` API). When ``pipeline=True``, the
        ``origin`` argument is ignored — the corrected header's origin is
        used instead.
        """
        if pipeline:
            return mimic_pipeline_zarr_to_anatomical_stub(
                self.zarr_uri,
                self.metadata,
                self.processing,
                overlay_selector=self.overlay_selector,
                opened_zarr=self.opened_zarr,
            )
        legacy_origin = (origin or Origin.default())._legacy_kwargs()
        return zarr_to_sitk_stub(
            self.zarr_uri,
            self.metadata,
            level=level,
            scale_unit=scale_unit,
            opened_zarr=self.opened_zarr,
            **legacy_origin,
        )

    def apply_overlays(
        self,
        img: sitk.Image | ANTsImage,
        *,
        level: int = 3,
    ) -> None:
        """Apply pipeline overlay corrections in-place to ``img``.

        Type-dispatches on ``img`` to handle SimpleITK and ANTs. See
        :func:`aind_zarr_utils.image.apply_pipeline_overlays` for the
        per-level math; this method threads the asset's
        ``overlay_selector`` and cached ``opened_zarr`` through.
        """
        apply_pipeline_overlays(
            img,
            self.zarr_uri,
            self.processing,
            self.metadata,
            level=level,
            overlay_selector=self.overlay_selector,
            opened_zarr=self.opened_zarr,
        )
