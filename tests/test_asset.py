"""Smoke tests for the ``Asset`` façade and ``Origin`` value type."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from aind_zarr_utils.asset import Asset, TransformPaths
from aind_zarr_utils.origin import Origin

# ---------------------------------------------------------------- Origin ---


class TestOrigin:
    """Origin construction and translation to legacy kwargs."""

    def test_default(self) -> None:
        """Origin.default() encodes all-None — i.e. (0,0,0)."""
        o = Origin.default()
        assert o.point is None
        assert o.corner_code is None
        assert o.corner_lps is None
        assert o._legacy_kwargs() == {
            "set_origin": None,
            "set_corner": None,
            "set_corner_lps": None,
        }

    def test_at(self) -> None:
        """Origin.at(point) sets only the point field."""
        o = Origin.at((1.0, 2.0, 3.0))
        assert o.point == (1.0, 2.0, 3.0)
        assert o.corner_code is None
        assert o.corner_lps is None
        assert o._legacy_kwargs() == {
            "set_origin": (1.0, 2.0, 3.0),
            "set_corner": None,
            "set_corner_lps": None,
        }

    def test_at_corner(self) -> None:
        """Origin.at_corner(code, lps) sets the corner pair, not the point."""
        o = Origin.at_corner("RAI", (0.0, 0.0, 0.0))
        assert o.point is None
        assert o.corner_code == "RAI"
        assert o.corner_lps == (0.0, 0.0, 0.0)
        assert o._legacy_kwargs() == {
            "set_origin": None,
            "set_corner": "RAI",
            "set_corner_lps": (0.0, 0.0, 0.0),
        }

    def test_frozen(self) -> None:
        """Origin instances are immutable."""
        o = Origin.default()
        with pytest.raises(AttributeError):
            o.point = (1.0, 2.0, 3.0)  # type: ignore[misc]


# ---------------------------------------------------------------- Asset ---


def _make_explicit_asset(zarr_uri: str = "s3://bucket/data/asset/image.zarr/") -> Asset:
    metadata: dict[str, Any] = {"acquisition": {"axes": []}}
    processing: dict[str, Any] = {
        "processing_pipeline": {
            "pipeline_version": "3.1.0",
            "data_processes": [],
        }
    }
    return Asset(zarr_uri=zarr_uri, metadata=metadata, processing=processing)


class TestAssetExplicit:
    """The explicit ``Asset(...)`` constructor performs no I/O."""

    def test_explicit_construction_no_io(self) -> None:
        """No S3 / Zarr calls happen for the bare constructor."""
        with (
            patch("aind_zarr_utils.asset._open_zarr") as mock_open,
            patch("aind_zarr_utils.asset.alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike") as mock_align,
            patch("aind_zarr_utils.asset.pipeline_transforms_local_paths") as mock_xforms,
        ):
            asset = _make_explicit_asset()

            assert asset.zarr_uri == "s3://bucket/data/asset/image.zarr/"
            mock_open.assert_not_called()
            mock_align.assert_not_called()
            mock_xforms.assert_not_called()

    def test_default_overlay_selector_is_singleton(self) -> None:
        """Default overlay_selector is the cached ``get_selector()`` singleton."""
        from aind_zarr_utils.domain.selector import get_selector

        a = _make_explicit_asset()
        b = _make_explicit_asset()
        assert a.overlay_selector is get_selector()
        # Two assets share the same singleton (because get_selector is lru_cached).
        assert a.overlay_selector is b.overlay_selector

    def test_opened_zarr_lazy_and_cached(self) -> None:
        """``opened_zarr`` opens the Zarr on first access and caches the result."""
        sentinel = ("node", {"meta": True})
        with patch("aind_zarr_utils.asset._open_zarr", return_value=sentinel) as mock_open:
            asset = _make_explicit_asset()
            mock_open.assert_not_called()

            first = asset.opened_zarr
            second = asset.opened_zarr

            assert first is sentinel
            assert second is sentinel
            mock_open.assert_called_once_with(asset.zarr_uri)


class TestAssetEagerConstructors:
    """The ``from_*`` classmethods load metadata and pre-open the Zarr."""

    def test_from_root_eager_open(self) -> None:
        """from_root pre-opens the alignment Zarr after resolving metadata."""
        zarr_uri = "s3://bucket/data/asset/image_tile_fusing/OMEZarr/Ex.zarr"
        metadata: dict[str, Any] = {"acquisition": {"axes": []}}
        processing: dict[str, Any] = {"processing_pipeline": {"pipeline_version": "3.1.0", "data_processes": []}}
        sentinel = ("node", {"meta": True})

        with (
            patch(
                "aind_zarr_utils.asset.alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike",
                return_value=(zarr_uri, metadata, processing),
            ) as mock_align,
            patch("aind_zarr_utils.asset._open_zarr", return_value=sentinel) as mock_open,
        ):
            asset = Asset.from_root("s3://bucket/data/asset/")

            mock_align.assert_called_once()
            mock_open.assert_called_once_with(zarr_uri)
            assert asset.zarr_uri == zarr_uri
            assert asset.metadata is metadata
            assert asset.processing is processing
            # Subsequent .opened_zarr access does NOT re-open.
            assert asset.opened_zarr is sentinel
            mock_open.assert_called_once()

    def test_from_neuroglancer_uses_first_image_source(self) -> None:
        """from_neuroglancer pulls the first ``image_sources`` URL and delegates to from_zarr."""
        ng_state = {
            "layers": [
                {"name": "ch0", "type": "image", "source": "zarr://s3://bucket/x.zarr"},
            ]
        }
        zarr_uri = "s3://bucket/x.zarr"
        metadata: dict[str, Any] = {}
        processing: dict[str, Any] = {}
        sentinel = ("n", {})

        with (
            patch(
                "aind_zarr_utils.asset.alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike",
                return_value=(zarr_uri, metadata, processing),
            ) as mock_align,
            patch("aind_zarr_utils.asset._open_zarr", return_value=sentinel),
        ):
            asset = Asset.from_neuroglancer(ng_state)

            mock_align.assert_called_once()
            kwargs = mock_align.call_args.kwargs
            assert kwargs.get("a_zarr_uri") == "s3://bucket/x.zarr"
            assert asset.zarr_uri == zarr_uri

    def test_from_neuroglancer_no_image_sources_raises(self) -> None:
        """from_neuroglancer raises when the NG state has no image layers."""
        with pytest.raises(ValueError, match="No image sources"):
            Asset.from_neuroglancer({"layers": []})


class TestAssetCachesAcrossCalls:
    """Asset operations reuse the cached opened Zarr — no duplicate S3 reads."""

    def test_image_and_stub_share_opened_zarr(self) -> None:
        """A single Asset.image() + Asset.stub() flow opens the Zarr exactly once."""
        sentinel = ("node", {"meta": True})

        with (
            patch("aind_zarr_utils.asset._open_zarr", return_value=sentinel) as mock_open,
            patch("aind_zarr_utils.asset.zarr_to_sitk", return_value="sitk-img") as mock_sitk,
            patch(
                "aind_zarr_utils.asset.zarr_to_sitk_stub",
                return_value=("sitk-stub", (10, 20, 30)),
            ) as mock_stub,
        ):
            asset = _make_explicit_asset()
            asset.image(level=3, library="sitk")
            asset.stub()

            # _open_zarr is called exactly once across both methods.
            mock_open.assert_called_once_with(asset.zarr_uri)

            # Both delegated functions received the cached opened_zarr.
            assert mock_sitk.call_args.kwargs.get("opened_zarr") is sentinel
            assert mock_stub.call_args.kwargs.get("opened_zarr") is sentinel

    def test_full_workflow_opens_once_resolves_transforms_once(self) -> None:
        """Across image() + stub() + transform(), each S3 resource is acquired
        at most once.

        This is the cache invariant the redesign was meant to enforce: a
        full neuroglancer-to-CCF workflow on one Asset opens the Zarr
        exactly once and resolves the transform-chain paths exactly once,
        regardless of how many Asset methods participate.
        """
        import numpy as np

        from aind_zarr_utils.points import Points, Space

        sentinel_zarr = ("node", {"meta": True})

        with (
            patch("aind_zarr_utils.asset._open_zarr", return_value=sentinel_zarr) as mock_open,
            patch(
                "aind_zarr_utils.asset.pipeline_transforms_local_paths",
                return_value=(["pt"], [True], ["img"], [False]),
            ) as mock_xforms,
            patch("aind_zarr_utils.asset.zarr_to_sitk", return_value="sitk-img"),
            patch(
                "aind_zarr_utils.asset.zarr_to_sitk_stub",
                return_value=("sitk-stub", (10, 20, 30)),
            ),
            patch(
                "aind_zarr_utils.asset.mimic_pipeline_zarr_to_anatomical_stub",
                return_value=("pipeline-stub", (10, 20, 30)),
            ),
            # The transform graph's pipeline-anat edge needs an actual sitk
            # stub object; bypass it by mocking the pipeline path's
            # annotation_indices_to_anatomical and the ANTs call.
            patch(
                "aind_zarr_utils.points.annotation_indices_to_anatomical",
                return_value={"a": np.array([[1.0, 2.0, 3.0]])},
            ),
            patch(
                "aind_zarr_utils.points.apply_ants_transforms_to_point_arr",
                side_effect=lambda pts, **_: pts + 1,
            ),
        ):
            asset = _make_explicit_asset()

            asset.image(level=3, library="sitk")
            asset.stub()
            pts = Points(values={"a": np.array([[0.0, 0.0, 0.0]])}, space=Space.ZARR_INDICES)
            asset.transform(pts, to=Space.CCF_MM)

            # The full chain opens the Zarr exactly once across all three methods.
            mock_open.assert_called_once_with(asset.zarr_uri)
            # Transform-chain paths are resolved exactly once even though
            # the transform crossed CCF.
            mock_xforms.assert_called_once()

    def test_transforms_property_caches(self) -> None:
        """The transforms property is computed at most once per asset."""
        with patch(
            "aind_zarr_utils.asset.pipeline_transforms_local_paths",
            return_value=(["pt"], [True], ["img"], [False]),
        ) as mock_xforms:
            asset = _make_explicit_asset()
            first = asset.transforms
            second = asset.transforms

            assert isinstance(first, TransformPaths)
            assert first is second
            mock_xforms.assert_called_once()


def test_origin_default_used_when_omitted_on_image() -> None:
    """Asset.image(origin=None) translates to the default legacy triple."""
    with (
        patch("aind_zarr_utils.asset._open_zarr", return_value=("n", {})),
        patch("aind_zarr_utils.asset.zarr_to_sitk", return_value="img") as mock_sitk,
    ):
        asset = _make_explicit_asset()
        asset.image()
        kwargs = mock_sitk.call_args.kwargs
        assert kwargs.get("set_origin") is None
        assert kwargs.get("set_corner") is None
        assert kwargs.get("set_corner_lps") is None


def test_origin_at_corner_propagates_to_zarr_to_sitk() -> None:
    """Asset.image(origin=Origin.at_corner(...)) forwards the corner kwargs."""
    with (
        patch("aind_zarr_utils.asset._open_zarr", return_value=("n", {})),
        patch("aind_zarr_utils.asset.zarr_to_sitk", return_value="img") as mock_sitk,
    ):
        asset = _make_explicit_asset()
        asset.image(origin=Origin.at_corner("RAI", (0.0, 0.0, 0.0)))
        kwargs = mock_sitk.call_args.kwargs
        assert kwargs.get("set_origin") is None
        assert kwargs.get("set_corner") == "RAI"
        assert kwargs.get("set_corner_lps") == (0.0, 0.0, 0.0)
