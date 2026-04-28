"""Tests for ``Points``, ``Space``, and the transform graph in ``points.py``.

The test pyramid here is:

1. Pure-construction tests (no I/O): ``Points`` direct construction,
   ``from_swc``, ``from_neuroglancer`` (the last uses real NG-state input).
2. Path-resolver tests for the BFS that walks the tree.
3. Edge tests that exercise each individual ``(src → dst)`` edge against a
   real on-disk ``real_ome_zarr`` fixture (the same fixture
   ``test_zarr.py`` uses), plus mocks for the ANTs-transform calls when
   the edge crosses the CCF boundary.
4. End-to-end ``Asset.transform`` tests for multi-hop paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from aind_zarr_utils.asset import Asset
from aind_zarr_utils.points import (
    Points,
    Space,
    _path,
    transform_points,
)

# ----------------------------------------------------------- construction ---


class TestPoints:
    """Bare construction; verifies invariants on the dataclass."""

    def test_direct_construction(self) -> None:
        arr = np.array([[1.0, 2.0, 3.0]])
        pts = Points(values={"a": arr}, space=Space.ZARR_INDICES)
        assert pts.space is Space.ZARR_INDICES
        assert pts.values["a"] is arr
        assert pts.descriptions is None

    def test_frozen_assignment_protection(self) -> None:
        pts = Points(values={"a": np.zeros((0, 3))}, space=Space.CCF_MM)
        with pytest.raises(AttributeError):
            pts.space = Space.ZARR_INDICES  # type: ignore[misc]


class TestFromSWC:
    """``Points.from_swc`` performs unit conversion and axis reordering only."""

    def test_micrometer_zyx_default(self) -> None:
        arr_um = np.array([[1000.0, 2000.0, 3000.0]])  # 1, 2, 3 mm in zyx
        pts = Points.from_swc({"n0": arr_um})
        assert pts.space is Space.LS_SCALED_MM
        np.testing.assert_allclose(pts.values["n0"], [[1.0, 2.0, 3.0]])

    def test_single_array_wrapped(self) -> None:
        arr = np.array([[1.0, 2.0, 3.0]])
        pts = Points.from_swc(arr, units="millimeter")
        # Bare-array input wraps in a single-key dict.
        assert list(pts.values.keys()) == ["_"]
        np.testing.assert_allclose(pts.values["_"], arr)

    def test_xyz_axis_order_reorders_to_zyx(self) -> None:
        # Input is in xyz order; expected output reorders to zyx.
        arr_xyz = np.array([[1.0, 2.0, 3.0]])  # x=1, y=2, z=3
        pts = Points.from_swc(arr_xyz, axis_order="xyz", units="millimeter")
        np.testing.assert_allclose(pts.values["_"], [[3.0, 2.0, 1.0]])  # z, y, x

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected"):
            Points.from_swc({"n0": np.array([1.0, 2.0, 3.0])})  # 1-D


class TestFromNeuroglancer:
    """``Points.from_neuroglancer`` wraps the existing reader and types descriptions as lists."""

    def test_basic(self, neuroglancer_test_data: dict) -> None:
        pts = Points.from_neuroglancer(neuroglancer_test_data)
        assert pts.space is Space.ZARR_INDICES
        assert "annotations_layer1" in pts.values
        assert "annotations_layer2" in pts.values
        # Each layer is an (N, 3) z, y, x array.
        assert pts.values["annotations_layer1"].shape == (2, 3)
        assert pts.values["annotations_layer2"].shape == (1, 3)

    def test_descriptions_as_lists(self, neuroglancer_test_data: dict) -> None:
        pts = Points.from_neuroglancer(neuroglancer_test_data)
        assert pts.descriptions is not None
        # Descriptions are list[str | None], not NDArray.
        for desc_list in pts.descriptions.values():
            assert isinstance(desc_list, list)

    def test_omit_descriptions(self, neuroglancer_test_data: dict) -> None:
        pts = Points.from_neuroglancer(neuroglancer_test_data, return_description=False)
        assert pts.descriptions is None


# ----------------------------------------------------------- path solver ---


class TestPath:
    """The BFS path resolver returns the unique path through the tree."""

    def test_identity(self) -> None:
        assert _path(Space.CCF_MM, Space.CCF_MM) == [Space.CCF_MM]

    def test_adjacent(self) -> None:
        assert _path(Space.LS_SCALED_MM, Space.ZARR_INDICES) == [
            Space.LS_SCALED_MM,
            Space.ZARR_INDICES,
        ]

    def test_indices_to_ccf_three_hops(self) -> None:
        assert _path(Space.ZARR_INDICES, Space.CCF_MM) == [
            Space.ZARR_INDICES,
            Space.LS_PIPELINE_ANATOMICAL_MM,
            Space.CCF_MM,
        ]

    def test_scaled_to_ccf_full_chain(self) -> None:
        assert _path(Space.LS_SCALED_MM, Space.CCF_MM) == [
            Space.LS_SCALED_MM,
            Space.ZARR_INDICES,
            Space.LS_PIPELINE_ANATOMICAL_MM,
            Space.CCF_MM,
        ]

    def test_ccf_back_to_indices(self) -> None:
        assert _path(Space.CCF_MM, Space.ZARR_INDICES) == [
            Space.CCF_MM,
            Space.LS_PIPELINE_ANATOMICAL_MM,
            Space.ZARR_INDICES,
        ]

    def test_anatomical_to_pipeline_via_indices(self) -> None:
        # The two anatomical spaces are siblings; their connection is via
        # ZARR_INDICES.
        assert _path(Space.LS_ANATOMICAL_MM, Space.LS_PIPELINE_ANATOMICAL_MM) == [
            Space.LS_ANATOMICAL_MM,
            Space.ZARR_INDICES,
            Space.LS_PIPELINE_ANATOMICAL_MM,
        ]


# ----------------------------------------------------------- edge tests ---


def _make_asset_for_real_zarr(
    real_ome_zarr: str,
    *,
    metadata: dict,
    processing: dict,
) -> Asset:
    """Build an Asset around the real on-disk Zarr fixture, no S3."""
    return Asset(zarr_uri=real_ome_zarr, metadata=metadata, processing=processing)


class TestEdgesIndicesScaled:
    """``ZARR_INDICES ↔ LS_SCALED_MM`` is per-axis multiply / divide."""

    def test_indices_to_scaled_round_trip(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_processing_data: dict,
    ) -> None:
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=mock_processing_data,
        )
        original = Points(
            values={"a": np.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])},
            space=Space.ZARR_INDICES,
        )
        scaled = asset.transform(original, to=Space.LS_SCALED_MM)
        recovered = asset.transform(scaled, to=Space.ZARR_INDICES)
        # The fixture uses (1, 1, 1) mm spacing at level 0, so scaled == indices.
        np.testing.assert_allclose(scaled.values["a"], original.values["a"])
        # Note: the reverse path goes through ``scaled_points_to_indices`` which
        # currently rounds; for unit-spacing the rounded value equals the input.
        np.testing.assert_allclose(recovered.values["a"], original.values["a"])
        assert scaled.space is Space.LS_SCALED_MM
        assert recovered.space is Space.ZARR_INDICES


class TestEdgesIndicesAnatomical:
    """``ZARR_INDICES ↔ LS_ANATOMICAL_MM`` uses the *base* (uncorrected) stub."""

    def test_round_trip_through_base_anatomical(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_processing_data: dict,
    ) -> None:
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=mock_processing_data,
        )
        # Use integer indices so the stub-based reverse mapping returns the
        # exact same values.
        original = Points(
            values={"a": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])},
            space=Space.ZARR_INDICES,
        )
        anat = asset.transform(original, to=Space.LS_ANATOMICAL_MM)
        recovered = asset.transform(anat, to=Space.ZARR_INDICES)

        assert anat.space is Space.LS_ANATOMICAL_MM
        assert recovered.space is Space.ZARR_INDICES
        np.testing.assert_allclose(recovered.values["a"], original.values["a"], atol=1e-10)


class TestEdgesIndicesPipelineAnatomical:
    """``ZARR_INDICES ↔ LS_PIPELINE_ANATOMICAL_MM`` uses the corrected stub."""

    def test_round_trip_through_pipeline_anatomical(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_overlay_selector: object,
    ) -> None:
        # Use a comprehensive processing dict so _build_pipeline_header can
        # find the Image-importing process; mock_overlay_selector neutralises
        # the actual overlays so the corrected stub == base stub here.
        from tests.conftest import create_comprehensive_processing_data

        processing = create_comprehensive_processing_data()
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=processing,
        )
        original = Points(
            values={"a": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])},
            space=Space.ZARR_INDICES,
        )
        pipe = asset.transform(original, to=Space.LS_PIPELINE_ANATOMICAL_MM)
        recovered = asset.transform(pipe, to=Space.ZARR_INDICES)

        assert pipe.space is Space.LS_PIPELINE_ANATOMICAL_MM
        np.testing.assert_allclose(recovered.values["a"], original.values["a"], atol=1e-10)


class TestEdgesAcrossCCF:
    """``LS_PIPELINE_ANATOMICAL_MM ↔ CCF_MM`` invokes ANTs transforms (mocked)."""

    def test_pipeline_anat_to_ccf_per_layer(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_overlay_selector: object,
        mock_ants_transforms: object,
        mock_transform_path_resolution: object,
    ) -> None:
        # mock_ants_transforms returns ``points + 100`` as the "CCF" output.
        from tests.conftest import create_comprehensive_processing_data

        processing = create_comprehensive_processing_data()
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=processing,
        )
        pipe = Points(
            values={"a": np.array([[1.0, 2.0, 3.0]])},
            space=Space.LS_PIPELINE_ANATOMICAL_MM,
        )
        ccf = asset.transform(pipe, to=Space.CCF_MM)
        assert ccf.space is Space.CCF_MM
        np.testing.assert_allclose(ccf.values["a"], [[101.0, 102.0, 103.0]])

    def test_ccf_to_pipeline_anat_batched(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_overlay_selector: object,
        mock_ants_transforms: object,
        mock_transform_path_resolution: object,
    ) -> None:
        from tests.conftest import create_comprehensive_processing_data

        processing = create_comprehensive_processing_data()
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=processing,
        )
        ccf = Points(
            values={
                "a": np.array([[1.0, 2.0, 3.0]]),
                "b": np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            },
            space=Space.CCF_MM,
        )
        pipe = asset.transform(ccf, to=Space.LS_PIPELINE_ANATOMICAL_MM)
        assert pipe.space is Space.LS_PIPELINE_ANATOMICAL_MM
        # Layer keys and per-layer shapes are preserved across the batched call.
        assert set(pipe.values.keys()) == {"a", "b"}
        assert pipe.values["a"].shape == (1, 3)
        assert pipe.values["b"].shape == (2, 3)
        np.testing.assert_allclose(pipe.values["a"], [[101.0, 102.0, 103.0]])
        np.testing.assert_allclose(pipe.values["b"], [[104.0, 105.0, 106.0], [107.0, 108.0, 109.0]])


# ----------------------------------------------------------- end-to-end ---


class TestAssetTransform:
    """``Asset.transform`` walks multi-hop paths and preserves descriptions."""

    def test_identity_transform_returns_same_instance(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_processing_data: dict,
    ) -> None:
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=mock_processing_data,
        )
        pts = Points(values={"a": np.zeros((0, 3))}, space=Space.ZARR_INDICES)
        out = asset.transform(pts, to=Space.ZARR_INDICES)
        assert out is pts

    def test_descriptions_propagate_across_hops(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_processing_data: dict,
    ) -> None:
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=mock_processing_data,
        )
        pts = Points(
            values={"a": np.array([[0.0, 0.0, 0.0]])},
            space=Space.ZARR_INDICES,
            descriptions={"a": ["origin-point"]},
        )
        anat = asset.transform(pts, to=Space.LS_ANATOMICAL_MM)
        assert anat.descriptions == {"a": ["origin-point"]}

    def test_full_chain_scaled_to_ccf(
        self,
        real_ome_zarr: str,
        mock_nd_metadata: dict,
        mock_overlay_selector: object,
        mock_ants_transforms: object,
        mock_transform_path_resolution: object,
    ) -> None:
        """LS_SCALED_MM → CCF_MM walks all three intermediate edges."""
        from tests.conftest import create_comprehensive_processing_data

        processing = create_comprehensive_processing_data()
        asset = _make_asset_for_real_zarr(
            real_ome_zarr,
            metadata=mock_nd_metadata,
            processing=processing,
        )
        pts = Points(
            values={"a": np.array([[0.0, 0.0, 0.0]])},
            space=Space.LS_SCALED_MM,
        )
        ccf = asset.transform(pts, to=Space.CCF_MM)
        assert ccf.space is Space.CCF_MM
        assert ccf.values["a"].shape == (1, 3)

    def test_unsupported_space_value_in_dispatch_table(
        self,
    ) -> None:
        """Sanity: every adjacent pair in the tree has a registered edge."""
        from aind_zarr_utils.points import _ADJ, _EDGES

        for src, neighbours in _ADJ.items():
            for dst in neighbours:
                assert (src, dst) in _EDGES, f"missing edge {src} → {dst}"


# --------------------------------------------------- equivalence smoke ---


def test_neuroglancer_round_trip_uses_transform_method(
    real_ome_zarr: str,
    mock_nd_metadata: dict,
    mock_overlay_selector: object,
    mock_ants_transforms: object,
    mock_transform_path_resolution: object,
    neuroglancer_test_data: dict,
) -> None:
    """Asset.transform(Points.from_neuroglancer(ng), to=CCF_MM) runs end-to-end.

    Doesn't assert numerical equivalence with ``neuroglancer_to_ccf``
    (the auto-metadata helper would attempt extra S3 calls under the
    same mocks); instead verifies structural correctness — same layer
    keys, ``(N, 3)`` outputs, descriptions preserved.
    """
    from tests.conftest import create_comprehensive_processing_data

    processing = create_comprehensive_processing_data()
    asset = Asset(zarr_uri=real_ome_zarr, metadata=mock_nd_metadata, processing=processing)
    pts = Points.from_neuroglancer(neuroglancer_test_data)
    ccf = asset.transform(pts, to=Space.CCF_MM)

    assert ccf.space is Space.CCF_MM
    assert set(ccf.values.keys()) == set(pts.values.keys())
    for k in pts.values:
        assert ccf.values[k].shape == pts.values[k].shape
    # descriptions came along.
    assert ccf.descriptions is not None
    assert set(ccf.descriptions.keys()) == set(pts.values.keys())


def test_transform_points_freestanding_function(
    real_ome_zarr: str,
    mock_nd_metadata: dict,
    mock_processing_data: dict,
) -> None:
    """``transform_points(asset, pts, to=...)`` matches ``asset.transform``."""
    asset = Asset(zarr_uri=real_ome_zarr, metadata=mock_nd_metadata, processing=mock_processing_data)
    pts = Points(values={"a": np.array([[0.0, 1.0, 2.0]])}, space=Space.ZARR_INDICES)
    via_method = asset.transform(pts, to=Space.LS_ANATOMICAL_MM)
    via_function = transform_points(asset, pts, to=Space.LS_ANATOMICAL_MM)
    np.testing.assert_array_equal(via_method.values["a"], via_function.values["a"])
    assert via_method.space == via_function.space


def test_neuroglancer_to_ccf_numerical_equivalence_new_vs_legacy(
    real_ome_zarr: str,
    mock_nd_metadata: dict,
    mock_overlay_selector: object,
    mock_ants_transforms: object,
    mock_transform_path_resolution: object,
    neuroglancer_test_data: dict,
) -> None:
    """The new ``Asset.transform`` path matches the legacy ``neuroglancer_to_ccf``.

    Both paths internally:

    1. Convert NG → ZARR_INDICES.
    2. Map indices → pipeline anatomical via the corrected pipeline stub.
    3. Apply the pipeline's point ANTs chain.

    With identical metadata + processing + overlay-selector, the output
    arrays must be numerically equal. ANTs is mocked to ``points + 100``
    (via ``mock_ants_transforms``) so the result is deterministic and the
    test does not depend on the registration files being available.
    """
    from aind_zarr_utils.pipeline_transformed import neuroglancer_to_ccf
    from tests.conftest import create_comprehensive_processing_data

    processing = create_comprehensive_processing_data()

    # New path: Asset.transform with mocked ANTs / transforms / overlays.
    asset = Asset(zarr_uri=real_ome_zarr, metadata=mock_nd_metadata, processing=processing)
    new_pts = asset.transform(
        Points.from_neuroglancer(neuroglancer_test_data),
        to=Space.CCF_MM,
    )

    # Legacy path: same inputs, same mocks (the conftest fixtures patch
    # both the old and new module homes — see commit C6).
    legacy_dict, _ = neuroglancer_to_ccf(
        neuroglancer_test_data,
        zarr_uri=real_ome_zarr,
        metadata=mock_nd_metadata,
        processing_data=processing,
    )

    # Same layer keys; same per-layer numerical values.
    assert set(new_pts.values.keys()) == set(legacy_dict.keys())
    for key in legacy_dict:
        np.testing.assert_allclose(
            new_pts.values[key],
            legacy_dict[key],
            atol=1e-10,
            err_msg=f"layer {key!r} disagrees between Asset.transform and neuroglancer_to_ccf",
        )
