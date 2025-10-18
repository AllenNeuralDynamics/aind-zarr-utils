"""Tests for pipeline_domain_selector module."""

from datetime import date, datetime

import numpy as np
import pytest
from aind_anatomical_utils.anatomical_volume import AnatomicalHeader
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from aind_zarr_utils import pipeline_domain_selector as pds


class TestOverlays:
    """Test overlay classes."""

    def test_spacing_scale_overlay(self):
        """Test SpacingScaleOverlay."""
        overlay = pds.SpacingScaleOverlay(scale=2.0)

        header = AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1.0, 1.5, 2.0),
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )

        result = overlay(header, {}, 0)

        assert result.spacing == (2.0, 3.0, 4.0)
        assert result.origin == header.origin  # Unchanged
        assert np.array_equal(result.direction, header.direction)

    def test_flip_index_axes_overlay(self):
        """Test FlipIndexAxesOverlay."""
        overlay = pds.FlipIndexAxesOverlay(flip_i=True, flip_k=True)

        direction = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        header = AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=direction,
            size_ijk=(10, 10, 10),
        )

        result = overlay(header, {}, 0)

        expected_direction = np.array(
            [
                [-1, 0, 0],  # Flipped i column
                [0, 1, 0],  # Unchanged j column
                [0, 0, -1],  # Flipped k column
            ]
        )

        assert np.array_equal(result.direction, expected_direction)

    def test_permute_index_axes_overlay(self):
        """Test PermuteIndexAxesOverlay."""
        overlay = pds.PermuteIndexAxesOverlay(order=(2, 0, 1))  # k, i, j

        direction = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        header = AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1.0, 2.0, 3.0),
            direction=direction,
            size_ijk=(10, 20, 30),
        )

        result = overlay(header, {}, 0)

        # Check reordered spacing (k, i, j order)
        assert result.spacing == (3.0, 1.0, 2.0)

        # Check reordered size
        assert result.size_ijk == (30, 10, 20)

        # Check reordered direction columns
        expected_direction = direction[:, [2, 0, 1]]
        assert np.array_equal(result.direction, expected_direction)

    def test_set_lps_world_spacing_overlay(self):
        """Test SetLpsWorldSpacingOverlay."""
        overlay = pds.SetLpsWorldSpacingOverlay(lps_spacing_mm=(0.5, 1.0, 1.5))

        # Identity direction matrix
        header = AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(999, 999, 999),  # Will be replaced
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )

        result = overlay(header, {}, multiscale_no=0)

        # Should set spacing to LPS world values (since direction is identity)
        assert result.spacing == (0.5, 1.0, 1.5)

    def test_set_lps_world_spacing_overlay_with_multiscale(self):
        """Test SetLpsWorldSpacingOverlay with multiscale downsampling."""
        overlay = pds.SetLpsWorldSpacingOverlay(lps_spacing_mm=(1.0, 1.0, 2.0))

        header = AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(999, 999, 999),
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )

        # multiscale_no=3 means (1/2)^3 = 1/8 scaling
        result = overlay(header, {}, multiscale_no=3)

        expected_scaling = (1 / 2) ** 3  # 0.125
        expected_spacing = (
            1.0 * expected_scaling,
            1.0 * expected_scaling,
            2.0 * expected_scaling,
        )

        assert result.spacing == expected_spacing

    def test_force_corner_anchor_overlay(self, monkeypatch):
        """Test ForceCornerAnchorOverlay."""

        # Mock the fix_corner_compute_origin function
        def mock_compute_origin(
            size,
            spacing,
            direction,
            target_point,
            corner_code,
            target_frame,
            use_outer_box,
        ):
            return (10.0, 20.0, 30.0), None, None

        monkeypatch.setattr(
            "aind_zarr_utils.pipeline_domain_selector.fix_corner_compute_origin",
            mock_compute_origin,
        )

        overlay = pds.ForceCornerAnchorOverlay(
            corner_code="RAS", target_point_labeled=(5.0, 5.0, 5.0)
        )

        header = AnatomicalHeader(
            origin=(0, 0, 0),  # Will be changed
            spacing=(1, 1, 1),
            direction=np.eye(3),
            size_ijk=(100, 100, 100),
        )

        result = overlay(header, {}, 0)

        assert result.origin == (10.0, 20.0, 30.0)  # From mock
        assert result.spacing == header.spacing  # Unchanged
        assert np.array_equal(result.direction, header.direction)


class TestOverlaySelector:
    """Test OverlaySelector functionality."""

    def test_overlay_selector_empty(self):
        """Test empty OverlaySelector."""
        selector = pds.OverlaySelector()
        overlays = selector.select(version="1.0.0", meta={})
        assert overlays == []

    def test_overlay_selector_version_matching(self):
        """Test version-based overlay selection."""
        rule = pds.OverlayRule(
            name="test_rule",
            spec=SpecifierSet(">=1.0.0,<2.0.0"),
            factory=lambda meta: pds.SpacingScaleOverlay(scale=2.0),
        )

        selector = pds.OverlaySelector(rules=(rule,))

        # Should match
        overlays = selector.select(version="1.5.0", meta={})
        assert len(overlays) == 1
        assert isinstance(overlays[0], pds.SpacingScaleOverlay)

        # Should not match
        overlays = selector.select(version="2.1.0", meta={})
        assert len(overlays) == 0

    def test_overlay_selector_date_filtering(self):
        """Test date-based overlay filtering."""
        rule = pds.OverlayRule(
            name="date_rule",
            spec=SpecifierSet(">=1.0.0"),
            factory=lambda meta: pds.SpacingScaleOverlay(scale=1.5),
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )

        selector = pds.OverlaySelector(rules=(rule,))

        # Should match (date in range)
        meta_in_range = {"acq_date": "2024-06-15"}
        overlays = selector.select(version="1.0.0", meta=meta_in_range)
        assert len(overlays) == 1

        # Should not match (date before range)
        meta_before = {"acq_date": "2023-12-31"}
        overlays = selector.select(version="1.0.0", meta=meta_before)
        assert len(overlays) == 0

        # Should not match (date after range)
        meta_after = {"acq_date": "2025-01-01"}
        overlays = selector.select(version="1.0.0", meta=meta_after)
        assert len(overlays) == 0

    def test_overlay_selector_predicate(self):
        """Test predicate-based overlay filtering."""
        rule = pds.OverlayRule(
            name="predicate_rule",
            spec=SpecifierSet(">=0.0.0"),
            factory=lambda meta: pds.SpacingScaleOverlay(scale=3.0),
            predicate=lambda meta: meta.get("has_feature", False),
        )

        selector = pds.OverlaySelector(rules=(rule,))

        # Should match (predicate true)
        overlays = selector.select(version="1.0.0", meta={"has_feature": True})
        assert len(overlays) == 1

        # Should not match (predicate false)
        overlays = selector.select(
            version="1.0.0", meta={"has_feature": False}
        )
        assert len(overlays) == 0

    def test_overlay_selector_priority_ordering(self):
        """Test overlay execution priority ordering."""
        overlay1 = pds.SpacingScaleOverlay(scale=2.0, priority=50)
        overlay2 = pds.FlipIndexAxesOverlay(flip_i=True, priority=10)
        overlay3 = pds.PermuteIndexAxesOverlay(order=(1, 2, 0), priority=30)

        rule1 = pds.OverlayRule(
            "rule1", SpecifierSet(">=0.0.0"), lambda m: overlay1
        )
        rule2 = pds.OverlayRule(
            "rule2", SpecifierSet(">=0.0.0"), lambda m: overlay2
        )
        rule3 = pds.OverlayRule(
            "rule3", SpecifierSet(">=0.0.0"), lambda m: overlay3
        )

        selector = pds.OverlaySelector(rules=(rule1, rule2, rule3))
        overlays = selector.select(version="1.0.0", meta={})

        # Should be ordered by overlay.priority: overlay2(10), overlay3(30),
        # overlay1(50)
        assert len(overlays) == 3
        assert overlays[0].priority == 10  # FlipIndexAxesOverlay
        assert overlays[1].priority == 30  # PermuteIndexAxesOverlay
        assert overlays[2].priority == 50  # SpacingScaleOverlay

    def test_overlay_selector_group_exclusivity(self):
        """Test group exclusivity in overlay selection."""
        rule1 = pds.OverlayRule(
            "rule1",
            SpecifierSet(">=0.0.0"),
            lambda m: pds.SpacingScaleOverlay(scale=2.0),
            group="spacing",
            rule_priority=10,
        )
        rule2 = pds.OverlayRule(
            "rule2",
            SpecifierSet(">=0.0.0"),
            lambda m: pds.SpacingScaleOverlay(scale=3.0),
            group="spacing",
            rule_priority=20,
        )

        selector = pds.OverlaySelector(rules=(rule1, rule2))
        overlays = selector.select(version="1.0.0", meta={})

        # Should only get the first matching rule from the group
        assert len(overlays) == 1
        assert overlays[0].scale == 2.0  # From rule1 (lower rule_priority)

    def test_overlay_selector_stop_after(self):
        """Test stop_after behavior."""
        rule1 = pds.OverlayRule(
            "rule1",
            SpecifierSet(">=0.0.0"),
            lambda m: pds.SpacingScaleOverlay(scale=2.0),
            rule_priority=10,
        )
        rule2 = pds.OverlayRule(
            "rule2",
            SpecifierSet(">=0.0.0"),
            lambda m: pds.FlipIndexAxesOverlay(flip_i=True),
            rule_priority=20,
            stop_after=True,
        )
        rule3 = pds.OverlayRule(
            "rule3",
            SpecifierSet(">=0.0.0"),
            lambda m: pds.PermuteIndexAxesOverlay(order=(1, 2, 0)),
            rule_priority=30,
        )

        selector = pds.OverlaySelector(rules=(rule1, rule2, rule3))
        overlays = selector.select(version="1.0.0", meta={})

        # Should stop after rule2, so rule3 should not be included
        # Rules are sorted by priority (highest first), so rule2 (priority 20)
        # comes before rule1 (priority 10)
        assert len(overlays) == 2
        assert isinstance(
            overlays[0], pds.FlipIndexAxesOverlay
        )  # From rule2 (priority 20, stop_after=True)
        assert isinstance(
            overlays[1], pds.SpacingScaleOverlay
        )  # From rule1 (priority 10)

    def test_with_rule(self):
        """Test adding rules immutably."""
        original = pds.OverlaySelector()
        rule = pds.OverlayRule(
            "test",
            SpecifierSet(">=0.0.0"),
            lambda m: pds.SpacingScaleOverlay(scale=1.0),
        )

        new_selector = original.with_rule(rule)

        assert len(original.rules) == 0  # Original unchanged
        assert len(new_selector.rules) == 1
        assert new_selector.rules[0] is rule

    def test_with_rules(self):
        """Test adding multiple rules immutably."""
        original = pds.OverlaySelector()
        rules = [
            pds.OverlayRule(
                "test1",
                SpecifierSet(">=0.0.0"),
                lambda m: pds.SpacingScaleOverlay(scale=1.0),
            ),
            pds.OverlayRule(
                "test2",
                SpecifierSet(">=0.0.0"),
                lambda m: pds.FlipIndexAxesOverlay(),
            ),
        ]

        new_selector = original.with_rules(rules)

        assert len(original.rules) == 0
        assert len(new_selector.rules) == 2


class TestUtilityFunctions:
    """Test utility functions."""

    def test_as_date_none(self):
        """Test _as_date with None."""
        assert pds._as_date(None) is None

    def test_as_date_date_object(self):
        """Test _as_date with date object."""
        d = date(2024, 6, 15)
        assert pds._as_date(d) == d

    def test_as_date_datetime_object(self):
        """Test _as_date with datetime object."""
        dt = datetime(2024, 6, 15, 14, 30, 0)
        result = pds._as_date(dt)
        # Should return just the date component, not full datetime
        assert result == date(2024, 6, 15)
        assert isinstance(result, date)
        assert not isinstance(result, datetime)

    def test_as_date_iso_string(self):
        """Test _as_date with ISO string."""
        result = pds._as_date("2024-06-15")
        assert result == date(2024, 6, 15)

    def test_as_date_invalid_string(self):
        """Test _as_date with invalid string."""
        with pytest.raises(ValueError):
            pds._as_date("not-a-date")

    def test_require_cardinal_valid(self):
        """Test _require_cardinal with valid matrix."""
        # Identity matrix is cardinal
        D = np.eye(3)
        pds._require_cardinal(D)  # Should not raise

        # Permutation matrix is cardinal
        D = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        pds._require_cardinal(D)  # Should not raise

        # Signed permutation is cardinal
        D = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        pds._require_cardinal(D)  # Should not raise

    def test_require_cardinal_invalid(self):
        """Test _require_cardinal with invalid matrix."""
        # Non-orthogonal matrix
        D = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="Direction is not cardinal"):
            pds._require_cardinal(D)

    def test_lps_world_to_index_spacing_cardinal(self):
        """Test LPS world to index spacing conversion."""
        # Identity direction (LPS = index)
        D = np.eye(3)
        lps_spacing = (0.5, 1.0, 2.0)

        result = pds.lps_world_to_index_spacing_cardinal(D, lps_spacing)
        assert result == (0.5, 1.0, 2.0)

        # Permuted axes (index k, i, j corresponds to world x, y, z)
        D = np.array(
            [
                [0, 0, 1],  # k -> x
                [1, 0, 0],  # i -> y
                [0, 1, 0],  # j -> z
            ]
        )

        result = pds.lps_world_to_index_spacing_cardinal(D, lps_spacing)
        assert result == (1.0, 2.0, 0.5)  # y, z, x spacing

    def test_apply_overlays(self):
        """Test apply_overlays function."""
        base_header = AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )

        overlays = [
            pds.SpacingScaleOverlay(scale=2.0, priority=10),
            pds.FlipIndexAxesOverlay(flip_i=True, priority=20),
        ]

        final_header, applied = pds.apply_overlays(
            base_header, overlays, {}, registration_multiscale_no=0
        )

        # Both overlays should be applied and tracked
        assert len(applied) == 2
        assert "spacing_scale" in applied
        assert "flip_index_axes" in applied

        # Check final result
        assert final_header.spacing == (2.0, 2.0, 2.0)  # Scaled
        assert final_header.direction[0, 0] == -1.0  # Flipped i axis

    def test_apply_overlays_no_change(self):
        """Test apply_overlays when overlay makes no changes."""
        base_header = AnatomicalHeader(
            origin=(0, 0, 0),
            spacing=(2, 2, 2),  # Already scaled
            direction=np.eye(3),
            size_ijk=(10, 10, 10),
        )

        # Overlay that would double the spacing, but it's already doubled
        overlay = pds.SpacingScaleOverlay(scale=1.0, priority=10)  # No change

        final_header, applied = pds.apply_overlays(
            base_header, [overlay], {}, registration_multiscale_no=0
        )

        # No changes detected, so overlay not in applied list
        assert len(applied) == 0
        assert final_header.spacing == base_header.spacing

    def test_estimate_pipeline_multiscale(self):
        """Test estimate_pipeline_multiscale function."""
        zarr_metadata = {}  # Not used in current implementation

        # Test version in supported range
        version = Version("0.0.25")
        result = pds.estimate_pipeline_multiscale(zarr_metadata, version)
        assert result == 3

        # Test version at boundaries
        version = Version("0.0.18")  # Lower bound
        result = pds.estimate_pipeline_multiscale(zarr_metadata, version)
        assert result == 3

        version = Version("0.0.33")  # Just below upper bound
        result = pds.estimate_pipeline_multiscale(zarr_metadata, version)
        assert result == 3

        # Test unsupported version
        version = Version("0.0.17")  # Below range
        result = pds.estimate_pipeline_multiscale(zarr_metadata, version)
        assert result is None

        version = Version("0.0.34")  # At upper bound (excluded)
        result = pds.estimate_pipeline_multiscale(zarr_metadata, version)
        assert result is None


class TestBuiltInSelectors:
    """Test built-in selector functions."""

    def test_get_selector(self):
        """Test get_selector returns a selector with built-in rules."""
        selector = pds.get_selector()

        assert isinstance(selector, pds.OverlaySelector)
        assert len(selector.rules) > 0  # Should have built-in rules

    def test_extend_selector(self):
        """Test extend_selector adds to default rules."""
        new_rule = pds.OverlayRule(
            "test_extension",
            SpecifierSet(">=0.0.0"),
            lambda m: pds.SpacingScaleOverlay(scale=5.0),
        )

        extended = pds.extend_selector(new_rule)
        original = pds.get_selector()

        assert len(extended.rules) == len(original.rules) + 1
        assert extended.rules[-1] is new_rule

    def test_make_selector(self):
        """Test make_selector creates selector from rules."""
        rules = [
            pds.OverlayRule(
                "rule1",
                SpecifierSet(">=0.0.0"),
                lambda m: pds.SpacingScaleOverlay(scale=1.0),
            ),
            pds.OverlayRule(
                "rule2",
                SpecifierSet(">=0.0.0"),
                lambda m: pds.FlipIndexAxesOverlay(),
            ),
        ]

        selector = pds.make_selector(rules)

        assert isinstance(selector, pds.OverlaySelector)
        assert len(selector.rules) == 2
        assert selector.rules[0] is rules[0]
        assert selector.rules[1] is rules[1]
