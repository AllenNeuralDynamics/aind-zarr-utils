"""Tests for ``aind_zarr_utils.image``.

Focused regression tests for the level > 0 ANTs convention conversion.
The math being tested matches the inline formula that lived in
``apply_pipeline_overlays_to_ants`` before the C4 split — these tests
pin the algorithm so future refactors that change behaviour are caught.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import (
    AnatomicalHeader,
    fix_corner_compute_origin,
)
from aind_anatomical_utils.coordinate_systems import _OPPOSITE_AXES

from aind_zarr_utils.image import _to_ants_convention


def _expected_inline(
    header: AnatomicalHeader,
    level: int,
    ants_shape: tuple[int, ...],
    ants_direction: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, float, float]]:
    """Compute the expected (spacing, origin) using the original inline formula.

    Mirrors ``apply_pipeline_overlays_to_ants`` (level > 0 branch) before the
    C4 split. Used as the regression oracle.
    """
    scale = 2**level
    spacing_rev_scaled = tuple(s * scale for s in reversed(header.spacing))
    origin_code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(header.direction_tuple())
    corner_code = "".join(_OPPOSITE_AXES[d] for d in origin_code)
    ants_origin, _, _ = fix_corner_compute_origin(
        ants_shape,
        spacing_rev_scaled,
        ants_direction,
        target_point=header.origin,
        corner_code=corner_code,
    )
    return spacing_rev_scaled, ants_origin


@pytest.fixture
def identity_header() -> AnatomicalHeader:
    """Header with identity direction and ITK orientation 'LPS'."""
    return AnatomicalHeader(
        origin=(1.0, 2.0, 3.0),
        spacing=(0.1, 0.2, 0.4),
        direction=np.eye(3),
        size_ijk=(8, 16, 32),
    )


@pytest.fixture
def rpi_header() -> AnatomicalHeader:
    """Header whose direction encodes the 'RPI' anatomical orientation.

    Column 0 → (-1, 0, 0): i-axis points toward R (= -L).
    Column 1 → ( 0, 1, 0): j-axis points toward P.
    Column 2 → ( 0, 0,-1): k-axis points toward I (= -S).
    """
    direction = np.diag((-1.0, 1.0, -1.0))
    return AnatomicalHeader(
        origin=(5.0, 10.0, -2.0),
        spacing=(0.5, 0.6, 0.7),
        direction=direction,
        size_ijk=(4, 8, 16),
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_to_ants_convention_spacing_is_reversed_and_scaled(identity_header: AnatomicalHeader, level: int) -> None:
    """Spacing is the reverse of the SITK spacing, then scaled by 2**level."""
    new_spacing, _ = _to_ants_convention(
        identity_header,
        level=level,
        ants_shape=(2, 4, 8),
        ants_direction=np.eye(3),
    )
    expected = tuple(s * 2**level for s in reversed(identity_header.spacing))
    assert new_spacing == pytest.approx(expected)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_to_ants_convention_matches_inline_formula_identity(identity_header: AnatomicalHeader, level: int) -> None:
    """The helper output equals the original inline formula (identity direction)."""
    ants_shape = (
        identity_header.size_ijk[0] >> level,
        identity_header.size_ijk[1] >> level,
        identity_header.size_ijk[2] >> level,
    )
    ants_direction = np.eye(3)

    new_spacing, new_origin = _to_ants_convention(identity_header, level, ants_shape, ants_direction)
    expected_spacing, expected_origin = _expected_inline(identity_header, level, ants_shape, ants_direction)

    assert new_spacing == pytest.approx(expected_spacing)
    assert tuple(new_origin) == pytest.approx(tuple(expected_origin))


@pytest.mark.parametrize("level", [1, 2, 3])
def test_to_ants_convention_matches_inline_formula_rpi(rpi_header: AnatomicalHeader, level: int) -> None:
    """The helper output equals the original inline formula (RPI direction)."""
    ants_shape = (
        rpi_header.size_ijk[0] >> level,
        rpi_header.size_ijk[1] >> level,
        rpi_header.size_ijk[2] >> level,
    )
    # The active overlays in the default rule set don't modify direction,
    # so the ANTs image's direction matches the SITK-convention header's.
    ants_direction = rpi_header.direction.copy()

    new_spacing, new_origin = _to_ants_convention(rpi_header, level, ants_shape, ants_direction)
    expected_spacing, expected_origin = _expected_inline(rpi_header, level, ants_shape, ants_direction)

    assert new_spacing == pytest.approx(expected_spacing)
    assert tuple(new_origin) == pytest.approx(tuple(expected_origin))


def test_to_ants_convention_uses_opposite_corner_code(
    identity_header: AnatomicalHeader,
) -> None:
    """The corner code passed to fix_corner_compute_origin is the *opposite*
    of the SITK orientation code derived from the direction matrix.

    For identity direction the SITK orientation is 'LPS' and its opposite
    'RAI'. The helper must call fix_corner_compute_origin with corner_code='RAI'.
    """
    sitk_code = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(identity_header.direction_tuple())
    expected_corner = "".join(_OPPOSITE_AXES[d] for d in sitk_code)

    # Run the helper and the explicit inline formula with the expected
    # corner code; outputs must agree.
    ants_shape = (4, 8, 16)
    ants_direction = np.eye(3)
    _, helper_origin = _to_ants_convention(
        identity_header, level=1, ants_shape=ants_shape, ants_direction=ants_direction
    )

    direct_origin, _, _ = fix_corner_compute_origin(
        ants_shape,
        tuple(s * 2 for s in reversed(identity_header.spacing)),
        ants_direction,
        target_point=identity_header.origin,
        corner_code=expected_corner,
    )

    assert tuple(helper_origin) == pytest.approx(tuple(direct_origin))
