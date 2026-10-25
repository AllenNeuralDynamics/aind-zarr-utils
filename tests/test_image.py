"""Tests for ``aind_zarr_utils.image``: level rescaling and the ANTs projection."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk
from aind_anatomical_utils.anatomical_volume import AnatomicalHeader

from aind_zarr_utils.image import _rescale_level, _to_ants_convention


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
    """Header whose direction encodes the 'RPI' anatomical orientation."""
    return AnatomicalHeader(
        origin=(5.0, 10.0, -2.0),
        spacing=(0.5, 0.6, 0.7),
        direction=np.diag((-1.0, 1.0, -1.0)),
        size_ijk=(8, 16, 24),
    )


def _physical(header: AnatomicalHeader, index_ijk: np.ndarray) -> np.ndarray:
    return np.asarray(header.origin) + header.direction @ (np.asarray(header.spacing) * index_ijk)


@pytest.mark.parametrize("header_name", ["identity_header", "rpi_header"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_rescale_level_matches_bin_shrink(header_name: str, level: int, request: pytest.FixtureRequest) -> None:
    """A coarser level's header equals what ``BinShrink`` gives the level-0 image."""
    header: AnatomicalHeader = request.getfixturevalue(header_name)
    factor = 2**level
    image = sitk.Image(list(header.size_ijk), sitk.sitkUInt8)
    header.update_sitk(image)
    shrunk = sitk.BinShrink(image, [factor] * 3)

    coarse = _rescale_level(header, factor, shrunk.GetSize())

    assert coarse.spacing == pytest.approx(shrunk.GetSpacing())
    assert coarse.origin == pytest.approx(shrunk.GetOrigin())


@pytest.mark.parametrize("level", [1, 2, 3])
def test_rescale_level_keeps_voxels_over_their_sources(rpi_header: AnatomicalHeader, level: int) -> None:
    """A coarse voxel's center is the mean of the level-0 centers beneath it, and the round trip is exact."""
    factor = 2**level
    coarse = _rescale_level(rpi_header, factor, (1, 1, 1))
    sources = np.stack([_physical(rpi_header, np.full(3, k, dtype=float)) for k in range(factor)])

    assert _physical(coarse, np.zeros(3)) == pytest.approx(sources.mean(axis=0))
    back = _rescale_level(coarse, 1 / factor, rpi_header.size_ijk)
    assert back.origin == pytest.approx(rpi_header.origin)
    assert back.spacing == pytest.approx(rpi_header.spacing)


def test_to_ants_convention_reverses_spacing(identity_header: AnatomicalHeader) -> None:
    """Spacing comes back in ANTs (array) index order, unscaled."""
    spacing, _ = _to_ants_convention(identity_header, ants_shape=(32, 16, 8), ants_direction=np.eye(3)[:, ::-1])
    assert spacing == pytest.approx(tuple(reversed(identity_header.spacing)))


@pytest.mark.parametrize("header_name", ["identity_header", "rpi_header"])
def test_to_ants_convention_maps_every_voxel_to_the_same_point(
    header_name: str, request: pytest.FixtureRequest
) -> None:
    """An array voxel lands at the same LPS point through the SimpleITK header and its ANTs projection."""
    header: AnatomicalHeader = request.getfixturevalue(header_name)
    ants_shape = tuple(reversed(header.size_ijk))
    ants_direction = header.direction[:, ::-1]

    spacing, origin = _to_ants_convention(header, ants_shape, ants_direction)

    for array_index in [np.zeros(3), np.array([3.0, 1.0, 2.0]), np.array(ants_shape, dtype=float) - 1]:
        ants_point = np.asarray(origin) + ants_direction @ (np.asarray(spacing) * array_index)
        assert ants_point == pytest.approx(_physical(header, array_index[::-1]))
