"""Every pyramid level is placed over its level-0 sources, in the anatomical and the pipeline frames.

The ``real_ome_zarr`` fixture has level-0 size 10 on each axis and level 3 of size 1, so the pyramid truncates two
level-0 voxels at the far end of every axis: anchoring a corner on the level-0 grid instead of the level read would
misplace that level.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aind_zarr_utils import zarr as zarr_mod
from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_anatomical_stub

# The overlay's anchor for the default rule set: the template's origin, its first voxel's center.
PIPELINE_RAS_ANCHOR = np.array([-1.5114, -1.5, 1.5])


def _level0_center_of(level: int, index: np.ndarray) -> np.ndarray:
    """Continuous level-0 index of a level voxel's center."""
    factor = 2**level
    return factor * np.asarray(index, dtype=float) + (factor - 1) / 2


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize(
    "origin_kwargs",
    [{}, {"set_origin": (1.0, -2.0, 3.0)}, {"set_corner": "RAS", "set_corner_lps": (-1.0, -2.0, 3.0)}],
)
def test_stub_level_sits_over_its_sources(real_ome_zarr, mock_nd_metadata, level, origin_kwargs) -> None:
    """A level voxel's center is where its level-0 sources' center is, under every origin argument."""
    stub_0, _ = zarr_mod.zarr_to_sitk_stub(real_ome_zarr, mock_nd_metadata, level=0, **origin_kwargs)
    stub_l, size_l = zarr_mod.zarr_to_sitk_stub(real_ome_zarr, mock_nd_metadata, level=level, **origin_kwargs)

    for index in [np.zeros(3), np.array(size_l, dtype=float) - 1]:
        at_level = stub_l.TransformContinuousIndexToPhysicalPoint(tuple(index))
        at_level0 = stub_0.TransformContinuousIndexToPhysicalPoint(tuple(_level0_center_of(level, index)))
        assert at_level == pytest.approx(at_level0)


@pytest.mark.parametrize("level", [0, 2])
def test_images_share_the_stub_frame(real_ome_zarr, mock_nd_metadata, level) -> None:
    """SimpleITK and ANTs images of a level take the stub's origin and spacing."""
    stub, _ = zarr_mod.zarr_to_sitk_stub(real_ome_zarr, mock_nd_metadata, level=level)
    sitk_img = zarr_mod.zarr_to_sitk(real_ome_zarr, mock_nd_metadata, level=level)
    ants_img = zarr_mod.zarr_to_ants(real_ome_zarr, mock_nd_metadata, level=level)

    assert sitk_img.GetOrigin() == pytest.approx(stub.GetOrigin())
    assert sitk_img.GetSpacing() == pytest.approx(stub.GetSpacing())
    assert ants_img.origin == pytest.approx(stub.GetOrigin())


def test_pipeline_stub_anchors_the_registered_level(real_ome_zarr, mock_nd_metadata, mock_processing_data) -> None:
    """The pipeline frame puts the registered level-3 voxel's center, not a level-0 voxel's, at the anchor.

    Level 3 here is one voxel, the mean of level-0 voxels 0-7 on each axis, so its center is level-0 index 3.5.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stub, _ = mimic_pipeline_zarr_to_anatomical_stub(real_ome_zarr, mock_nd_metadata, mock_processing_data)

    registered_center = stub.TransformContinuousIndexToPhysicalPoint((3.5, 3.5, 3.5))
    assert registered_center == pytest.approx(tuple(PIPELINE_RAS_ANCHOR))
