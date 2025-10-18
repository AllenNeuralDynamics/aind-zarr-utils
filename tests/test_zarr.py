import numpy as np
import pytest
import SimpleITK as sitk

from aind_zarr_utils import zarr as zarr_mod


def test_direction_from_acquisition_metadata():
    acq_metadata = {
        "axes": [
            {"dimension": "0", "name": "X", "direction": "LEFT_RIGHT"},
            {"dimension": "1", "name": "Y", "direction": "POSTERIOR_ANTERIOR"},
            {"dimension": "2", "name": "Z", "direction": "INFERIOR_SUPERIOR"},
        ]
    }
    dims, axes, dirs = zarr_mod.direction_from_acquisition_metadata(
        acq_metadata
    )
    assert set(dims) == {"0", "1", "2"}
    assert set(axes) == {"x", "y", "z"}
    assert set(dirs) == {"R", "A", "S"}


def test_direction_from_nd_metadata():
    nd_metadata = {
        "acquisition": {
            "axes": [
                {"dimension": "0", "name": "X", "direction": "LEFT_RIGHT"},
                {
                    "dimension": "1",
                    "name": "Y",
                    "direction": "POSTERIOR_ANTERIOR",
                },
                {
                    "dimension": "2",
                    "name": "Z",
                    "direction": "INFERIOR_SUPERIOR",
                },
            ]
        }
    }
    dims, axes, dirs = zarr_mod.direction_from_nd_metadata(nd_metadata)
    assert set(dims) == {"0", "1", "2"}
    assert set(axes) == {"x", "y", "z"}
    assert set(dirs) == {"R", "A", "S"}


def test_units_to_meter():
    assert zarr_mod._units_to_meter("micrometer") == 1e-6
    assert zarr_mod._units_to_meter("millimeter") == 1e-3
    assert zarr_mod._units_to_meter("centimeter") == 1e-2
    assert zarr_mod._units_to_meter("meter") == 1.0
    assert zarr_mod._units_to_meter("kilometer") == 1e3
    with pytest.raises(ValueError):
        zarr_mod._units_to_meter("foo")


def test_unit_conversion():
    assert zarr_mod._unit_conversion("meter", "meter") == 1.0
    assert zarr_mod._unit_conversion("millimeter", "meter") == 1e-3
    assert zarr_mod._unit_conversion("meter", "millimeter") == 1e3
    assert zarr_mod._unit_conversion("centimeter", "millimeter") == 10.0


# Tests using real zarr infrastructure


def test_open_zarr(real_ome_zarr):
    """Test opening a real OME-ZARR file."""
    image_node, zarr_meta = zarr_mod._open_zarr(real_ome_zarr)
    # Verify we got a real Node object with data
    assert hasattr(image_node, "data")
    assert len(image_node.data) == 4  # 4 resolution levels
    # Verify metadata has required fields
    assert "axes" in zarr_meta
    assert "coordinateTransformations" in zarr_meta
    # Verify axes are correct
    axes_names = [ax["name"] for ax in zarr_meta["axes"]]
    assert axes_names == ["t", "c", "z", "y", "x"]


def test_zarr_to_numpy(real_ome_zarr):
    """Test converting real OME-ZARR to numpy array."""
    arr, meta, level = zarr_mod.zarr_to_numpy(real_ome_zarr, level=0)
    # Level 0 has full resolution: (1, 1, 10, 10, 10)
    assert arr.shape == (1, 1, 10, 10, 10)
    assert "axes" in meta
    assert level == 0
    # Verify actual data was read (not just shape)
    assert arr.dtype == np.float32
    # Data was created with np.arange, so check it's not all zeros/ones
    assert not np.all(arr == 0)
    assert not np.all(arr == 1)


def test_zarr_to_numpy_anatomical(real_ome_zarr, mock_nd_metadata):
    """Test converting OME-ZARR to anatomical numpy representation."""
    arr, dirs, spacing, size = zarr_mod._zarr_to_numpy_anatomical(
        real_ome_zarr, mock_nd_metadata, level=0
    )
    # Should extract only spatial dimensions (z, y, x)
    assert arr.shape == (10, 10, 10)
    # Verify anatomical directions from mock metadata
    assert set(dirs) == {"S", "A", "R"}
    # Verify we got spacing and size for 3 spatial dimensions
    assert len(spacing) == 3
    assert len(size) == 3
    # Verify size matches shape
    assert size == [10, 10, 10]


def test_zarr_to_ants_and_sitk(real_ome_zarr, mock_nd_metadata):
    """Test converting real OME-ZARR to ANTs and SimpleITK images."""
    ants_img = zarr_mod.zarr_to_ants(real_ome_zarr, mock_nd_metadata, level=0)
    # Test that we got a real ANTs image with expected properties
    assert hasattr(ants_img, "spacing")
    assert hasattr(ants_img, "origin")
    assert hasattr(ants_img, "direction")
    assert len(ants_img.spacing) == 3
    # Verify spacing matches OME-ZARR metadata (1.0 mm at level 0)
    assert np.allclose(ants_img.spacing, [1.0, 1.0, 1.0])

    sitk_img = zarr_mod.zarr_to_sitk(real_ome_zarr, mock_nd_metadata, level=0)
    # Test that we got a real SimpleITK image with expected properties
    assert isinstance(sitk_img, sitk.Image)
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    assert len(spacing) == 3
    assert len(origin) == 3
    assert len(direction) == 9
    # Verify spacing matches (SimpleITK reverses order: x, y, z)
    assert np.allclose(spacing, [1.0, 1.0, 1.0])
    # Verify size matches expected dimensions
    size = sitk_img.GetSize()
    assert size == (10, 10, 10)


def test_zarr_to_sitk_stub(real_ome_zarr, mock_nd_metadata):
    """Test creating SimpleITK stub from real OME-ZARR metadata."""
    stub_img, size_ijk = zarr_mod.zarr_to_sitk_stub(
        real_ome_zarr, mock_nd_metadata, level=0
    )
    # Test that we got a real SimpleITK image stub with expected properties
    assert isinstance(stub_img, sitk.Image)
    spacing = stub_img.GetSpacing()
    origin = stub_img.GetOrigin()
    direction = stub_img.GetDirection()
    assert len(spacing) == 3
    assert len(origin) == 3
    assert len(direction) == 9
    # Verify size_ijk tuple has correct values
    assert len(size_ijk) == 3
    assert size_ijk == (10, 10, 10)
