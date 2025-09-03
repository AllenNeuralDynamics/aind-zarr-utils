import pytest

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


# Use shared zarr infrastructure from conftest.py


def test_open_zarr(mock_zarr_operations):
    image_node, zarr_meta = zarr_mod._open_zarr("fake_uri")
    assert hasattr(image_node, "data")
    assert "axes" in zarr_meta


def test_zarr_to_numpy(mock_zarr_operations):
    arr, meta, level = zarr_mod.zarr_to_numpy("fake_uri", level=0)
    assert arr.shape == (1, 1, 10, 10, 10)
    assert "axes" in meta
    assert level == 0


def test_zarr_to_numpy_anatomical(mock_zarr_operations, mock_nd_metadata):
    arr, dirs, spacing, size = zarr_mod._zarr_to_numpy_anatomical(
        "fake_uri", mock_nd_metadata, level=0
    )
    assert arr.shape == (10, 10, 10)
    assert set(dirs) == {"S", "A", "R"}
    assert len(spacing) == 3
    assert len(size) == 3


def test_zarr_to_ants_and_sitk(
    mock_zarr_operations, mock_nd_metadata, mock_sitk_module, mock_ants_module
):
    ants_img = zarr_mod.zarr_to_ants("fake_uri", mock_nd_metadata, level=0)
    assert hasattr(ants_img, "spacing")

    sitk_img = zarr_mod.zarr_to_sitk("fake_uri", mock_nd_metadata, level=0)
    assert hasattr(sitk_img, "_spacing")
    assert hasattr(sitk_img, "_origin")
    assert hasattr(sitk_img, "_direction")


def test_zarr_to_sitk_stub(
    mock_zarr_operations, mock_nd_metadata, mock_sitk_module
):
    stub_img, size_ijk = zarr_mod.zarr_to_sitk_stub(
        "fake_uri", mock_nd_metadata, level=0
    )
    assert hasattr(stub_img, "_spacing")
    assert hasattr(stub_img, "_origin")
    assert hasattr(stub_img, "_direction")
    assert len(size_ijk) == 3
