import numpy as np
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


def make_fake_image_node(shape=(1, 1, 3, 4, 5), level=0):
    class FakeData:
        def __init__(self, shape):
            self.shape = shape

        def compute(self):
            return np.ones(self.shape)

    class FakeNode:
        def __init__(self, shape):
            self.data = {level: FakeData(shape)}
            self.metadata = {
                "coordinateTransformations": [
                    [{"scale": [1.0, 2.0, 3.0, 4.0, 5.0]}] * (level + 1)
                ],
                "axes": [
                    {"name": "t", "unit": "second"},
                    {"name": "c", "unit": ""},
                    {"name": "z", "unit": "millimeter"},
                    {"name": "y", "unit": "millimeter"},
                    {"name": "x", "unit": "millimeter"},
                ],
            }

    return FakeNode(shape)


def fake_reader(*args, **kwargs):
    return [make_fake_image_node(*args, **kwargs)]


def test_open_zarr(monkeypatch):
    monkeypatch.setattr(zarr_mod, "Reader", lambda url: fake_reader)
    monkeypatch.setattr(zarr_mod, "parse_url", lambda uri: uri)
    image_node, zarr_meta = zarr_mod._open_zarr("fake_uri")
    assert hasattr(image_node, "data")
    assert "axes" in zarr_meta


def test_zarr_to_numpy(monkeypatch):
    monkeypatch.setattr(zarr_mod, "Reader", lambda url: fake_reader)
    monkeypatch.setattr(zarr_mod, "parse_url", lambda uri: uri)
    arr, meta, level = zarr_mod.zarr_to_numpy("fake_uri", level=0)
    assert arr.shape == (1, 1, 3, 4, 5)
    assert "axes" in meta
    assert level == 0


def test_zarr_to_numpy_anatomical(monkeypatch):
    monkeypatch.setattr(zarr_mod, "Reader", lambda url: fake_reader)
    monkeypatch.setattr(zarr_mod, "parse_url", lambda uri: uri)
    nd_metadata = {
        "acquisition": {
            "axes": [
                {
                    "dimension": "2",
                    "name": "Z",
                    "direction": "INFERIOR_SUPERIOR",
                },
                {
                    "dimension": "3",
                    "name": "Y",
                    "direction": "POSTERIOR_ANTERIOR",
                },
                {"dimension": "4", "name": "X", "direction": "LEFT_RIGHT"},
            ]
        }
    }
    arr, dirs, spacing = zarr_mod._zarr_to_numpy_anatomical(
        "fake_uri", nd_metadata, level=0
    )
    assert arr.shape == (3, 4, 5)
    assert set(dirs) == {"S", "A", "R"}
    assert len(spacing) == 3


def test_zarr_to_ants_and_sitk(monkeypatch):
    monkeypatch.setattr(zarr_mod, "Reader", lambda url: fake_reader)
    monkeypatch.setattr(zarr_mod, "parse_url", lambda uri: uri)
    nd_metadata = {
        "acquisition": {
            "axes": [
                {
                    "dimension": "2",
                    "name": "Z",
                    "direction": "INFERIOR_SUPERIOR",
                },
                {
                    "dimension": "3",
                    "name": "Y",
                    "direction": "POSTERIOR_ANTERIOR",
                },
                {"dimension": "4", "name": "X", "direction": "LEFT_RIGHT"},
            ]
        }
    }

    # Patch ants and sitk
    class DummyAntsImage:
        pass

    class DummySitkImage:
        def SetSpacing(self, spacing):
            self.spacing = spacing

        def SetOrigin(self, origin):
            self.origin = origin

        def SetDirection(self, direction):
            self.direction = direction

    monkeypatch.setattr(
        zarr_mod.ants,
        "from_numpy",
        lambda arr, spacing, direction, origin: DummyAntsImage(),
    )
    monkeypatch.setattr(
        zarr_mod.sitk, "GetImageFromArray", lambda arr: DummySitkImage()
    )

    class DummyDICOMOrient:
        @staticmethod
        def GetDirectionCosinesFromOrientation(dir_str):
            return tuple(range(9))

    monkeypatch.setattr(
        zarr_mod.sitk, "DICOMOrientImageFilter", DummyDICOMOrient
    )
    ants_img = zarr_mod.zarr_to_ants("fake_uri", nd_metadata, level=0)
    assert isinstance(ants_img, DummyAntsImage)
    sitk_img = zarr_mod.zarr_to_sitk("fake_uri", nd_metadata, level=0)
    assert hasattr(sitk_img, "spacing")
    assert hasattr(sitk_img, "origin")
    assert hasattr(sitk_img, "direction")


def test_zarr_to_sitk_stub(monkeypatch):
    monkeypatch.setattr(zarr_mod, "Reader", lambda url: fake_reader)
    monkeypatch.setattr(zarr_mod, "parse_url", lambda uri: uri)
    nd_metadata = {
        "acquisition": {
            "axes": [
                {
                    "dimension": "2",
                    "name": "Z",
                    "direction": "INFERIOR_SUPERIOR",
                },
                {
                    "dimension": "3",
                    "name": "Y",
                    "direction": "POSTERIOR_ANTERIOR",
                },
                {"dimension": "4", "name": "X", "direction": "LEFT_RIGHT"},
            ]
        }
    }

    class DummySitkImage:
        def SetSpacing(self, spacing):
            self.spacing = spacing

        def SetOrigin(self, origin):
            self.origin = origin

        def SetDirection(self, direction):
            self.direction = direction

    monkeypatch.setattr(
        zarr_mod.sitk, "Image", lambda shape, dtype: DummySitkImage()
    )

    class DummyDICOMOrient:
        @staticmethod
        def GetDirectionCosinesFromOrientation(dir_str):
            return tuple(range(9))

    monkeypatch.setattr(
        zarr_mod.sitk, "DICOMOrientImageFilter", DummyDICOMOrient
    )
    stub_img = zarr_mod.zarr_to_sitk_stub("fake_uri", nd_metadata, level=0)
    assert hasattr(stub_img, "spacing")
    assert hasattr(stub_img, "origin")
    assert hasattr(stub_img, "direction")
