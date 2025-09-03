import numpy as np
import pytest

from aind_zarr_utils import neuroglancer as ng

# Use shared infrastructure from conftest.py


def test_resolve_layer_names():
    layers = [
        {"name": "a", "type": "annotation"},
        {"name": "b", "type": "image"},
        {"name": "c", "type": "annotation"},
    ]
    assert ng._resolve_layer_names(layers, None, "annotation") == ["a", "c"]
    assert ng._resolve_layer_names(layers, "a", "annotation") == ["a"]
    assert ng._resolve_layer_names(layers, ["a", "c"], "annotation") == [
        "a",
        "c",
    ]
    with pytest.raises(ValueError):
        ng._resolve_layer_names(layers, 123, "annotation")


def test_extract_spacing():
    dim_data = {"z": (1.0, "mm"), "y": (2.0, "mm"), "x": (3.0, "mm")}
    spacing, units = ng._extract_spacing(dim_data)
    assert np.allclose(spacing, [1.0, 2.0, 3.0])
    assert units == ["mm", "mm", "mm"]
    with pytest.raises(ValueError):
        ng._extract_spacing({"z": (1, "mm"), "y": (2, "mm")})


def test_process_layer_and_descriptions():
    layer = {
        "annotations": [
            {"point": [1, 2, 3, 4], "description": "desc1"},
            {"point": [5, 6, 7, 8], "description": "desc2"},
        ]
    }
    points, descs = ng._process_layer_and_descriptions(
        layer, return_description=True
    )
    assert points.shape == (2, 3)
    assert np.allclose(points[0], [1, 2, 3])
    assert descs[0] == "desc1"
    # With spacing
    points2, _ = ng._process_layer_and_descriptions(
        layer, spacing=np.array([2, 3, 4]), return_description=False
    )
    assert np.allclose(points2[0], [2, 6, 12])
    # Bad shape
    bad_layer = {"annotations": [{"point": [1, 2, 3]}]}
    with pytest.raises(ValueError):
        ng._process_layer_and_descriptions(bad_layer)


def test_process_annotation_layers():
    layers = [
        {
            "name": "a",
            "annotations": [{"point": [1, 2, 3, 4], "description": "d1"}],
        },
        {
            "name": "b",
            "annotations": [{"point": [5, 6, 7, 8], "description": "d2"}],
        },
    ]
    ann, desc = ng._process_annotation_layers(
        layers, ["a", "b"], return_description=True
    )
    assert set(ann.keys()) == {"a", "b"}
    assert desc["a"][0] == "d1"
    ann2, desc2 = ng._process_annotation_layers(
        layers, ["a"], spacing=np.array([2, 2, 2]), return_description=True
    )
    assert np.allclose(ann2["a"], [[2, 4, 6]])


def test_get_layer_by_name():
    layers = [{"name": "foo"}, {"name": "bar"}]
    assert ng._get_layer_by_name(layers, "foo") == {"name": "foo"}
    with pytest.raises(ValueError):
        ng._get_layer_by_name(layers, "baz")


def test_neuroglancer_annotations_to_indices(monkeypatch):
    data = {
        "layers": [
            {
                "name": "a",
                "type": "annotation",
                "annotations": [
                    {"point": [1, 2, 3, 4], "description": "desc"}
                ],
            }
        ]
    }
    monkeypatch.setattr(
        ng, "_resolve_layer_names", lambda layers, names, layer_type: ["a"]
    )
    monkeypatch.setattr(
        ng,
        "_process_annotation_layers",
        lambda layers, names, return_description: (
            {"a": np.array([[1, 2, 3]])},
            {"a": np.array(["desc"])},
        ),
    )
    ann, desc = ng.neuroglancer_annotations_to_indices(data)
    assert "a" in ann
    assert "a" in desc


def test_neuroglancer_annotations_to_anatomical(
    mock_annotation_functions, monkeypatch
):
    data = {"layers": []}
    monkeypatch.setattr(
        ng,
        "neuroglancer_annotations_to_indices",
        lambda *a, **k: (
            {"a": np.array([[1, 2, 3]])},
            {"a": np.array(["desc"])},
        ),
    )
    points, desc = ng.neuroglancer_annotations_to_anatomical(
        data, "uri", {}, layer_names=None
    )
    assert "a" in points
    assert np.allclose(points["a"], [[2, 3, 4]])
    assert "a" in desc


def test_neuroglancer_annotations_to_global(monkeypatch):
    data = {
        "dimensions": {"z": (1, "mm"), "y": (2, "mm"), "x": (3, "mm")},
        "layers": [
            {
                "name": "a",
                "type": "annotation",
                "annotations": [
                    {"point": [1, 2, 3, 4], "description": "desc"}
                ],
            }
        ],
    }
    monkeypatch.setattr(
        ng, "_resolve_layer_names", lambda layers, names, layer_type: ["a"]
    )
    monkeypatch.setattr(
        ng,
        "_process_annotation_layers",
        lambda layers, names, spacing, return_description: (
            {"a": np.array([[1, 2, 3]])},
            {"a": np.array(["desc"])},
        ),
    )
    ann, units, desc = ng.neuroglancer_annotations_to_global(data)
    assert "a" in ann
    assert units == ["mm", "mm", "mm"]
    assert "a" in desc


def test_get_image_sources():
    data = {
        "layers": [
            {"name": "img1", "type": "image", "source": "url1"},
            {"name": "img2", "type": "image", "source": "url2"},
            {"name": "ann", "type": "annotation"},
        ]
    }
    sources = ng.get_image_sources(data)
    assert sources == {"img1": "url1", "img2": "url2"}
