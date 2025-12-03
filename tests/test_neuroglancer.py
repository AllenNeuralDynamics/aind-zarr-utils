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


def test_neuroglancer_annotations_to_indices(neuroglancer_test_data):
    """Test converting neuroglancer annotations to indices using real data."""
    ann, desc = ng.neuroglancer_annotations_to_indices(neuroglancer_test_data)

    # Should have annotations from both annotation layers
    assert "annotations_layer1" in ann
    assert "annotations_layer2" in ann
    assert "annotations_layer1" in desc
    assert "annotations_layer2" in desc

    # Verify layer1 has 2 points (strips 4th dimension from points)
    assert ann["annotations_layer1"].shape == (2, 3)
    assert np.allclose(ann["annotations_layer1"][0], [10.0, 20.0, 30.0])
    assert np.allclose(ann["annotations_layer1"][1], [15.0, 25.0, 35.0])
    assert desc["annotations_layer1"][0] == "point1"
    assert desc["annotations_layer1"][1] == "point2"

    # Verify layer2 has 1 point
    assert ann["annotations_layer2"].shape == (1, 3)
    assert np.allclose(ann["annotations_layer2"][0], [5.0, 10.0, 15.0])
    assert desc["annotations_layer2"][0] == "point3"


def test_neuroglancer_annotations_to_anatomical(
    neuroglancer_test_data, real_ome_zarr, mock_nd_metadata
):
    """Test converting neuroglancer annotations to anatomical space using real
    data."""
    points, desc = ng.neuroglancer_annotations_to_anatomical(
        neuroglancer_test_data,
        real_ome_zarr,
        mock_nd_metadata,
        layer_names=["annotations_layer1"],
    )

    # Should only have layer1 since we specified it
    assert "annotations_layer1" in points
    assert "annotations_layer1" in desc

    # Verify we got transformed coordinates (not just indices)
    # The transformation uses real SimpleITK under the hood
    assert points["annotations_layer1"].shape == (2, 3)
    # Values should be different from input indices due to transformation
    assert not np.allclose(points["annotations_layer1"][0], [10.0, 20.0, 30.0])

    # Descriptions should be preserved
    assert desc["annotations_layer1"][0] == "point1"
    assert desc["annotations_layer1"][1] == "point2"


def test_neuroglancer_annotations_to_scaled(neuroglancer_test_data):
    """Test converting neuroglancer annotations to scaled coordinates using
    real data."""
    ann, units, desc = ng.neuroglancer_annotations_to_scaled(
        neuroglancer_test_data
    )

    # Should have both annotation layers
    assert "annotations_layer1" in ann
    assert "annotations_layer2" in ann

    # Verify units match dimensions
    assert units == ["millimeter", "millimeter", "millimeter"]

    # Verify annotations are scaled by spacing from dimensions
    # dimensions: z=1.0mm, y=2.0mm, x=3.0mm
    # point1: [10, 20, 30] → [10*1.0, 20*2.0, 30*3.0] = [10, 40, 90]
    assert np.allclose(ann["annotations_layer1"][0], [10.0, 40.0, 90.0])
    assert np.allclose(ann["annotations_layer1"][1], [15.0, 50.0, 105.0])

    # point3: [5, 10, 15] → [5*1.0, 10*2.0, 15*3.0] = [5, 20, 45]
    assert np.allclose(ann["annotations_layer2"][0], [5.0, 20.0, 45.0])

    # Descriptions should be preserved
    assert desc["annotations_layer1"][0] == "point1"
    assert desc["annotations_layer2"][0] == "point3"


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
