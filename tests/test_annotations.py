import numpy as np

from aind_zarr_utils import annotations as ann


def test_annotation_indices_to_anatomical(monkeypatch):
    # Dummy transform function just returns indices + 1
    def dummy_transform(img, indices):
        return indices + 1

    monkeypatch.setattr(
        ann, "transform_sitk_indices_to_physical_points", dummy_transform
    )
    img = object()
    annotations = {"a": np.array([[1, 2, 3], [4, 5, 6]])}
    result = ann.annotation_indices_to_anatomical(img, annotations)
    # Should reverse indices and add 1
    expected = {
        "a": np.array([[4, 3, 2], [7, 6, 5]])
    }  # [1,2,3][::-1] = [3,2,1] + 1 = [4,3,2]
    assert np.allclose(result["a"], expected["a"])


def test_annotations_and_descriptions_to_dict():
    annotation_points = {"a": [[1, 2, 3], [4, 5, 6]]}
    descriptions = {"a": ["foo", None]}
    result = ann.annotations_and_descriptions_to_dict(
        annotation_points, descriptions
    )
    assert "a" in result
    assert result["a"]["foo"] == [1, 2, 3]
    assert result["a"]["1"] == [4, 5, 6]


def test_pts_and_descriptions_to_pt_dict():
    points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    descriptions = ["foo,bar", None, " baz\n"]
    result = ann._pts_and_descriptions_to_pt_dict(points, descriptions)
    # Should sanitize and assign numeric label for None
    assert result["foobar"] == [1, 2, 3]
    assert result["1"] == [4, 5, 6]
    assert result["baz"] == [7, 8, 9]
