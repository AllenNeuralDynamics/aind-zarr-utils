import numpy as np
import SimpleITK as sitk

from aind_zarr_utils import annotations as ann


def test_annotation_indices_to_anatomical():
    """Test converting annotation indices to anatomical coordinates using real
    SimpleITK."""
    # Create a real SimpleITK image with known properties
    # Spacing: 2mm in each direction
    # Origin: (10, 20, 30) mm
    # Direction: identity (no rotation)
    img = sitk.Image([5, 5, 5], sitk.sitkUInt8)
    img.SetSpacing((2.0, 2.0, 2.0))
    img.SetOrigin((10.0, 20.0, 30.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    # Annotation indices in (z, y, x) order (Python/numpy convention)
    annotations = {"a": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])}

    result = ann.annotation_indices_to_anatomical(img, annotations)

    # Expected physical points:
    # SimpleITK uses (x, y, z) order, so indices [1,2,3] in zyx → [3,2,1] in
    # xyz
    # Physical point = origin + index * spacing
    # For [1, 2, 3] (zyx) → [3, 2, 1] (xyz) → (10+3*2, 20+2*2, 30+1*2) = (16,
    # 24, 32)
    # For [2, 3, 4] (zyx) → [4, 3, 2] (xyz) → (10+4*2, 20+3*2, 30+2*2) = (18,
    # 26, 34)
    expected = {"a": np.array([[16.0, 24.0, 32.0], [18.0, 26.0, 34.0]])}

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
