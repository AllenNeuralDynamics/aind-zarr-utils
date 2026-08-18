import warnings

import numpy as np
import pytest

from aind_zarr_utils import neuroglancer as ng
from aind_zarr_utils.formats import neuroglancer as ng_impl

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
    points, descs = ng._process_layer_and_descriptions(layer, return_description=True)
    assert points.shape == (2, 3)
    assert np.allclose(points[0], [1, 2, 3])
    assert descs[0] == "desc1"
    # With spacing
    points2, _ = ng._process_layer_and_descriptions(layer, spacing=np.array([2, 3, 4]), return_description=False)
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
    ann, desc = ng._process_annotation_layers(layers, ["a", "b"], return_description=True)
    assert set(ann.keys()) == {"a", "b"}
    assert desc["a"][0] == "d1"
    ann2, desc2 = ng._process_annotation_layers(layers, ["a"], spacing=np.array([2, 2, 2]), return_description=True)
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


def test_neuroglancer_annotations_to_anatomical(neuroglancer_test_data, real_ome_zarr, mock_nd_metadata):
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
    ann, units, desc = ng.neuroglancer_annotations_to_scaled(neuroglancer_test_data)

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


# --- per-layer dimension order ---------------------------------------------
#
# A layer's points are stored in its own source.transform.outputDimensions
# order, frozen when the layer was created. Neuroglancer matches layer space to
# global space by name, so one state may legitimately hold several orders.
# 781469's annotations are the live case: ProbeC is z,y,x and ProbeA/B z,x,y.


def _layer(name, order, points, scale=(2e-06, 1.8e-06, 1.8e-06)):
    """An annotation layer declaring *order*, with matching per-axis scales."""
    by_name = dict(zip(("z", "y", "x"), scale))
    return {
        "name": name,
        "type": "annotation",
        "source": {
            "url": "local://annotations",
            "transform": {
                "outputDimensions": {
                    **{d: [by_name[d], "m"] for d in order},
                    "t": [0.001, "s"],
                }
            },
        },
        "annotations": [{"point": list(p), "description": None} for p in points],
    }


def _state(*layers):
    return {"dimensions": {"z": [2e-06, "m"], "y": [1.8e-06, "m"], "x": [1.8e-06, "m"]}, "layers": list(layers)}


def test_layer_declaring_zxy_is_permuted_to_canonical():
    """z,x,y points come back as z,y,x -- the 781469 ProbeA/B case."""
    state = _state(_layer("probeA", ("z", "x", "y"), [(825.0, 3193.0, 1893.0, 0.0)]))

    ann, _ = ng.neuroglancer_annotations_to_indices(state)

    # declared z=825, x=3193, y=1893  ->  canonical (z, y, x)
    assert np.allclose(ann["probeA"][0], [825.0, 1893.0, 3193.0])


def test_canonical_layer_is_untouched():
    """A z,y,x layer must be byte-identical to the legacy positional read."""
    state = _state(_layer("probeC", ("z", "y", "x"), [(331.0, 3869.0, 2752.0, 0.0)]))

    ann, _ = ng.neuroglancer_annotations_to_indices(state)

    assert np.allclose(ann["probeC"][0], [331.0, 3869.0, 2752.0])


def test_mixed_orders_in_one_state_warn():
    """Legal, but it means any single-order assumption is wrong somewhere."""
    state = _state(
        _layer("probeC", ("z", "y", "x"), [(1.0, 2.0, 3.0, 0.0)]),
        _layer("probeA", ("z", "x", "y"), [(1.0, 2.0, 3.0, 0.0)]),
    )

    with pytest.warns(UserWarning, match="mixes annotation dimension orders"):
        ann, _ = ng.neuroglancer_annotations_to_indices(state)

    # Same stored numbers, different meaning: only probeA is permuted.
    assert np.allclose(ann["probeC"][0], [1.0, 2.0, 3.0])
    assert np.allclose(ann["probeA"][0], [1.0, 3.0, 2.0])


def test_uniform_state_does_not_warn(recwarn):
    """The common case must stay quiet, or the warning is worthless."""
    state = _state(
        _layer("a", ("z", "y", "x"), [(1.0, 2.0, 3.0, 0.0)]),
        _layer("b", ("z", "y", "x"), [(4.0, 5.0, 6.0, 0.0)]),
    )

    ng.neuroglancer_annotations_to_indices(state)

    assert [w for w in recwarn if "dimension orders" in str(w.message)] == []


def test_transform_matrix_is_refused():
    """We only permute and scale; applying a matrix is out of scope."""
    layer = _layer("probeA", ("z", "y", "x"), [(1.0, 2.0, 3.0, 0.0)])
    layer["source"]["transform"]["matrix"] = [[1, 0, 0, 0]] * 3
    state = _state(layer)

    with pytest.raises(ValueError, match="transform matrix"):
        ng.neuroglancer_annotations_to_indices(state)


def test_missing_spatial_dimension_is_named():
    """Fail saying what was found, rather than silently taking three columns."""
    layer = _layer("probeA", ("z", "y", "x"), [(1.0, 2.0, 3.0, 0.0)])
    dims = layer["source"]["transform"]["outputDimensions"]
    dims["q"] = dims.pop("x")
    state = _state(layer)

    with pytest.raises(ValueError, match="missing"):
        ng.neuroglancer_annotations_to_indices(state)


def test_scaled_uses_the_layers_own_scales():
    """A pasted-in layer can carry another volume's spacing."""
    state = _state(_layer("probeA", ("z", "x", "y"), [(10.0, 20.0, 30.0, 0.0)]))

    ann, units, _ = ng.neuroglancer_annotations_to_scaled(state)

    # canonical (z, y, x) = (10, 30, 20) scaled by (2e-6, 1.8e-6, 1.8e-6)
    assert np.allclose(ann["probeA"][0], [10 * 2e-06, 30 * 1.8e-06, 20 * 1.8e-06])
    assert units == ["m", "m", "m"]


def test_scaled_refuses_disagreeing_units():
    """One `units` return is only honest while every layer matches it."""
    layer = _layer("probeA", ("z", "y", "x"), [(1.0, 2.0, 3.0, 0.0)])
    layer["source"]["transform"]["outputDimensions"]["x"] = [1.8, "um"]
    state = _state(layer)

    with pytest.raises(ValueError, match="global dimensions"):
        ng.neuroglancer_annotations_to_scaled(state)


def test_named_keeps_each_layers_own_order():
    """The honest primitive: no permutation, space travels with the points."""
    state = _state(_layer("probeA", ("z", "x", "y"), [(825.0, 3193.0, 1893.0, 0.0)]))

    named = ng.neuroglancer_annotations_to_named(state)["probeA"]

    assert named.names == ("z", "x", "y", "t")
    assert np.allclose(named.points[0][:3], [825.0, 3193.0, 1893.0])
    assert np.allclose(named.to_canonical()[0], [825.0, 1893.0, 3193.0])


def test_annotation_layer_spaces_reports_each_order():
    state = _state(
        _layer("probeC", ("z", "y", "x"), [(1.0, 2.0, 3.0, 0.0)]),
        _layer("probeA", ("z", "x", "y"), [(1.0, 2.0, 3.0, 0.0)]),
    )

    assert ng.annotation_layer_spaces(state) == {
        "probeC": ("z", "y", "x", "t"),
        "probeA": ("z", "x", "y", "t"),
    }


# --- reference-model cross-check -------------------------------------------
#
# We parse the dimensions mapping ourselves rather than depend on neuroglancer
# at runtime. This pins that parse to the reference implementation, which is a
# dev-only dependency.


@pytest.mark.parametrize(
    "dimensions",
    [
        {"z": [2e-06, "m"], "y": [1.8e-06, "m"], "x": [1.8e-06, "m"], "t": [0.001, "s"]},
        {"z": [2e-06, "m"], "x": [1.8e-06, "m"], "y": [1.8e-06, "m"], "t": [0.001, "s"]},
        {"z": [500, "nm"], "y": [1.8, "um"], "x": [1.8, "um"]},
        {"z": [4e-06, "m"], "y": [4e-06, "m"], "x": [4e-06, "m"], "t": [0.001, "s"]},
        {"z": [1, ""], "y": [1, ""], "x": [1, ""]},
    ],
)
def test_our_space_parse_matches_neuroglancer(dimensions):
    """Names, scales and units must agree with the reference, order included."""
    viewer_state = pytest.importorskip("neuroglancer.viewer_state")

    reference = viewer_state.CoordinateSpace(dimensions)
    ours = ng_impl._coordinate_space(dimensions)

    assert ours is not None
    assert ours.names == reference.names
    assert ours.units == reference.units
    assert np.allclose(ours.scales, reference.scales)
    assert ours.rank == reference.rank


def test_coordinate_array_dimension_is_not_mistaken_for_a_scale():
    """A coordinate-array dimension carries labels, not a scale; refuse to guess."""
    assert ng_impl._coordinate_space({"z": {"coordinates": [0], "labels": ["a"]}}) is None


def test_transformless_layer_warns_when_global_order_contradicts_it():
    """Nothing on the layer says the order, and the state says it isn't z,y,x."""
    state = {
        "dimensions": {"z": [2e-06, "m"], "x": [1.8e-06, "m"], "y": [1.8e-06, "m"]},
        "layers": [
            {
                "name": "bare",
                "type": "annotation",
                "annotations": [{"point": [1.0, 2.0, 3.0, 0.0], "description": None}],
            }
        ],
    }

    with pytest.warns(UserWarning, match="declares no coordinate space"):
        ng.neuroglancer_annotations_to_indices(state)


def test_transformless_layer_is_quiet_when_global_order_agrees():
    state = {
        "dimensions": {"z": [2e-06, "m"], "y": [1.8e-06, "m"], "x": [1.8e-06, "m"]},
        "layers": [
            {
                "name": "bare",
                "type": "annotation",
                "annotations": [{"point": [1.0, 2.0, 3.0, 0.0], "description": None}],
            }
        ],
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ann, _ = ng.neuroglancer_annotations_to_indices(state)

    assert np.allclose(ann["bare"][0], [1.0, 2.0, 3.0])
