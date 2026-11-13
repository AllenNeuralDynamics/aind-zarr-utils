"""Fusion-version rules correct pyramid translations without caller-supplied evidence."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr
from aind_metadata_reader import Applicability

from aind_zarr_utils import Asset
from aind_zarr_utils import zarr as zarr_mod
from aind_zarr_utils.domain.pyramid import DEFAULT_FUSION_PYRAMID_RULES
from aind_zarr_utils.pipeline_transformed import base_and_pipeline_zarr_to_sitk

FIXTURES = Path(__file__).parent / "fixtures" / "pyramid_translations"
CHANNEL = "Ex_639_Em_667.zarr"
# Level-3 centre of the first 8^3 block of level-0 voxels, in LPS millimetres.
CORRECTED = (-0.0063, -0.0063, 0.007)
RECORDED = {"771432": (0, 0, 0), "837772": (-0.0035, -0.0035, 0.0035)}


def _write_pyramid(path, attributes):
    format_keyword = {"zarr_format": 2} if hasattr(zarr, "create_array") else {"zarr_version": 2}
    group = zarr.open_group(str(path), mode="w", **format_keyword)
    base = np.arange(16**3, dtype=np.float32).reshape(16, 16, 16)
    for level in range(4):
        factor = 2**level
        n = 16 // factor
        averaged = base.reshape(n, factor, n, factor, n, factor).mean(axis=(1, 3, 5))
        if hasattr(group, "create_array"):
            group.create_array(str(level), data=averaged[None, None], chunks=(1, 1, n, n, n))
        else:
            group.create_dataset(str(level), data=averaged[None, None], chunks=(1, 1, n, n, n), dimension_separator="/")
    group.attrs.update(attributes)
    return str(path)


def _fixture(tmp_path, subject, *, code_version=None, notes=None, attributes=None):
    """Write a 4-level block-mean pyramid with a captured asset's attributes and processing."""
    recorded = json.loads((FIXTURES / f"{subject}-zattrs.json").read_text())
    recorded["multiscales"][0]["datasets"] = recorded["multiscales"][0]["datasets"][:4]
    if attributes is not None:
        attributes(recorded)
    uri = _write_pyramid(tmp_path / CHANNEL, recorded)
    processing = json.loads((FIXTURES / f"{subject}-processing.json").read_text())
    for process in processing["processing_pipeline"]["data_processes"]:
        if code_version is not None:
            process["code_version"] = code_version
        if notes is not None:
            process["notes"] = notes
    processing["processing_pipeline"]["data_processes"].append(
        {"name": "Image importing", "code_version": "0.0.31", "input_location": f"/fixture/{CHANNEL}/3"}
    )
    return uri, processing


@pytest.fixture(params=["771432", "837772"])
def subject(request):
    return request.param


def test_default_rules_correct_each_fusion_writer(tmp_path, subject, mock_nd_metadata):
    uri, processing = _fixture(tmp_path, subject)
    asset = Asset(uri, mock_nd_metadata, processing)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert asset.stub(level=3)[0].GetOrigin() == pytest.approx(CORRECTED)
        assert asset.image(level=3).GetOrigin() == pytest.approx(CORRECTED)
        assert asset.stub(level=0)[0].GetOrigin() == (0, 0, 0)
    evaluation = asset.pyramid_geometry.evaluations[0]
    assert evaluation.outcome == Applicability.APPLICABLE
    rule = DEFAULT_FUSION_PYRAMID_RULES[0 if subject == "771432" else 1]
    assert rule.evidence in evaluation.evidence


def test_free_functions_correct_only_when_given_processing(tmp_path, subject, mock_nd_metadata):
    uri, processing = _fixture(tmp_path, subject)
    stub, _ = zarr_mod.zarr_to_sitk_stub(uri, mock_nd_metadata, level=3, processing_data=processing)
    assert stub.GetOrigin() == pytest.approx(CORRECTED)
    image = zarr_mod.zarr_to_sitk(uri, mock_nd_metadata, level=3, processing_data=processing)
    assert image.GetOrigin() == pytest.approx(CORRECTED)
    ants_image = zarr_mod.zarr_to_ants(uri, mock_nd_metadata, level=3, processing_data=processing)
    assert ants_image.origin == pytest.approx(CORRECTED)
    raw, _ = zarr_mod.zarr_to_sitk_stub(uri, mock_nd_metadata, level=3)
    assert raw.GetOrigin() == pytest.approx(RECORDED[subject])


def test_pipeline_frame_is_independent_of_the_physical_correction(tmp_path, subject, mock_nd_metadata):
    uri, processing = _fixture(tmp_path, subject)
    base, pipeline = base_and_pipeline_zarr_to_sitk(uri, mock_nd_metadata, processing, level=3)
    assert base.GetOrigin() == pytest.approx(CORRECTED)
    recorded = Asset(uri, mock_nd_metadata, processing, pyramid_rules=()).image(level=3, pipeline=True)
    assert pipeline.GetOrigin() == recorded.GetOrigin()
    assert pipeline.GetSpacing() == recorded.GetSpacing()
    assert pipeline.GetDirection() == recorded.GetDirection()


def test_versions_past_the_last_verified_are_corrected_with_a_warning(tmp_path, mock_nd_metadata):
    uri, processing = _fixture(tmp_path, "837772", code_version="0.0.7")
    with pytest.warns(UserWarning, match="newer than 0.0.6"):
        origin = Asset(uri, mock_nd_metadata, processing).stub(level=3)[0].GetOrigin()
    assert origin == pytest.approx(CORRECTED)


@pytest.mark.parametrize(
    ("subject_id", "options", "message"),
    [
        ("771432", {"code_version": "0.0.5"}, "No pyramid-translation rule covers aind-smartspim-fuse"),
        ("771432", {"notes": "Fusing channel with BigStitcher"}, "disagree about the writer"),
        (
            "837772",
            {
                "attributes": lambda a: [
                    d.__setitem__("coordinateTransformations", d["coordinateTransformations"][:1])
                    for d in a["multiscales"][0]["datasets"]
                ]
            },
            "are absent, but rule",
        ),
    ],
    ids=["no-rule", "writer-disagreement", "unexpected-signature"],
)
def test_unrecognized_producers_keep_recorded_geometry(tmp_path, mock_nd_metadata, subject_id, options, message):
    uri, processing = _fixture(tmp_path, subject_id, **options)
    recorded, _ = zarr_mod.zarr_to_sitk_stub(uri, mock_nd_metadata, level=3)
    asset = Asset(uri, mock_nd_metadata, processing)
    with pytest.warns(UserWarning, match=message):
        assert asset.stub(level=3)[0].GetOrigin() == recorded.GetOrigin()
    assert asset.pyramid_geometry is None


def test_missing_fusion_record_warns_only_for_coarse_levels(tmp_path, mock_nd_metadata):
    uri, processing = _fixture(tmp_path, "771432")
    processing["processing_pipeline"]["data_processes"] = [
        p for p in processing["processing_pipeline"]["data_processes"] if p["name"] != "Image tile fusing"
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        zarr_mod.zarr_to_sitk_stub(uri, mock_nd_metadata, level=0, processing_data=processing)
    with pytest.warns(UserWarning, match="No unique 'Image tile fusing' process"):
        stub, _ = zarr_mod.zarr_to_sitk_stub(uri, mock_nd_metadata, level=3, processing_data=processing)
    assert stub.GetOrigin() == pytest.approx(RECORDED["771432"])


def test_empty_rules_use_recorded_geometry_silently(tmp_path, subject, mock_nd_metadata):
    uri, processing = _fixture(tmp_path, subject)
    asset = Asset(uri, mock_nd_metadata, processing, pyramid_rules=())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert asset.stub(level=3)[0].GetOrigin() == pytest.approx(RECORDED[subject])
    assert asset.pyramid_geometry is None
