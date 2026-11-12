"""Exercise the reader boundary against retained historical and newer metadata."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from aind_metadata_reader import Document, MetadataError, read_acquisition, read_processing
from botocore.exceptions import ClientError

from aind_zarr_utils import Asset, Points, Space
from aind_zarr_utils import asset as asset_mod
from aind_zarr_utils import zarr as zarr_mod
from aind_zarr_utils.io import paths
from aind_zarr_utils.io.metadata import anatomical_axes, direction_from_acquisition_metadata
from aind_zarr_utils.io.processing import (
    _get_zarr_import_view,
    _registration_input_path,
    _registration_pipeline_major,
    image_atlas_alignment_path_relative_from_processing,
)
from aind_zarr_utils.io.transforms import pipeline_transforms

FIXTURES = Path(__file__).parent / "fixtures" / "metadata_reader"


def capture(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def selected_views():
    metadata = capture("771432-docdb-converted-v2.json")
    acquisition = read_acquisition(Document(metadata), frame_scope="local")
    processing = capture("791132-s3-processing.json")
    imported = read_processing(Document(processing)).one("Image importing")
    duplicate = copy.deepcopy(imported.document.raw)
    pipeline = copy.deepcopy(processing["pipelines"][0])
    pipeline["name"] = "Repeated registration"
    pipeline["version"] = "5.1.0"
    processing["pipelines"].append(pipeline)
    duplicate["pipeline_name"] = pipeline["name"]
    processing["data_processes"].append(duplicate)
    registration = read_processing(Document(processing)).find("Image importing")[-1]
    return metadata, processing, acquisition, registration


@pytest.mark.parametrize("release", ["v2.9.0", "v3.0.0"])
def test_new_coordinate_frames_match_real_zarr_axes_without_fabricated_dimensions(release, real_ome_zarr):
    metadata = capture(f"{release}-exaspim_acquisition.json")
    original = copy.deepcopy(metadata)
    _, _, directions, spacing, size = zarr_mod._zarr_to_anatomical(real_ome_zarr, metadata, level=0)
    assert directions == ["I", "P", "R"]  # Zarr z, y, x order, not metadata X, Y, Z.
    assert spacing == [1, 1, 1]
    assert size == [10, 10, 10]
    with pytest.raises(MetadataError, match="Array dimensions are not established"):
        direction_from_acquisition_metadata(metadata)
    assert metadata == original


def test_historical_acquisition_keeps_the_same_zarr_orientation(real_ome_zarr):
    legacy = capture("771432-docdb-historical-v1.json")
    converted = capture("771432-docdb-converted-v2.json")
    with pytest.raises(MetadataError, match="found 2"):
        anatomical_axes(converted)
    selected = read_acquisition(Document(converted), frame_scope="local")
    old = zarr_mod.zarr_to_sitk(real_ome_zarr, legacy, level=0)
    new = zarr_mod.zarr_to_sitk(real_ome_zarr, selected, level=0)
    np.testing.assert_array_equal(old.GetDirection(), new.GetDirection())
    assert old.GetSpacing() == new.GetSpacing()
    assert old.GetOrigin() == new.GetOrigin()


def test_unfamiliar_direction_cannot_silently_become_a_cardinal_axis():
    metadata = {
        "axes": [
            {"name": "x", "dimension": 0, "direction": "right-ish"},
            {"name": "y", "dimension": 1, "direction": "Anterior_to_posterior"},
            {"name": "z", "dimension": 2, "direction": "Superior_to_inferior"},
        ]
    }
    with pytest.raises(MetadataError, match="cardinal"):
        anatomical_axes(metadata)


def test_real_v2_processing_locates_the_file_input_and_pipeline_version(caplog):
    metadata = capture("791132-s3-processing.json")
    original = copy.deepcopy(metadata)
    assert _registration_input_path(metadata).endswith("/Ex_639_Em_680.zarr/3")
    assert image_atlas_alignment_path_relative_from_processing(metadata) == "image_atlas_alignment/Ex_639_Em_680/"
    assert _registration_pipeline_major(metadata) == 6
    assert "greater than max verified" in caplog.text
    assert metadata == original


@pytest.mark.parametrize("filename", ["771432-docdb-historical-v1.json", "771432-docdb-converted-v2.json"])
def test_paired_processing_preserves_registration_producer_version(filename):
    metadata = capture(filename)
    process = _get_zarr_import_view(metadata)
    assert process.code_version == "0.0.31"
    assert process.pipeline_version == "3.0.1"


def test_repeated_registration_requires_selection_and_retains_pipeline_association():
    metadata = capture("791132-s3-processing.json")
    imported = read_processing(Document(metadata)).one("Image importing")
    duplicate = copy.deepcopy(imported.document.raw)
    duplicate["output_parameters"]["input_location"] = "/data/other.zarr/2"
    metadata["data_processes"].append(duplicate)
    with pytest.raises(ValueError, match="Ambiguous"):
        _registration_input_path(metadata)
    assert _registration_input_path(metadata, "s3://bucket/other.zarr").endswith("other.zarr/2")
    assert _registration_pipeline_major(metadata, "s3://bucket/other.zarr") == 6
    with pytest.raises(ValueError, match="No registration import matches"):
        _get_zarr_import_view(metadata, "s3://bucket/unknown.zarr")


def test_unique_registration_can_be_used_for_another_image_channel():
    metadata = capture("791132-s3-processing.json")
    assert _get_zarr_import_view(metadata, "s3://bucket/other.zarr").code_version == "0.0.37"


def test_split_metadata_files_are_loaded_without_synthesizing_an_envelope(tmp_path):
    acquisition = capture("v2.9.0-exaspim_acquisition.json")
    processing = capture("791132-s3-processing.json")
    (tmp_path / "acquisition.json").write_text(json.dumps(acquisition))
    (tmp_path / "processing.json").write_text(json.dumps(processing))
    uri, metadata, recorded_processing = paths.alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
        asset_uri=str(tmp_path)
    )
    assert uri == str(tmp_path / "image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr")
    assert metadata == acquisition
    assert recorded_processing == processing


def test_existing_aggregate_is_not_replaced_with_a_different_source(tmp_path):
    aggregate = capture("771432-docdb-historical-v1.json")
    (tmp_path / "metadata.nd.json").write_text(json.dumps(aggregate))
    (tmp_path / "acquisition.json").write_text("{}")
    assert paths._acquisition_metadata(str(tmp_path)) == aggregate


@pytest.mark.parametrize("code", ["NoSuchKey", "404", "AccessDenied", "InternalError"])
def test_s3_fallback_distinguishes_missing_metadata_from_failed_reads(monkeypatch, code):
    error = ClientError({"Error": {"Code": code}}, "GetObject")
    read = Mock(side_effect=[error, {"object_type": "Acquisition"}])
    monkeypatch.setattr(paths, "get_json", read)
    if code in {"NoSuchKey", "404"}:
        assert paths._acquisition_metadata("s3://bucket/asset") == {"object_type": "Acquisition"}
        assert read.call_args.args[0] == "s3://bucket/asset/acquisition.json"
    else:
        with pytest.raises(ClientError):
            paths._acquisition_metadata("s3://bucket/asset")
        assert read.call_count == 1


def test_malformed_aggregate_does_not_trigger_fallback(monkeypatch):
    read = Mock(side_effect=json.JSONDecodeError("Bad JSON", "", 0))
    monkeypatch.setattr(paths, "get_json", read)
    with pytest.raises(json.JSONDecodeError):
        paths._acquisition_metadata("s3://bucket/asset")
    assert read.call_count == 1


@pytest.mark.parametrize("selection", ["acquisition", "frame"])
def test_asset_uses_selected_frame_for_images_stubs_and_points(selected_views, real_ome_zarr, selection):
    metadata, processing, acquisition, registration = selected_views
    if selection == "frame":
        acquisition = read_acquisition(Document(metadata)).one_frame(
            pointer="/acquisition/data_streams/0/configurations/0/local_coordinate_system"
        )
    selected = Asset(real_ome_zarr, metadata, processing, acquisition=acquisition, registration=registration)
    with pytest.raises(MetadataError, match="found 2"):
        Asset(real_ome_zarr, metadata, processing).stub()
    expected = zarr_mod.zarr_to_sitk(real_ome_zarr, acquisition, level=0)
    assert selected.image(level=0).GetDirection() == expected.GetDirection()
    assert selected.stub()[0].GetDirection() == expected.GetDirection()
    points = Points({"cell": np.array([[1.5, 2.0, 3.5]])}, Space.ZARR_INDICES)
    anatomical = selected.transform(points, to=Space.LS_ANATOMICAL_MM)
    np.testing.assert_allclose(anatomical.values["cell"], [[-3.5, 2.0, -1.5]])
    restored = selected.transform(anatomical, to=Space.ZARR_INDICES)
    np.testing.assert_allclose(restored.values["cell"], points.values["cell"])
    assert selected.acquisition is acquisition
    assert selected.registration is registration
    assert dict(selected.metadata) == metadata
    assert dict(selected.processing) == processing


def test_selected_run_disambiguates_same_channel_and_retains_pipeline_association(selected_views):
    _, processing, _, registration = selected_views
    uri = "s3://bucket/asset/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr"
    with pytest.raises(ValueError, match="Ambiguous"):
        pipeline_transforms(uri, processing)
    assert registration.pipeline_version == "5.1.0"
    assert registration.pipeline.pointer == "/pipelines/1"
    assert _registration_pipeline_major(registration) == 5
    paths_for_run, _ = pipeline_transforms(uri, registration)
    assert paths_for_run.base == "s3://bucket/asset/image_atlas_alignment/Ex_639_Em_680"


@pytest.mark.parametrize("method", ["stub", "image", "apply_overlays"])
def test_pipeline_geometry_keeps_original_metadata_for_overlays(selected_views, real_ome_zarr, method):
    _, processing, acquisition, registration = selected_views
    # Supply a different, unambiguous frame as raw metadata so a lost selection
    # would yield a different direction rather than merely a selection error.
    raw_metadata = capture("v2.9.0-exaspim_acquisition.json")
    selector = Mock()
    selector.select.return_value = []
    asset = Asset(
        real_ome_zarr,
        raw_metadata,
        processing,
        acquisition=acquisition,
        registration=registration,
        overlay_selector=selector,
    )
    expected = zarr_mod.zarr_to_sitk(real_ome_zarr, acquisition, level=0)
    if method == "apply_overlays":
        image = asset.image(level=0)
        asset.apply_overlays(image, level=0)
    elif method == "stub":
        image, _ = asset.stub(pipeline=True)
    else:
        image = asset.image(level=0, pipeline=True)
    assert image.GetDirection() == expected.GetDirection()
    assert selector.select.call_args.kwargs["meta"] is asset.metadata
    assert selector.select.call_args.kwargs["version"] == registration.code_version


def test_asset_downloads_transforms_using_selected_process(selected_views, monkeypatch):
    metadata, processing, acquisition, registration = selected_views
    resolver = Mock(return_value=(["points.mat"], [True], ["image.mat"], [False]))
    monkeypatch.setattr(asset_mod, "pipeline_transforms_local_paths", resolver)
    asset = Asset(
        "s3://bucket/asset/image.zarr", metadata, processing, acquisition=acquisition, registration=registration
    )
    assert asset.transforms.point_paths == ["points.mat"]
    assert asset.transforms.image_paths == ["image.mat"]
    assert resolver.call_count == 1
    assert resolver.call_args.args[1] is registration


@pytest.mark.parametrize("constructor", ["root", "zarr", "neuroglancer-root", "neuroglancer-zarr"])
def test_discovery_preserves_explicit_selections(selected_views, tmp_path, real_ome_zarr, monkeypatch, constructor):
    metadata, processing, acquisition, registration = selected_views
    (tmp_path / "metadata.nd.json").write_text(json.dumps(metadata))
    (tmp_path / "processing.json").write_text(json.dumps(processing))
    opened = zarr_mod._open_zarr(real_ome_zarr)
    monkeypatch.setattr(asset_mod, "_open_zarr", lambda _, **kwargs: opened)
    source = str(tmp_path / "image_tile_fusing/OMEZarr/other.zarr")
    monkeypatch.setattr(asset_mod, "get_image_sources", lambda *args, **kwargs: {"other": source})
    options = {"acquisition": acquisition, "registration": registration}
    if constructor == "root":
        asset = Asset.from_root(str(tmp_path), **options)
    elif constructor == "zarr":
        asset = Asset.from_zarr(source, **options)
    elif constructor == "neuroglancer-root":
        asset = Asset.from_neuroglancer({}, asset_uri=str(tmp_path), **options)
    else:
        asset = Asset.from_neuroglancer({}, **options)
    assert asset.alignment_zarr_uri == str(tmp_path / "image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr")
    assert asset.acquisition is acquisition
    assert asset.registration is registration
    assert dict(asset.metadata) == metadata
    assert dict(asset.processing) == processing
    assert asset.stub()[0].GetDirection() == zarr_mod.zarr_to_sitk(real_ome_zarr, acquisition, level=0).GetDirection()


def test_a_different_process_type_is_not_accepted_as_registration():
    process = read_processing(Document(capture("791132-s3-processing.json"))).find("Image tile fusing")[0]
    with pytest.raises(ValueError, match="Image importing"):
        _get_zarr_import_view(process)


def test_unresolved_selected_input_does_not_fall_back_to_another_run(selected_views):
    _, _, _, registration = selected_views
    unknown_producer = replace(registration, code_version="0.0.38")
    assert _registration_input_path(unknown_producer) is None


def test_selected_run_without_pipeline_association_does_not_borrow_one(selected_views):
    _, _, _, registration = selected_views
    missing_pipeline = replace(registration, pipeline_version=None)
    with pytest.raises(ValueError, match="Missing pipeline version"):
        _registration_pipeline_major(missing_pipeline)
