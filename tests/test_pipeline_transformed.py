"""Tests for pipeline_transformed module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aind_zarr_utils import pipeline_transformed as pt

# Long string constant to avoid line length issues
ATLAS_ALIGNMENT_NOTES = (
    "Template based registration: LS -> template -> Allen CCFv3 Atlas"
)


class TestPathUtilities:
    """Tests for path manipulation functions."""

    def test_asset_from_zarr_pathlike(self):
        zarr_path = Path("/data/acquisition/session.ome.zarr/0")
        asset_path = pt._asset_from_zarr_pathlike(zarr_path)
        assert asset_path == Path("/data")

    def test_asset_from_zarr_any_file_path(self):
        zarr_uri = "/data/acquisition/session.ome.zarr/0"
        asset_uri = pt._asset_from_zarr_any(zarr_uri)
        assert asset_uri == "/data"

    def test_asset_from_zarr_any_s3_uri(self):
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"
        asset_uri = pt._asset_from_zarr_any(zarr_uri)
        assert asset_uri == "s3://bucket/data"

    def test_zarr_base_name_pathlike(self):
        from pathlib import PurePath

        # Standard case
        p = PurePath("data/session.ome.zarr/0")
        assert pt._zarr_base_name_pathlike(p) == "session"

        # Multiple suffixes
        p = PurePath("data/session.zarr/0")
        assert pt._zarr_base_name_pathlike(p) == "session"

        # No zarr suffix
        p = PurePath("data/session/0")
        assert pt._zarr_base_name_pathlike(p) is None

    def test_zarr_base_name_any(self):
        # S3 URI
        base = "s3://bucket/data/session.ome.zarr/0"
        assert pt._zarr_base_name_any(base) == "session"

        # Local path
        base = "/data/session.ome.zarr/0"
        assert pt._zarr_base_name_any(base) == "session"


class TestDataClasses:
    """Tests for TransformChain and TemplatePaths dataclasses."""

    def test_transform_chain_creation(self):
        chain = pt.TransformChain(
            fixed="ccf",
            moving="template",
            forward_chain=["warp.nii.gz", "affine.mat"],
            forward_chain_invert=[False, False],
            reverse_chain=["affine.mat", "inverse_warp.nii.gz"],
            reverse_chain_invert=[True, False],
        )
        assert chain.fixed == "ccf"
        assert chain.moving == "template"
        assert len(chain.forward_chain) == 2
        assert len(chain.reverse_chain) == 2

    def test_template_paths_creation(self):
        chain = pt.TransformChain(
            fixed="ccf",
            moving="template",
            forward_chain=[],
            forward_chain_invert=[],
            reverse_chain=[],
            reverse_chain_invert=[],
        )
        paths = pt.TemplatePaths(base="s3://bucket/transforms/", chain=chain)
        assert paths.base == "s3://bucket/transforms/"
        assert paths.chain == chain


class TestProcessingDataParsing:
    """Tests for processing metadata parsing functions."""

    @pytest.fixture
    def sample_processing_data(self):
        return {
            "processing_pipeline": {
                "pipeline_version": "3.1.0",
                "data_processes": [
                    {
                        "name": "Image importing",
                        "code_version": "0.0.25",
                        "input_location": "s3://bucket/session.ome.zarr",
                    },
                    {
                        "name": "Image atlas alignment",
                        "notes": ATLAS_ALIGNMENT_NOTES,
                        "input_location": "s3://bucket/session.ome.zarr",
                    },
                ],
            }
        }

    def test_get_processing_pipeline_data_valid(self, sample_processing_data):
        pipeline = pt._get_processing_pipeline_data(sample_processing_data)
        assert pipeline["pipeline_version"] == "3.1.0"
        assert len(pipeline["data_processes"]) == 2

    def test_get_processing_pipeline_data_missing_version(self):
        data = {"processing_pipeline": {}}
        with pytest.raises(ValueError, match="Missing pipeline version"):
            pt._get_processing_pipeline_data(data)

    def test_get_processing_pipeline_data_wrong_major_version(self):
        data = {"processing_pipeline": {"pipeline_version": "2.0.0"}}
        with pytest.raises(ValueError, match="Unsupported pipeline version"):
            pt._get_processing_pipeline_data(data)

    def test_get_zarr_import_process(self, sample_processing_data):
        proc = pt._get_zarr_import_process(sample_processing_data)
        assert proc is not None
        assert proc["name"] == "Image importing"
        assert proc["code_version"] == "0.0.25"

    def test_get_zarr_import_process_not_found(self):
        data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }
        proc = pt._get_zarr_import_process(data)
        assert proc is None

    def test_get_image_atlas_alignment_process(self, sample_processing_data):
        proc = pt._get_image_atlas_alignment_process(sample_processing_data)
        assert proc is not None
        assert proc["name"] == "Image atlas alignment"

    def test_get_image_atlas_alignment_process_not_found(self):
        data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [
                    {"name": "Other process", "notes": "Different notes"}
                ],
            }
        }
        proc = pt._get_image_atlas_alignment_process(data)
        assert proc is None

    def test_image_atlas_alignment_path_relative(self, sample_processing_data):
        rel_path = pt.image_atlas_alignment_path_relative_from_processing(
            sample_processing_data
        )
        assert rel_path == "image_atlas_alignment/session/"

    def test_image_atlas_alignment_path_relative_not_found(self):
        data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }
        rel_path = pt.image_atlas_alignment_path_relative_from_processing(data)
        assert rel_path is None


class TestPipelineTransformConstants:
    """Tests for pipeline transform configuration constants."""

    def test_pipeline_template_transforms_structure(self):
        transforms = pt._PIPELINE_TEMPLATE_TRANSFORMS
        assert "SmartSPIM-template_2024-05-16_11-26-14" in transforms

        template = transforms["SmartSPIM-template_2024-05-16_11-26-14"]
        assert template.base.startswith("s3://")
        assert template.chain.fixed == "ccf"
        assert template.chain.moving == "template"
        assert len(template.chain.forward_chain) == 2
        assert len(template.chain.reverse_chain) == 2

    def test_pipeline_individual_transforms_structure(self):
        transforms = pt._PIPELINE_INDIVIDUAL_TRANSFORMS
        assert 3 in transforms

        chain = transforms[3]
        assert chain.fixed == "template"
        assert chain.moving == "individual"
        assert len(chain.forward_chain) == 2
        assert len(chain.reverse_chain) == 2


class TestMimicPipelineStub:
    """Tests for mimic_pipeline_zarr_to_anatomical_stub function."""

    def test_mimic_pipeline_stub_missing_import_process(self):
        processing_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }
        with pytest.raises(
            ValueError, match="Could not find zarr import process"
        ):
            pt.mimic_pipeline_zarr_to_anatomical_stub(
                "s3://bucket/session.ome.zarr", {}, processing_data
            )

    def test_mimic_pipeline_stub_missing_code_version(self):
        processing_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [{"name": "Image importing"}],
            }
        }
        with pytest.raises(ValueError, match="Pipeline version not found"):
            pt.mimic_pipeline_zarr_to_anatomical_stub(
                "s3://bucket/session.ome.zarr", {}, processing_data
            )


class TestPipelineTransforms:
    """Tests for pipeline_transforms function."""

    @pytest.fixture
    def sample_processing_data(self):
        return {
            "processing_pipeline": {
                "pipeline_version": "3.1.0",
                "data_processes": [
                    {
                        "name": "Image atlas alignment",
                        "notes": ATLAS_ALIGNMENT_NOTES,
                        "input_location": "s3://bucket/data/session.ome.zarr",
                    }
                ],
            }
        }

    def test_pipeline_transforms_success(self, sample_processing_data):
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"
        individual, template = pt.pipeline_transforms(
            zarr_uri, sample_processing_data
        )

        assert individual.base.endswith("image_atlas_alignment/session")
        assert individual.chain == pt._PIPELINE_INDIVIDUAL_TRANSFORMS[3]

        assert (
            template
            == pt._PIPELINE_TEMPLATE_TRANSFORMS[
                "SmartSPIM-template_2024-05-16_11-26-14"
            ]
        )

    def test_pipeline_transforms_missing_alignment_path(self):
        processing_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"

        with pytest.raises(
            ValueError, match="Could not determine image atlas alignment path"
        ):
            pt.pipeline_transforms(zarr_uri, processing_data)


class TestPipelinePointTransformsLocalPaths:
    """Tests for pipeline_point_transforms_local_paths function."""

    @pytest.fixture
    def sample_processing_data(self):
        return {
            "processing_pipeline": {
                "pipeline_version": "3.1.0",
                "data_processes": [
                    {
                        "name": "Image atlas alignment",
                        "notes": ATLAS_ALIGNMENT_NOTES,
                        "input_location": "s3://bucket/data/session.ome.zarr",
                    }
                ],
            }
        }

    def test_pipeline_point_transforms_local_paths(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        # Mock get_local_path_for_resource to return temporary paths
        def mock_get_local_path(uri, **kwargs):
            filename = Path(uri).name
            mock_path = tmp_path / filename
            mock_path.touch()

            class MockResult:
                def __init__(self, path):
                    self.path = path

            return MockResult(mock_path)

        with patch(
            "aind_zarr_utils.pipeline_transformed.get_local_path_for_resource",
            side_effect=mock_get_local_path,
        ):
            paths, inverted = pt.pipeline_point_transforms_local_paths(
                "s3://bucket/data/acquisition/session.ome.zarr/0",
                sample_processing_data,
                s3_client=mock_s3_client,
                cache_dir=tmp_path,
            )

        # Should have 4 transforms total (2 individual + 2 template)
        assert len(paths) == 4
        assert len(inverted) == 4

        # Check that all paths are strings
        for path in paths:
            assert isinstance(path, str)
            assert Path(path).exists()

        # Check inversion flags
        assert isinstance(inverted[0], bool)


class TestIndicesTransformations:
    """Tests for indices_to_ccf function (simplified)."""

    def test_indices_to_ccf_error_handling(self):
        """Test error handling when processing data is invalid."""
        annotation_indices = {"layer1": np.array([[10, 20, 30], [40, 50, 60]])}

        invalid_processing_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }

        with pytest.raises(
            ValueError, match="Could not find zarr import process"
        ):
            pt.indices_to_ccf(
                annotation_indices,
                "s3://bucket/session.ome.zarr",
                {},
                invalid_processing_data,
            )


class TestNeuroglancerToCCF:
    """Tests for neuroglancer_to_ccf function (simplified)."""

    def test_neuroglancer_to_ccf_error_handling(self):
        """Test error handling when processing data is invalid."""
        sample_neuroglancer_data = {
            "layers": [
                {
                    "name": "annotations",
                    "type": "annotation",
                    "annotations": [
                        {"point": [10, 20, 30, 0], "description": "point1"}
                    ],
                }
            ]
        }

        invalid_processing_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }

        with pytest.raises(
            ValueError, match="Could not find zarr import process"
        ):
            pt.neuroglancer_to_ccf(
                sample_neuroglancer_data,
                "s3://bucket/session.ome.zarr",
                {},
                invalid_processing_data,
            )


class TestIntegrationScenarios:
    """Integration tests combining multiple functions."""

    @pytest.fixture
    def complete_processing_data(self):
        return {
            "processing_pipeline": {
                "pipeline_version": "3.1.0",
                "data_processes": [
                    {
                        "name": "Image importing",
                        "code_version": "0.0.25",
                        "input_location": "s3://bucket/data/session.ome.zarr",
                    },
                    {
                        "name": "Image atlas alignment",
                        "notes": ATLAS_ALIGNMENT_NOTES,
                        "input_location": "s3://bucket/data/session.ome.zarr",
                    },
                ],
            }
        }

    def test_path_extraction_flow(self, complete_processing_data):
        """Test the path extraction flow without complex mocks."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"

        # Test asset path extraction
        asset_uri = pt._asset_from_zarr_any(zarr_uri)
        assert asset_uri == "s3://bucket/data"

        # Test alignment path resolution
        rel_path = pt.image_atlas_alignment_path_relative_from_processing(
            complete_processing_data
        )
        assert rel_path == "image_atlas_alignment/session/"

        # Test transform paths
        individual, template = pt.pipeline_transforms(
            zarr_uri, complete_processing_data
        )
        assert (
            individual.base == "s3://bucket/data/image_atlas_alignment/session"
        )

    def test_error_propagation(self):
        """Test that errors propagate correctly through the call chain."""
        # Test with missing processing data
        with pytest.raises(ValueError, match="Missing pipeline version"):
            pt._get_processing_pipeline_data({})

        # Test with incomplete processing data
        incomplete_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }

        with pytest.raises(
            ValueError, match="Could not find zarr import process"
        ):
            pt.mimic_pipeline_zarr_to_anatomical_stub(
                "s3://bucket/session.ome.zarr", {}, incomplete_data
            )
