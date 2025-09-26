"""Tests for pipeline_transformed module."""

import json
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
        transforms = pt._PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS
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
        assert individual.chain == pt._PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS[3]

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


class TestSWCDataTransformations:
    """Tests for SWC coordinate transformation functions."""

    def test_swc_data_to_zarr_indices_valid_input(
        self, sample_swc_data, mock_zarr_operations
    ):
        """Test basic SWC to zarr indices transformation."""
        zarr_uri = "/test/session.ome.zarr"

        result = pt.swc_data_to_zarr_indices(
            sample_swc_data,
            zarr_uri,
            swc_point_order="zyx",
            swc_point_units="micrometer",
        )

        # Should return same keys
        assert set(result.keys()) == set(sample_swc_data.keys())

        # Check that arrays are converted to integers
        for neuron_id, indices in result.items():
            assert indices.dtype == int
            assert indices.shape == sample_swc_data[neuron_id].shape

    def test_swc_data_to_zarr_indices_coordinate_orders(
        self, mock_zarr_operations
    ):
        """Test different coordinate order handling."""
        zarr_uri = "/test/session.ome.zarr"

        # Use simple data where coordinate differences are clear
        test_data = {
            "neuron_1": np.array(
                [[100.0, 200.0, 300.0]]
            )  # z!=x so reordering should be visible
        }

        # Test zyx order (default)
        result_zyx = pt.swc_data_to_zarr_indices(
            test_data, zarr_uri, swc_point_order="zyx"
        )

        # Test xyz order - should reorder the input coordinates
        result_xyz = pt.swc_data_to_zarr_indices(
            test_data, zarr_uri, swc_point_order="xyz"
        )

        # Both should succeed and return same shape
        assert result_zyx["neuron_1"].shape == result_xyz["neuron_1"].shape
        assert result_zyx["neuron_1"].dtype == int
        assert result_xyz["neuron_1"].dtype == int

    def test_swc_data_to_zarr_indices_unit_conversion(
        self, sample_swc_data, mock_zarr_operations
    ):
        """Test unit conversion between micrometer and millimeter."""
        zarr_uri = "/test/session.ome.zarr"

        # Test micrometer (should scale by 1000)
        result_micro = pt.swc_data_to_zarr_indices(
            sample_swc_data, zarr_uri, swc_point_units="micrometer"
        )

        # Test millimeter (no scaling)
        result_milli = pt.swc_data_to_zarr_indices(
            sample_swc_data, zarr_uri, swc_point_units="millimeter"
        )

        # Results should be different (unit conversion affects scaling)
        for neuron_id in sample_swc_data.keys():
            assert not np.array_equal(
                result_micro[neuron_id], result_milli[neuron_id]
            )

    def test_swc_data_to_zarr_indices_invalid_shapes(
        self, invalid_swc_data, mock_zarr_operations
    ):
        """Test error handling for malformed arrays."""
        zarr_uri = "/test/session.ome.zarr"

        # Test 1D array
        with pytest.raises(ValueError, match="Expected \\(N, 3\\) array"):
            pt.swc_data_to_zarr_indices(
                {"bad": invalid_swc_data["bad_shape_1d"]}, zarr_uri
            )

        # Test wrong number of columns
        with pytest.raises(ValueError, match="Expected \\(N, 3\\) array"):
            pt.swc_data_to_zarr_indices(
                {"bad": invalid_swc_data["wrong_cols"]}, zarr_uri
            )

    def test_swc_data_to_ccf_full_pipeline(
        self, sample_swc_data, mock_processing_data, mock_zarr_operations
    ):
        """Test end-to-end SWC to CCF transformation."""
        zarr_uri = "/test/session.ome.zarr"
        metadata = {"test": "metadata"}

        # Mock indices_to_ccf to return predictable results
        def mock_indices_to_ccf(indices, *args, **kwargs):
            return {k: v + 10 for k, v in indices.items()}

        with patch(
            "aind_zarr_utils.pipeline_transformed.indices_to_ccf",
            side_effect=mock_indices_to_ccf,
        ):
            result = pt.swc_data_to_ccf(
                sample_swc_data, zarr_uri, metadata, mock_processing_data
            )

        # Should return transformed coordinates
        assert set(result.keys()) == set(sample_swc_data.keys())
        for neuron_id in sample_swc_data.keys():
            assert result[neuron_id].shape == sample_swc_data[neuron_id].shape

    def test_swc_data_to_ccf_error_propagation(
        self, sample_swc_data, mock_zarr_operations
    ):
        """Test error handling from underlying functions."""
        zarr_uri = "/test/session.ome.zarr"
        metadata = {}

        invalid_processing_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [],
            }
        }

        with pytest.raises(
            ValueError, match="Could not find zarr import process"
        ):
            pt.swc_data_to_ccf(
                sample_swc_data, zarr_uri, metadata, invalid_processing_data
            )

    def test_alignment_zarr_uri_and_metadata_resolution(self, tmp_path):
        """Test URI and metadata resolution from asset paths."""
        # Create mock metadata files
        asset_dir = tmp_path / "asset"
        asset_dir.mkdir()

        zarr_dir = asset_dir / "acquisition" / "session.ome.zarr"
        zarr_dir.mkdir(parents=True)

        metadata_file = asset_dir / "metadata.nd.json"
        processing_file = asset_dir / "processing.json"

        metadata_file.write_text('{"test": "metadata"}')
        processing_content = {
            "processing_pipeline": {
                "pipeline_version": "3.1.0",
                "data_processes": [
                    {
                        "name": "Image atlas alignment",
                        "notes": ATLAS_ALIGNMENT_NOTES,
                        "input_location": str(zarr_dir),
                    }
                ],
            }
        }
        processing_file.write_text(json.dumps(processing_content))

        # Test asset URI resolution
        result = (
            pt.alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
                asset_uri=str(asset_dir)
            )
        )

        assert len(result) == 3  # zarr_uri, metadata, processing_data
        zarr_uri, metadata, processing_data = result

        assert "session.zarr" in zarr_uri
        assert metadata["test"] == "metadata"
        assert (
            processing_data["processing_pipeline"]["pipeline_version"]
            == "3.1.0"
        )

    @pytest.mark.parametrize("coordinate_order", ["zyx", "xyz", "yxz"])
    def test_swc_coordinate_order_parameter(
        self, sample_swc_data, mock_zarr_operations, coordinate_order
    ):
        """Test different coordinate order parameters."""
        zarr_uri = "/test/session.ome.zarr"

        result = pt.swc_data_to_zarr_indices(
            sample_swc_data, zarr_uri, swc_point_order=coordinate_order
        )

        # Should succeed for all valid coordinate orders
        assert set(result.keys()) == set(sample_swc_data.keys())
        for neuron_id, indices in result.items():
            assert indices.dtype == int
            assert indices.shape == sample_swc_data[neuron_id].shape

    @pytest.mark.parametrize("units", ["micrometer", "millimeter"])
    def test_swc_unit_parameter(
        self, sample_swc_data, mock_zarr_operations, units
    ):
        """Test different unit parameters."""
        zarr_uri = "/test/session.ome.zarr"

        result = pt.swc_data_to_zarr_indices(
            sample_swc_data, zarr_uri, swc_point_units=units
        )

        # Should succeed for all valid units
        assert set(result.keys()) == set(sample_swc_data.keys())
        for neuron_id, indices in result.items():
            assert indices.dtype == int
            assert indices.shape == sample_swc_data[neuron_id].shape

    def test_swc_data_to_ccf_auto_metadata_missing_files(
        self, sample_swc_data, tmp_path
    ):
        """Test error handling when metadata files are missing."""
        asset_dir = tmp_path / "asset"
        asset_dir.mkdir()

        # Missing metadata files should raise errors
        with pytest.raises((FileNotFoundError, ValueError)):
            pt.swc_data_to_ccf_auto_metadata(sample_swc_data, str(asset_dir))

    def test_swc_empty_data_handling(self, mock_zarr_operations):
        """Test handling of empty SWC data."""
        zarr_uri = "/test/session.ome.zarr"
        empty_data = {}

        result = pt.swc_data_to_zarr_indices(empty_data, zarr_uri)
        assert result == {}

    def test_swc_single_point_neuron(self, mock_zarr_operations):
        """Test handling of neurons with single points."""
        zarr_uri = "/test/session.ome.zarr"
        single_point_data = {
            "neuron_single": np.array([[100.0, 200.0, 300.0]])
        }

        result = pt.swc_data_to_zarr_indices(single_point_data, zarr_uri)

        assert "neuron_single" in result
        assert result["neuron_single"].shape == (1, 3)
        assert result["neuron_single"].dtype == int

    def test_swc_data_to_ccf_kwargs_forwarding(
        self, sample_swc_data, mock_processing_data, mock_zarr_operations
    ):
        """Test that kwargs are properly forwarded to indices_to_ccf."""
        zarr_uri = "/test/session.ome.zarr"
        metadata = {"test": "metadata"}

        # Mock indices_to_ccf to capture kwargs
        def mock_indices_to_ccf(indices, *args, **kwargs):
            # Verify our custom kwarg was passed through
            assert "test_kwarg" in kwargs
            assert kwargs["test_kwarg"] == "test_value"
            return {k: v + 10 for k, v in indices.items()}

        with patch(
            "aind_zarr_utils.pipeline_transformed.indices_to_ccf",
            side_effect=mock_indices_to_ccf,
        ):
            pt.swc_data_to_ccf(
                sample_swc_data,
                zarr_uri,
                metadata,
                mock_processing_data,
                test_kwarg="test_value",
            )

    def test_swc_data_to_ccf_auto_metadata_integration(
        self, sample_swc_data, tmp_path
    ):
        """Test end-to-end auto metadata SWC to CCF transformation."""
        # Create mock asset structure
        asset_dir = tmp_path / "asset"
        asset_dir.mkdir()

        zarr_dir = asset_dir / "acquisition" / "session.ome.zarr"
        zarr_dir.mkdir(parents=True)

        # Create minimal valid metadata files
        metadata_file = asset_dir / "metadata.nd.json"
        processing_file = asset_dir / "processing.json"

        metadata_content = {
            "acquisition": {
                "axes": [{"name": "Z"}, {"name": "Y"}, {"name": "X"}]
            }
        }
        processing_content = {
            "processing_pipeline": {
                "pipeline_version": "3.1.0",
                "data_processes": [
                    {
                        "name": "Image importing",
                        "code_version": "0.0.25",
                        "input_location": str(zarr_dir),
                    },
                    {
                        "name": "Image atlas alignment",
                        "notes": ATLAS_ALIGNMENT_NOTES,
                        "input_location": str(zarr_dir),
                    },
                ],
            }
        }

        metadata_file.write_text(json.dumps(metadata_content))
        processing_file.write_text(json.dumps(processing_content))

        # Mock the underlying transformation functions
        def mock_swc_data_to_ccf(*args, **kwargs):
            return {k: v + 100 for k, v in sample_swc_data.items()}

        with patch(
            "aind_zarr_utils.pipeline_transformed.swc_data_to_ccf",
            side_effect=mock_swc_data_to_ccf,
        ):
            result = pt.swc_data_to_ccf_auto_metadata(
                sample_swc_data, str(asset_dir)
            )

        # Verify the transformation was applied
        assert set(result.keys()) == set(sample_swc_data.keys())
        for neuron_id in sample_swc_data.keys():
            assert result[neuron_id].shape == sample_swc_data[neuron_id].shape


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
