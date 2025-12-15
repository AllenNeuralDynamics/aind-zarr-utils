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

    def test_template_transform_chains_separation(self):
        """Test that template transform chains are properly separated from
        template paths."""
        # Test that _PIPELINE_TEMPLATE_TRANSFORM_CHAINS exists and has expected
        # structure
        chains = pt._PIPELINE_TEMPLATE_TRANSFORM_CHAINS
        assert "SmartSPIM-template_2024-05-16_11-26-14" in chains

        chain = chains["SmartSPIM-template_2024-05-16_11-26-14"]
        assert isinstance(chain, pt.TransformChain)
        assert chain.fixed == "ccf"
        assert chain.moving == "template"

        # Test that _PIPELINE_TEMPLATE_TRANSFORMS references the chains
        templates = pt._PIPELINE_TEMPLATE_TRANSFORMS
        template = templates["SmartSPIM-template_2024-05-16_11-26-14"]
        assert template.chain == chain
        assert template.base.startswith("s3://")

    def test_transform_chain_reusability(self):
        """Test that transform chains can be reused with different bases."""
        # Get the standard template chain
        template_key = "SmartSPIM-template_2024-05-16_11-26-14"
        original_template = pt._PIPELINE_TEMPLATE_TRANSFORMS[template_key]
        original_chain = original_template.chain

        # Create a new TemplatePaths with the same chain but different base
        custom_base = "/local/templates/"
        custom_template = pt.TemplatePaths(
            base=custom_base, chain=original_chain
        )

        # Verify they share the same chain but have different bases
        assert custom_template.chain == original_chain
        assert custom_template.base == custom_base
        assert custom_template.base != original_template.base

        # Verify chain properties are the same
        assert (
            custom_template.chain.forward_chain == original_chain.forward_chain
        )
        assert (
            custom_template.chain.reverse_chain == original_chain.reverse_chain
        )

    def test_forward_vs_reverse_chain_differences(self):
        """Test that forward and reverse chains are different as expected."""
        # Test template chains
        template_chain = pt._PIPELINE_TEMPLATE_TRANSFORM_CHAINS[
            "SmartSPIM-template_2024-05-16_11-26-14"
        ]

        # Forward and reverse should have same files but different
        # order/inversion
        assert len(template_chain.forward_chain) == len(
            template_chain.reverse_chain
        )
        assert template_chain.forward_chain != template_chain.reverse_chain
        assert (
            template_chain.forward_chain_invert
            != template_chain.reverse_chain_invert
        )

        # Test individual chains
        individual_chain = pt._PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS[3]

        assert len(individual_chain.forward_chain) == len(
            individual_chain.reverse_chain
        )
        assert individual_chain.forward_chain != individual_chain.reverse_chain
        assert (
            individual_chain.forward_chain_invert
            != individual_chain.reverse_chain_invert
        )

    def test_chain_mathematical_consistency(self):
        """Test that chains have mathematically consistent inverse
        relationships."""
        # Test template chain inversion patterns
        template_chain = pt._PIPELINE_TEMPLATE_TRANSFORM_CHAINS[
            "SmartSPIM-template_2024-05-16_11-26-14"
        ]

        # Forward: [warp, affine] with [False, False] inversion
        # Reverse: [affine, inverse_warp] with [True, False] inversion
        assert "Warp" in template_chain.forward_chain[0]
        assert "Affine" in template_chain.forward_chain[1]
        assert "Affine" in template_chain.reverse_chain[0]
        assert "InverseWarp" in template_chain.reverse_chain[1]

        # Individual chain should follow same pattern
        individual_chain = pt._PIPELINE_INDIVIDUAL_TRANSFORM_CHAINS[3]

        assert "Warp" in individual_chain.forward_chain[0]
        assert "Affine" in individual_chain.forward_chain[1]
        assert "Affine" in individual_chain.reverse_chain[0]
        assert "InverseWarp" in individual_chain.reverse_chain[1]


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
        with pytest.raises(ValueError, match="Zarr import version not found"):
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

    def test_pipeline_transforms_with_template_base(
        self, sample_processing_data
    ):
        """Test pipeline_transforms with template_base parameter."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"
        custom_template_base = "/local/templates/"

        individual, template = pt.pipeline_transforms(
            zarr_uri,
            sample_processing_data,
            template_base=custom_template_base,
        )

        # Individual should still use the asset-based path
        assert individual.base.endswith("image_atlas_alignment/session")

        # Template should use custom base
        assert template.base == custom_template_base
        assert (
            template.chain
            == pt._PIPELINE_TEMPLATE_TRANSFORM_CHAINS[
                "SmartSPIM-template_2024-05-16_11-26-14"
            ]
        )

    def test_pipeline_transforms_template_used_parameter(
        self, sample_processing_data
    ):
        """Test pipeline_transforms with different template_used values."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"

        # Test with explicit template_used (should be same as default)
        individual, template = pt.pipeline_transforms(
            zarr_uri,
            sample_processing_data,
            template_used="SmartSPIM-template_2024-05-16_11-26-14",
        )

        assert (
            template
            == pt._PIPELINE_TEMPLATE_TRANSFORMS[
                "SmartSPIM-template_2024-05-16_11-26-14"
            ]
        )


class TestPipelineImageTransformsLocalPaths:
    """Tests for pipeline_image_transforms_local_paths function."""

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

    def test_pipeline_image_transforms_local_paths(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        """Test image transform local path resolution."""

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
            paths, inverted = pt.pipeline_image_transforms_local_paths(
                "s3://bucket/data/acquisition/session.ome.zarr/0",
                sample_processing_data,
                s3_client=mock_s3_client,
                cache_dir=tmp_path,
            )

        # Should have 4 transforms total (2 template + 2 individual)
        assert len(paths) == 4
        assert len(inverted) == 4

        # Check that all paths are strings
        for path in paths:
            assert isinstance(path, str)
            assert Path(path).exists()

        # Check inversion flags
        assert isinstance(inverted[0], bool)

    def test_pipeline_image_transforms_template_base_override(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        """Test custom template base path override."""
        custom_template_base = "/local/templates/"

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
            paths, inverted = pt.pipeline_image_transforms_local_paths(
                "s3://bucket/data/acquisition/session.ome.zarr/0",
                sample_processing_data,
                s3_client=mock_s3_client,
                cache_dir=tmp_path,
                template_base=custom_template_base,
            )

        # Should still have 4 transforms
        assert len(paths) == 4
        assert len(inverted) == 4

    def test_pipeline_image_transforms_forward_chain_usage(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        """Test that image transforms use forward chains."""
        paths_called = []

        def mock_get_local_path(uri, **kwargs):
            paths_called.append(uri)
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
            pt.pipeline_image_transforms_local_paths(
                "s3://bucket/data/acquisition/session.ome.zarr/0",
                sample_processing_data,
                s3_client=mock_s3_client,
                cache_dir=tmp_path,
            )

        # Verify forward chain files are requested
        # Template forward chain files
        assert any(
            "spim_template_to_ccf_syn_1Warp_25.nii.gz" in path
            for path in paths_called
        )
        assert any(
            "spim_template_to_ccf_syn_0GenericAffine_25.mat" in path
            for path in paths_called
        )

        # Individual forward chain files
        assert any(
            "ls_to_template_SyN_1Warp.nii.gz" in path for path in paths_called
        )
        assert any(
            "ls_to_template_SyN_0GenericAffine.mat" in path
            for path in paths_called
        )


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

    def test_pipeline_point_transforms_template_base_parameter(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        """Test that pipeline_point_transforms_local_paths accepts
        template_base."""
        custom_template_base = "/local/templates/"

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
                template_base=custom_template_base,
            )

        # Should work without errors and return proper structure
        assert len(paths) == 4
        assert len(inverted) == 4


class TestPipelineTransformsLocalPaths:
    """Tests for pipeline_transforms_local_paths function (combined
    transforms)."""

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

    def test_pipeline_transforms_local_paths_4tuple_return(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        """Test that pipeline_transforms_local_paths returns 4-tuple."""

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
            result = pt.pipeline_transforms_local_paths(
                "s3://bucket/data/acquisition/session.ome.zarr/0",
                sample_processing_data,
                s3_client=mock_s3_client,
                cache_dir=tmp_path,
            )

        # Should return 4-tuple: (pt_paths, pt_inverted, img_paths,
        # img_inverted)
        assert len(result) == 4
        pt_paths, pt_inverted, img_paths, img_inverted = result

        # Each should have 4 transforms (2 individual + 2 template)
        assert len(pt_paths) == 4
        assert len(pt_inverted) == 4
        assert len(img_paths) == 4
        assert len(img_inverted) == 4

        # All paths should be strings
        for path in pt_paths + img_paths:
            assert isinstance(path, str)
            assert Path(path).exists()

        # All inversion flags should be booleans
        for flag in pt_inverted + img_inverted:
            assert isinstance(flag, bool)

    def test_pipeline_transforms_local_paths_different_chains(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        """Test that point and image transforms use different chains."""
        paths_called = []

        def mock_get_local_path(uri, **kwargs):
            paths_called.append(uri)
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
            pt_paths, pt_inverted, img_paths, img_inverted = (
                pt.pipeline_transforms_local_paths(
                    "s3://bucket/data/acquisition/session.ome.zarr/0",
                    sample_processing_data,
                    s3_client=mock_s3_client,
                    cache_dir=tmp_path,
                )
            )

        # Point transforms should use reverse chains (inverse warp files)
        assert any("InverseWarp" in path for path in pt_paths)

        # Image transforms should use forward chains (regular warp files)
        assert any("1Warp.nii.gz" in path for path in img_paths)
        assert not any("InverseWarp" in path for path in img_paths)

    def test_pipeline_transforms_local_paths_template_base_forwarding(
        self, sample_processing_data, mock_s3_client, tmp_path
    ):
        """Test template_base parameter forwarding."""
        custom_template_base = "/local/templates/"

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
            result = pt.pipeline_transforms_local_paths(
                "s3://bucket/data/acquisition/session.ome.zarr/0",
                sample_processing_data,
                s3_client=mock_s3_client,
                cache_dir=tmp_path,
                template_base=custom_template_base,
            )

        # Should still return valid 4-tuple
        assert len(result) == 4
        pt_paths, pt_inverted, img_paths, img_inverted = result

        # All should have proper lengths
        for item in [pt_paths, pt_inverted, img_paths, img_inverted]:
            assert len(item) == 4


class TestTemplateBaseParameter:
    """Tests for template_base parameter functionality across functions."""

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

    def test_pipeline_transforms_template_base_override(
        self, sample_processing_data
    ):
        """Test pipeline_transforms with custom template_base."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"
        custom_template_base = "/local/templates/"

        individual, template = pt.pipeline_transforms(
            zarr_uri,
            sample_processing_data,
            template_base=custom_template_base,
        )

        # Individual should still point to the asset location
        assert individual.base.endswith("image_atlas_alignment/session")

        # Template should use custom base
        assert template.base == custom_template_base
        assert (
            template.chain
            == pt._PIPELINE_TEMPLATE_TRANSFORM_CHAINS[
                "SmartSPIM-template_2024-05-16_11-26-14"
            ]
        )

    def test_pipeline_transforms_template_base_none_fallback(
        self, sample_processing_data
    ):
        """Test pipeline_transforms fallback when template_base=None."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"

        individual, template = pt.pipeline_transforms(
            zarr_uri, sample_processing_data, template_base=None
        )

        # Template should use default from _PIPELINE_TEMPLATE_TRANSFORMS
        assert (
            template
            == pt._PIPELINE_TEMPLATE_TRANSFORMS[
                "SmartSPIM-template_2024-05-16_11-26-14"
            ]
        )
        assert template.base.startswith("s3://aind-open-data/")

    def test_indices_to_ccf_template_base_forwarding(
        self, mock_s3_client, tmp_path
    ):
        """Test that indices_to_ccf forwards template_base parameter."""
        annotation_indices = {"layer1": np.array([[10, 20, 30], [40, 50, 60]])}

        processing_data = {
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

        custom_template_base = "/local/templates/"

        # Mock all the underlying functions to avoid complex dependencies
        with (
            patch(
                "aind_zarr_utils.pipeline_transformed.mimic_pipeline_zarr_to_anatomical_stub"
            ) as mock_stub,
            patch(
                "aind_zarr_utils.pipeline_transformed.annotation_indices_to_anatomical"
            ) as mock_indices,
            patch(
                "aind_zarr_utils.pipeline_transformed.pipeline_point_transforms_local_paths"
            ) as mock_transforms,
            patch(
                "aind_zarr_utils.pipeline_transformed.apply_ants_transforms_to_point_arr"
            ) as mock_ants,
        ):
            # Set up mocks
            mock_stub.return_value = ("mock_stub", (100, 100, 100))
            mock_indices.return_value = {"layer1": np.array([[1.0, 2.0, 3.0]])}
            mock_transforms.return_value = (
                ["/path1", "/path2"],
                [False, True],
            )
            mock_ants.return_value = np.array([[10.0, 20.0, 30.0]])

            pt.indices_to_ccf(
                annotation_indices,
                "s3://bucket/session.ome.zarr",
                {},
                processing_data,
                template_base=custom_template_base,
            )

            # Verify template_base was forwarded to
            # pipeline_point_transforms_local_paths
            mock_transforms.assert_called_once()
            call_kwargs = mock_transforms.call_args[1]
            assert call_kwargs["template_base"] == custom_template_base

    def test_neuroglancer_to_ccf_template_base_forwarding(self):
        """Test that neuroglancer_to_ccf forwards template_base parameter."""
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

        processing_data = {
            "processing_pipeline": {
                "pipeline_version": "3.0.0",
                "data_processes": [
                    {
                        "name": "Image importing",
                        "code_version": "0.0.25",
                    }
                ],
            }
        }

        custom_template_base = "/local/templates/"

        # Mock underlying functions
        with (
            patch(
                "aind_zarr_utils.pipeline_transformed.neuroglancer_annotations_to_indices"
            ) as mock_ng_indices,
            patch(
                "aind_zarr_utils.pipeline_transformed.indices_to_ccf"
            ) as mock_indices_to_ccf,
        ):
            mock_ng_indices.return_value = (
                {"layer1": np.array([[1, 2, 3]])},
                None,
            )
            mock_indices_to_ccf.return_value = {
                "layer1": np.array([[10.0, 20.0, 30.0]])
            }

            pt.neuroglancer_to_ccf(
                sample_neuroglancer_data,
                "s3://bucket/session.ome.zarr",
                {},
                processing_data,
                template_base=custom_template_base,
            )

            # Verify template_base was forwarded to indices_to_ccf
            mock_indices_to_ccf.assert_called_once()
            call_kwargs = mock_indices_to_ccf.call_args[1]
            assert call_kwargs["template_base"] == custom_template_base


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

    def test_image_vs_point_transform_workflows(
        self,
        complete_processing_data,
        mock_transform_path_resolution,
        tmp_path,
    ):
        """Test complete workflows for both image and point transforms."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"

        # Test image transform workflow
        img_paths, img_inverted = pt.pipeline_image_transforms_local_paths(
            zarr_uri, complete_processing_data, cache_dir=tmp_path
        )

        # Test point transform workflow
        pt_paths, pt_inverted = pt.pipeline_point_transforms_local_paths(
            zarr_uri, complete_processing_data, cache_dir=tmp_path
        )

        # Both should have same number of transforms
        assert len(img_paths) == len(pt_paths) == 4

        # But should use different files (forward vs reverse chains)
        assert img_paths != pt_paths

        # Point transforms should include inverse warp files
        assert any("InverseWarp" in path for path in pt_paths)

        # Image transforms should not include inverse warp files
        assert not any("InverseWarp" in path for path in img_paths)

    def test_template_base_override_workflow(
        self,
        complete_processing_data,
        mock_template_base_paths,
        mock_transform_path_resolution,
        tmp_path,
    ):
        """Test end-to-end workflow with custom template base."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"
        custom_template_base = mock_template_base_paths["template_dir"]

        # Test with custom template base
        result = pt.pipeline_transforms_local_paths(
            zarr_uri,
            complete_processing_data,
            template_base=custom_template_base,
            cache_dir=tmp_path,
        )

        pt_paths, pt_inverted, img_paths, img_inverted = result

        # Should still get 4 transforms each
        assert len(pt_paths) == len(img_paths) == 4

        # All paths should exist
        for path in pt_paths + img_paths:
            assert Path(path).exists()

    def test_comprehensive_indices_to_ccf_workflow(
        self,
        comprehensive_processing_data,
        mock_template_base_paths,
        mock_transform_path_resolution,
        mock_overlay_selector,
        tmp_path,
    ):
        """Test complete indices_to_ccf workflow with template_base."""
        annotation_indices = {"layer1": np.array([[10, 20, 30], [40, 50, 60]])}
        zarr_uri = "s3://aind-open-data/SmartSPIM_776259_2025-06-26_20-44-52_stitched_2025-07-22_11-35-09/image_tile_fusing/OMEZarr/Ex_561_Em_600.zarr/"
        metadata = {"test": "metadata"}
        custom_template_base = mock_template_base_paths["template_dir"]

        # Mock the underlying components to focus on parameter flow
        with (
            patch(
                "aind_zarr_utils.pipeline_transformed.mimic_pipeline_zarr_to_anatomical_stub"
            ) as mock_stub,
            patch(
                "aind_zarr_utils.pipeline_transformed.annotation_indices_to_anatomical"
            ) as mock_indices,
            patch(
                "aind_zarr_utils.pipeline_transformed.apply_ants_transforms_to_point_arr"
            ) as mock_ants,
        ):
            # Set up mocks
            mock_stub.return_value = ("mock_stub", (100, 100, 100))
            mock_indices.return_value = {
                "layer1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            }
            mock_ants.return_value = np.array(
                [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]
            )

            result = pt.indices_to_ccf(
                annotation_indices,
                zarr_uri,
                metadata,
                comprehensive_processing_data,
                template_base=custom_template_base,
                cache_dir=tmp_path,
            )

            # Verify result structure
            assert "layer1" in result
            assert result["layer1"].shape == (2, 3)

            # Verify template_base was used in the pipeline
            mock_stub.assert_called_once()
            mock_indices.assert_called_once()
            mock_ants.assert_called_once()

    def test_neuroglancer_to_ccf_integration_workflow(
        self, comprehensive_processing_data, mock_template_base_paths, tmp_path
    ):
        """Test complete neuroglancer_to_ccf workflow integration."""
        neuroglancer_data = {
            "layers": [
                {
                    "name": "annotations",
                    "type": "annotation",
                    "annotations": [
                        {"point": [10, 20, 30, 0], "description": "point1"},
                        {"point": [40, 50, 60, 0], "description": "point2"},
                    ],
                }
            ]
        }

        zarr_uri = "s3://bucket/session.ome.zarr"
        metadata = {"test": "metadata"}
        custom_template_base = mock_template_base_paths["template_dir"]

        # Mock the underlying workflow components
        with (
            patch(
                "aind_zarr_utils.pipeline_transformed.neuroglancer_annotations_to_indices"
            ) as mock_ng_indices,
            patch(
                "aind_zarr_utils.pipeline_transformed.indices_to_ccf"
            ) as mock_indices_to_ccf,
        ):
            mock_ng_indices.return_value = (
                {"annotations": np.array([[1, 2, 3], [4, 5, 6]])},
                {"annotations": ["point1", "point2"]},
            )
            mock_indices_to_ccf.return_value = {
                "annotations": np.array(
                    [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]
                )
            }

            points_ccf, descriptions = pt.neuroglancer_to_ccf(
                neuroglancer_data,
                zarr_uri,
                metadata,
                comprehensive_processing_data,
                template_base=custom_template_base,
                cache_dir=tmp_path,
            )

            # Verify the workflow executed correctly
            assert "annotations" in points_ccf
            assert points_ccf["annotations"].shape == (2, 3)
            assert descriptions is not None
            assert "annotations" in descriptions

            # Verify template_base was forwarded
            mock_indices_to_ccf.assert_called_once()
            call_kwargs = mock_indices_to_ccf.call_args[1]
            assert call_kwargs["template_base"] == custom_template_base

    def test_helper_function_isolation(
        self,
        complete_processing_data,
        mock_transform_path_resolution,
        tmp_path,
    ):
        """Test that helper functions work independently and consistently."""
        zarr_uri = "s3://bucket/data/acquisition/session.ome.zarr/0"

        # Get transform paths from main function
        individual, template = pt.pipeline_transforms(
            zarr_uri, complete_processing_data
        )

        # Test helper functions directly
        img_paths, img_inverted = pt._pipeline_image_transforms_local_paths(
            individual, template, cache_dir=tmp_path
        )

        pt_paths, pt_inverted = pt._pipeline_point_transforms_local_paths(
            individual, template, cache_dir=tmp_path
        )

        # Should be consistent with public API results
        public_result = pt.pipeline_transforms_local_paths(
            zarr_uri, complete_processing_data, cache_dir=tmp_path
        )

        (
            public_pt_paths,
            public_pt_inverted,
            public_img_paths,
            public_img_inverted,
        ) = public_result

        # Helper results should match public API results
        assert pt_paths == public_pt_paths
        assert pt_inverted == public_pt_inverted
        assert img_paths == public_img_paths
        assert img_inverted == public_img_inverted
