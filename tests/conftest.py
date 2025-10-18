"""
Shared testing infrastructure for aind-zarr-utils.

This module provides unified mock objects and fixtures to be used across
all test modules, reducing duplication and ensuring consistent behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import SimpleITK as sitk
import zarr
from botocore.exceptions import ClientError

# ============================================================================
# S3 Infrastructure
# ============================================================================


class UnifiedS3Client:
    """
    Comprehensive S3 client mock supporting all operations across modules.

    Extends the original DummyS3Client pattern to support:
    - Basic operations (get_object, list_objects)
    - Advanced operations (head_object, download_file)
    - Error simulation (ClientError with various HTTP codes)
    - Range requests for s3_cache peek operations
    """

    def __init__(
        self,
        data: dict = None,
        *,
        etag: str = "mock-etag-12345",
        content_length: int = 1024,
        simulate_head_blocked: bool = False,
        simulate_errors: dict = None,
    ):
        self.data = data or {}
        self.etag = etag
        self.content_length = content_length
        self.simulate_head_blocked = simulate_head_blocked
        self.simulate_errors = simulate_errors or {}

        # Track downloads for validation
        self.downloads = []

    # ---- Core S3 Operations (from original DummyS3Client) ----

    def get_object(self, Bucket: str, Key: str, Range: str = None):
        """Mock S3 get_object with optional Range support for s3_cache."""
        if f"{Bucket}/{Key}" in self.simulate_errors:
            error_code = self.simulate_errors[f"{Bucket}/{Key}"]
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": error_code}},
                "GetObject",
            )

        headers = {"etag": f'"{self.etag}"'}

        # Handle range requests for s3_cache peek operations
        if Range and Range.startswith("bytes="):
            # Parse range like "bytes=0-0"
            range_match = re.match(r"bytes=(\d+)-(\d+)", Range)
            if range_match:
                start, end = range_match.groups()
                headers["content-range"] = (
                    f"bytes {start}-{end}/{self.content_length}"
                )

        return {
            "Body": self,
            "ETag": f'"{self.etag}"',
            "ResponseMetadata": {"HTTPHeaders": headers},
        }

    def head_object(self, Bucket: str, Key: str):
        """Mock S3 head_object with optional blocking simulation."""
        if self.simulate_head_blocked:
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 403}}, "HeadObject"
            )

        if f"{Bucket}/{Key}" in self.simulate_errors:
            error_code = self.simulate_errors[f"{Bucket}/{Key}"]
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": error_code}},
                "HeadObject",
            )

        return {
            "ETag": f'"{self.etag}"',
            "ContentLength": self.content_length,
        }

    def download_file(self, Bucket: str, Key: str, Filename: str, Config=None):
        """Mock S3 download_file for s3_cache testing."""
        self.downloads.append((Bucket, Key, Filename))

        if f"{Bucket}/{Key}" in self.simulate_errors:
            error_code = self.simulate_errors[f"{Bucket}/{Key}"]
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": error_code}},
                "GetObject",
            )

        # Create mock file content
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        with open(Filename, "w") as f:
            f.write(json.dumps(self.data))

    # ---- File-like interface for Body operations ----

    def read(self, *args, **kwargs):
        return json.dumps(self.data).encode()

    def __iter__(self):
        return iter([json.dumps(self.data).encode()])

    def __next__(self):
        raise StopIteration

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def readlines(self):
        return [json.dumps(self.data).encode()]

    def readline(self):
        return json.dumps(self.data).encode()

    def seek(self, *args, **kwargs):
        pass

    def tell(self):
        return 0

    # ---- Dict-like interface compatibility ----

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __bool__(self):
        return True

    def __len__(self):
        return 1

    def __contains__(self, item):
        return item in self.data

    def __eq__(self, other):
        if hasattr(other, "data"):
            return self.data == other.data
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.data))

    def __call__(self, *args, **kwargs):
        return self.data

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __setattr__(self, key, value):
        if key in (
            "data",
            "etag",
            "content_length",
            "simulate_head_blocked",
            "simulate_errors",
            "downloads",
        ):
            object.__setattr__(self, key, value)
        else:
            setattr(self.data, key, value)


class DummyResponse:
    """HTTP response mock for requests operations."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")


@pytest.fixture
def mock_s3_client():
    """Provide a unified S3 client mock for all tests."""
    return UnifiedS3Client()


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for URL-based JSON fetching."""

    def _mock_get(url, **kwargs):
        # Default response data
        return DummyResponse({"mocked": "data", "url": url})

    return _mock_get


@pytest.fixture
def mock_overlay_selector(monkeypatch):
    """Mock OverlaySelector for pipeline_transformed tests."""
    from unittest.mock import Mock

    # Mock the get_selector function to return our custom mock
    selector = Mock()
    selector.select = Mock(return_value=[])  # Return empty list of overlays

    # Mock the apply_overlays function to bypass overlay logic entirely
    def mock_apply_overlays(header, overlays, meta, multiscale_no):
        return header, []  # Return header unchanged with no applied overlays

    monkeypatch.setattr(
        "aind_zarr_utils.pipeline_transformed.apply_overlays",
        mock_apply_overlays,
    )

    return selector


# ============================================================================
# Zarr Infrastructure
# ============================================================================


class MockZarrData:
    """Mock zarr data array with compute capability."""

    def __init__(self, shape: tuple[int, ...] = (10, 10, 10)):
        self.shape = shape

    def compute(self) -> np.ndarray:
        """Return mock numpy array data."""
        return np.ones(self.shape, dtype=np.float32)


class UnifiedZarrNode:
    """
    Comprehensive zarr node mock supporting all operations across modules.

    Supports:
    - Multi-level data access (for different resolution levels)
    - Realistic metadata structures
    - Coordinate transformations
    - Axis information
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (1, 1, 10, 10, 10),
        levels: int = 4,
        metadata: dict = None,
        coordinate_transforms: list = None,
        axes: list = None,
    ):
        # Create multi-level data structure
        self.data = {}
        for level in range(levels):
            scale_factor = 2**level
            scaled_shape = tuple(
                max(1, s // scale_factor)
                if i >= 2
                else s  # Only scale spatial dims
                for i, s in enumerate(shape)
            )
            self.data[level] = MockZarrData(scaled_shape)

        # Set up metadata
        if coordinate_transforms is None:
            coordinate_transforms = []
            for level in range(levels):
                scale = [1.0, 1.0] + [2.0**level] * (
                    len(shape) - 2
                )  # Scale spatial dims
                coordinate_transforms.append([{"scale": scale}])

        if axes is None:
            axes = [
                {"name": "t", "unit": "second"},
                {"name": "c", "unit": ""},
                {"name": "z", "unit": "millimeter"},
                {"name": "y", "unit": "millimeter"},
                {"name": "x", "unit": "millimeter"},
            ][: len(shape)]

        self.metadata = metadata or {
            "coordinateTransformations": coordinate_transforms,
            "axes": axes,
        }


class MockZarrReader:
    """Mock zarr Reader for opening zarr stores."""

    def __init__(self, node: UnifiedZarrNode):
        self.node = node

    def __call__(self, *args, **kwargs):
        """Return list containing the zarr node."""
        return [self.node]


@pytest.fixture
def mock_zarr_node():
    """Provide a basic zarr node for testing."""
    return UnifiedZarrNode()


@pytest.fixture
def mock_zarr_operations(monkeypatch):
    """Mock zarr operations consistently across modules."""

    def mock_open_zarr(uri):
        node = UnifiedZarrNode()
        return node, node.metadata

    def mock_parse_url(uri):
        return uri  # Simple pass-through for testing

    def mock_reader(uri):
        return MockZarrReader(UnifiedZarrNode())

    monkeypatch.setattr("aind_zarr_utils.zarr._open_zarr", mock_open_zarr)
    monkeypatch.setattr("aind_zarr_utils.zarr.parse_url", mock_parse_url)
    monkeypatch.setattr("aind_zarr_utils.zarr.Reader", mock_reader)

    return {
        "open_zarr": mock_open_zarr,
        "parse_url": mock_parse_url,
        "reader": mock_reader,
    }


@pytest.fixture
def real_ome_zarr(tmp_path):
    """
    Create a real OME-ZARR file with proper structure for testing.

    Creates a multi-scale zarr with:
    - 4 resolution levels (0, 1, 2, 3)
    - 5D data: (t=1, c=1, z=10, y=10, x=10) at level 0
    - Proper OME-ZARR v0.4 metadata with axes and coordinate transformations
    - Real numpy data that can be read

    Returns the path to the zarr store.
    """
    zarr_path = tmp_path / "test.ome.zarr"

    # Open zarr group (zarr v3 API)
    root = zarr.open_group(str(zarr_path), mode="w")

    # Create multiscale data with 4 levels
    levels = 4
    base_shape = (1, 1, 10, 10, 10)  # t, c, z, y, x

    # Prepare multiscale metadata
    datasets = []
    for level in range(levels):
        # Calculate scaled shape (only scale spatial dimensions z, y, x)
        scale_factor = 2**level
        scaled_shape = (
            base_shape[0],  # t unchanged
            base_shape[1],  # c unchanged
            max(1, base_shape[2] // scale_factor),  # z
            max(1, base_shape[3] // scale_factor),  # y
            max(1, base_shape[4] // scale_factor),  # x
        )

        # Create array with incrementing values for validation
        data = np.arange(np.prod(scaled_shape), dtype=np.float32).reshape(
            scaled_shape
        )

        # Write to zarr (zarr v3 API)
        root.create_array(name=str(level), data=data, chunks=(1, 1, 5, 5, 5))

        # Add to multiscale datasets metadata
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,  # t
                            1.0,  # c
                            1.0 * scale_factor,  # z spacing in mm
                            1.0 * scale_factor,  # y spacing in mm
                            1.0 * scale_factor,  # x spacing in mm
                        ],
                    }
                ],
            }
        )

    # Set OME-ZARR multiscales metadata
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "test",
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "millimeter"},
                {"name": "y", "type": "space", "unit": "millimeter"},
                {"name": "x", "type": "space", "unit": "millimeter"},
            ],
            "datasets": datasets,
            "coordinateTransformations": [{"type": "identity"}],
        }
    ]

    return str(zarr_path)


# ============================================================================
# Processing Metadata Factories
# ============================================================================


def create_processing_metadata(
    version: str = "3.1.0",
    include_processes: list[dict] = None,
    include_zarr_import: bool = True,
    include_atlas_alignment: bool = True,
) -> dict[str, Any]:
    """
    Factory for realistic processing.json structures.

    Parameters
    ----------
    version : str
        Pipeline version string (e.g., "3.1.0")
    include_processes : list[dict], optional
        Custom process list; if None, generates standard processes
    include_zarr_import : bool
        Whether to include "Image importing" process
    include_atlas_alignment : bool
        Whether to include "Image atlas alignment" process

    Returns
    -------
    dict
        Mock processing metadata matching real structure
    """
    if include_processes is not None:
        processes = include_processes
    else:
        processes = []

        if include_zarr_import:
            processes.append(
                {
                    "name": "Image importing",
                    "code_version": version,
                    "parameters": {"some": "param"},
                    "start_date_time": "2024-01-01T00:00:00",
                    "end_date_time": "2024-01-01T01:00:00",
                }
            )

        if include_atlas_alignment:
            processes.append(
                {
                    "name": "Image atlas alignment",
                    "notes": (
                        "Template based registration: LS -> template -> "
                        "Allen CCFv3 Atlas"
                    ),
                    "input_location": "/some/path/Ex_639_Em_667.ome.zarr",
                    "parameters": {
                        "template": "SmartSPIM-template_2024-05-16_11-26-14"
                    },
                    "start_date_time": "2024-01-01T01:00:00",
                    "end_date_time": "2024-01-01T02:00:00",
                }
            )

    return {
        "processing_pipeline": {
            "pipeline_version": version,
            "data_processes": processes,
            "processor_full_name": "SmartSPIM Pipeline",
            "pipeline_url": "https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        },
        "processing_date": "2024-01-01",
        "pipeline_version": version,  # Also at top level sometimes
    }


def create_nd_metadata(
    axes: list[dict] = None,
    subject_id: str = "123456",
    session_name: str = "SmartSPIM_123456_2024-01-01_15-30-00",
) -> dict[str, Any]:
    """
    Factory for realistic metadata.nd.json structures.

    Parameters
    ----------
    axes : list[dict], optional
        Custom axis definitions; if None, uses standard 3D spatial
    subject_id : str
        Subject identifier
    session_name : str
        Session name

    Returns
    -------
    dict
        Mock ND metadata matching real structure
    """
    if axes is None:
        axes = [
            {"dimension": "2", "name": "Z", "direction": "INFERIOR_SUPERIOR"},
            {"dimension": "3", "name": "Y", "direction": "POSTERIOR_ANTERIOR"},
            {"dimension": "4", "name": "X", "direction": "LEFT_RIGHT"},
        ]

    return {
        "acquisition": {
            "axes": axes,
            "chamber_immersion": {"medium": "air"},
            "tiles": [],  # Simplified
        },
        "subject": {
            "subject_id": subject_id,
            "species": {"name": "Mus musculus"},
        },
        "session": {
            "session_name": session_name,
            "session_start_time": "2024-01-01T15:30:00",
        },
        "procedures": [],
        "instrument": {"instrument_id": "SmartSPIM.1"},
        "acq_date": "2024-01-01",  # Used by overlay selectors
    }


@pytest.fixture
def mock_processing_data():
    """Provide standard mock processing metadata."""
    return create_processing_metadata()


@pytest.fixture
def mock_nd_metadata():
    """Provide standard mock ND metadata."""
    return create_nd_metadata()


# ============================================================================
# SWC Testing Infrastructure
# ============================================================================


@pytest.fixture
def sample_swc_data():
    """Realistic SWC coordinate data for testing."""
    return {
        "neuron_001": np.array([[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]]),
        "neuron_002": np.array([[50.0, 150.0, 250.0]]),
        "neuron_003": np.array(
            [[10.5, 20.5, 30.5], [15.5, 25.5, 35.5], [20.5, 30.5, 40.5]]
        ),
    }


@pytest.fixture
def invalid_swc_data():
    """Invalid SWC data for error testing."""
    return {
        "bad_shape_1d": np.array([1, 2, 3]),  # 1D instead of 2D
        "wrong_cols": np.array([[1, 2]]),  # 2 cols instead of 3
        "wrong_cols_4": np.array([[1, 2, 3, 4]]),  # 4 cols instead of 3
        "empty": np.array([]).reshape(0, 3),  # Empty array with correct shape
        "non_numeric": "not_an_array",  # Wrong type
    }


# ============================================================================
# Neuroglancer Testing Infrastructure
# ============================================================================


@pytest.fixture
def neuroglancer_test_data():
    """
    Provide realistic neuroglancer JSON data for testing.

    Returns a neuroglancer state with annotation layers and dimension
    information.
    """
    return {
        "dimensions": {
            "z": (1.0, "millimeter"),
            "y": (2.0, "millimeter"),
            "x": (3.0, "millimeter"),
        },
        "layers": [
            {
                "name": "annotations_layer1",
                "type": "annotation",
                "annotations": [
                    {
                        "point": [10.0, 20.0, 30.0, 40.0],
                        "description": "point1",
                    },
                    {
                        "point": [15.0, 25.0, 35.0, 45.0],
                        "description": "point2",
                    },
                ],
            },
            {
                "name": "image_layer",
                "type": "image",
                "source": "zarr://s3://bucket/data.zarr",
            },
            {
                "name": "annotations_layer2",
                "type": "annotation",
                "annotations": [
                    {
                        "point": [5.0, 10.0, 15.0, 20.0],
                        "description": "point3",
                    },
                ],
            },
        ],
    }


# ============================================================================
# Annotation Testing Infrastructure
# ============================================================================


@pytest.fixture
def mock_annotation_functions(monkeypatch):
    """Mock annotation processing functions for neuroglancer tests."""

    def mock_annotation_indices_to_anatomical(stub_img, annotations):
        """Mock coordinate transformation - just add 1 to all values."""
        return {k: v + 1 for k, v in annotations.items()}

    def mock_zarr_to_sitk_stub(zarr_uri, metadata, **kwargs):
        """Mock zarr to sitk stub conversion using real SimpleITK."""
        # Create a real SimpleITK image stub
        stub_img = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        stub_img.SetSpacing((1.0, 1.0, 1.0))
        stub_img.SetOrigin((0.0, 0.0, 0.0))
        stub_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        return stub_img, (10, 10, 10)

    monkeypatch.setattr(
        "aind_zarr_utils.neuroglancer.annotation_indices_to_anatomical",
        mock_annotation_indices_to_anatomical,
    )
    monkeypatch.setattr(
        "aind_zarr_utils.neuroglancer.zarr_to_sitk_stub",
        mock_zarr_to_sitk_stub,
    )

    return {
        "indices_to_anatomical": mock_annotation_indices_to_anatomical,
        "zarr_to_sitk_stub": mock_zarr_to_sitk_stub,
    }


# ============================================================================
# Pipeline Transform Testing Infrastructure
# ============================================================================


@pytest.fixture
def mock_template_base_paths(tmp_path):
    """Provide mock local template paths for testing template_base
    functionality."""
    template_dir = tmp_path / "local_templates"
    template_dir.mkdir()

    # Create mock template transform files
    template_files = [
        "spim_template_to_ccf_syn_1Warp_25.nii.gz",
        "spim_template_to_ccf_syn_0GenericAffine_25.mat",
        "spim_template_to_ccf_syn_1InverseWarp_25.nii.gz",
    ]

    for filename in template_files:
        (template_dir / filename).touch()

    return {
        "template_dir": str(template_dir),
        "template_files": template_files,
    }


@pytest.fixture
def mock_transform_path_resolution(monkeypatch):
    """Mock get_local_path_for_resource for transform path testing."""
    from pathlib import Path

    def mock_get_local_path(
        uri, s3_client=None, anonymous=False, cache_dir=None
    ):
        """Mock function that returns a predictable local path."""
        # Extract filename from URI
        filename = Path(uri).name

        # Create a mock path in cache_dir if provided, otherwise use temp
        if cache_dir:
            mock_path = Path(cache_dir) / filename
        else:
            mock_path = Path("/tmp") / filename

        # Ensure parent directory exists
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()  # Create the file

        class MockResult:
            def __init__(self, path):
                self.path = str(path)

        return MockResult(mock_path)

    monkeypatch.setattr(
        "aind_zarr_utils.pipeline_transformed.get_local_path_for_resource",
        mock_get_local_path,
    )

    return mock_get_local_path


@pytest.fixture
def mock_ants_transforms(monkeypatch):
    """Mock ANTs transform application for pipeline testing."""

    def mock_apply_ants_transforms(points, transform_list, whichtoinvert):
        """Mock ANTs transform application - just add 100 to coordinates."""
        return points + 100

    monkeypatch.setattr(
        "aind_zarr_utils.pipeline_transformed.apply_ants_transforms_to_point_arr",
        mock_apply_ants_transforms,
    )

    return mock_apply_ants_transforms


def create_comprehensive_processing_data(
    version: str = "3.1.0",
    include_image_importing: bool = True,
    include_atlas_alignment: bool = True,
    input_zarr_name: str = "session.ome.zarr",
) -> dict[str, Any]:
    """
    Factory for comprehensive processing metadata with both importing and
    alignment.

    This extends the base create_processing_metadata to ensure both processes
    are present for testing pipeline transform functionality.

    Parameters
    ----------
    version : str
        Pipeline version
    include_image_importing : bool
        Include Image importing process
    include_atlas_alignment : bool
        Include Image atlas alignment process
    input_zarr_name : str
        Name of the input zarr file (affects channel inference)

    Returns
    -------
    dict
        Complete processing metadata suitable for pipeline transform testing
    """
    processes = []

    if include_image_importing:
        processes.append(
            {
                "name": "Image importing",
                "code_version": version,
                "input_location": f"s3://bucket/data/{input_zarr_name}",
                "parameters": {"import_type": "zarr"},
            }
        )

    if include_atlas_alignment:
        processes.append(
            {
                "name": "Image atlas alignment",
                "notes": (
                    "Template based registration: LS -> template -> Allen "
                    "CCFv3 Atlas"
                ),
                "input_location": f"s3://bucket/data/{input_zarr_name}",
                "parameters": {
                    "template": "SmartSPIM-template_2024-05-16_11-26-14"
                },
            }
        )

    return create_processing_metadata(
        version=version,
        include_processes=processes,
        include_zarr_import=False,  # We're providing our own
        include_atlas_alignment=False,  # We're providing our own
    )


@pytest.fixture
def comprehensive_processing_data():
    """Processing data with both importing and alignment processes."""
    return create_comprehensive_processing_data()
