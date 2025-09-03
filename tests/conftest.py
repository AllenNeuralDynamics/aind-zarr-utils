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


# ============================================================================
# SimpleITK Infrastructure
# ============================================================================


class UnifiedSitkImage:
    """
    Comprehensive SimpleITK Image mock for all modules.

    Supports all Image operations needed across:
    - zarr.py: Basic image properties and stubs
    - pipeline_domain_selector.py: Header manipulation
    - annotations.py: Coordinate transformations
    """

    def __init__(
        self,
        size: tuple[int, int, int] = (10, 10, 10),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        direction: tuple[float, ...] = None,
        pixel_type: str = "uint8",
    ):
        self._size = size
        self._origin = origin
        self._spacing = spacing
        # Use identity matrix as default direction (cardinal)
        self._direction = direction or (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self.pixel_type = pixel_type

    # ---- Property getters ----

    def GetSize(self) -> tuple[int, int, int]:
        return self._size

    def GetOrigin(self) -> tuple[float, float, float]:
        return self._origin

    def GetSpacing(self) -> tuple[float, float, float]:
        return self._spacing

    def GetDirection(self) -> tuple[float, ...]:
        return self._direction

    # ---- Property setters ----

    def SetOrigin(self, origin: tuple[float, float, float]) -> None:
        self._origin = tuple(origin)

    def SetSpacing(self, spacing: tuple[float, float, float]) -> None:
        self._spacing = tuple(spacing)

    def SetDirection(self, direction: tuple[float, ...]) -> None:
        self._direction = tuple(direction)


class MockSitkModule:
    """Mock SimpleITK module with all needed classes and constants."""

    # Pixel type constants
    sitkUInt8 = "uint8"
    sitkFloat32 = "float32"
    sitkFloat64 = "float64"

    @staticmethod
    def Image(size: tuple[int, int, int], pixel_type: str) -> UnifiedSitkImage:
        """Create a new mock SimpleITK Image."""
        return UnifiedSitkImage(size=size, pixel_type=pixel_type)

    @staticmethod
    def GetImageFromArray(array: np.ndarray) -> UnifiedSitkImage:
        """Create mock Image from numpy array."""
        shape = array.shape if hasattr(array, "shape") else (10, 10, 10)
        # SimpleITK reverses array dimensions
        size = tuple(reversed(shape)) if len(shape) == 3 else (10, 10, 10)
        return UnifiedSitkImage(size=size)

    class DICOMOrientImageFilter:
        """Mock DICOM orientation filter."""

        @staticmethod
        def GetDirectionCosinesFromOrientation(
            orientation: str,
        ) -> tuple[float, ...]:
            """Return mock direction cosines for any orientation string."""
            # Return identity-like direction for testing
            return tuple(float(i) for i in range(9))


@pytest.fixture
def mock_sitk_module(monkeypatch):
    """Mock the entire SimpleITK module consistently across tests."""
    mock_sitk = MockSitkModule()
    monkeypatch.setattr("aind_zarr_utils.zarr.sitk", mock_sitk)
    return mock_sitk


@pytest.fixture
def mock_sitk_image():
    """Provide a basic mock SimpleITK image for direct use."""
    return UnifiedSitkImage()


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
# ANTs Infrastructure
# ============================================================================


class MockAntsImage:
    """Mock ANTs image for testing coordinate transformations."""

    def __init__(self, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        self.spacing = spacing
        self.origin = origin
        self.direction = np.eye(3)
        self.shape = (10, 10, 10)

    def set_spacing(self, spacing):
        self.spacing = spacing

    def set_origin(self, origin):
        self.origin = origin

    def set_direction(self, direction):
        self.direction = direction


class MockAntsModule:
    """Mock ANTs module for coordinate transformation testing."""

    @staticmethod
    def from_numpy(
        array: np.ndarray, spacing=None, direction=None, origin=None
    ):
        """Mock ants.from_numpy."""
        img = MockAntsImage()
        if spacing is not None:
            img.spacing = spacing
        if direction is not None:
            img.direction = direction
        if origin is not None:
            img.origin = origin
        return img

    @staticmethod
    def image_read(filename: str):
        """Mock ants.image_read."""
        return MockAntsImage()

    @staticmethod
    def image_write(image, filename: str):
        """Mock ants.image_write."""
        # Create empty file to simulate write
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).touch()


@pytest.fixture
def mock_ants_module(monkeypatch):
    """Mock ANTs module for transformation testing."""
    mock_ants = MockAntsModule()
    monkeypatch.setattr("aind_zarr_utils.zarr.ants", mock_ants)
    return mock_ants


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
# Annotation Testing Infrastructure
# ============================================================================


@pytest.fixture
def mock_annotation_functions(monkeypatch):
    """Mock annotation processing functions for neuroglancer tests."""

    def mock_annotation_indices_to_anatomical(stub_img, annotations):
        """Mock coordinate transformation - just add 1 to all values."""
        return {k: v + 1 for k, v in annotations.items()}

    def mock_zarr_to_sitk_stub(zarr_uri, metadata, **kwargs):
        """Mock zarr to sitk stub conversion."""
        return UnifiedSitkImage(), (10, 10, 10)

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
