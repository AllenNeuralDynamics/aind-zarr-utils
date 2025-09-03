"""Tests for s3_cache module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aind_zarr_utils import s3_cache


def test_cache_manager_persistent():
    """Test CacheManager with persistent cache directory."""
    cache_dir = Path(tempfile.gettempdir()) / "test_cache"

    with s3_cache.CacheManager(cache_dir=cache_dir, persistent=True) as cm:
        assert cm.dir == cache_dir.resolve()
        assert cm.dir.exists()

    # Directory should still exist after context manager
    assert cm.dir.exists()

    # Clean up
    if cache_dir.exists():
        cache_dir.rmdir()


def test_cache_manager_ephemeral():
    """Test CacheManager with ephemeral cache directory."""
    with s3_cache.CacheManager(persistent=False) as cm:
        cache_path = cm.dir
        assert cache_path.exists()
        assert cache_path.is_dir()

    # Directory should be cleaned up after context manager
    assert not cache_path.exists()


def test_cache_manager_default_persistent():
    """Test CacheManager with default persistent behavior."""
    with s3_cache.CacheManager() as cm:
        expected_path = Path.home() / ".cache" / "s3-ants"
        assert cm.dir == expected_path
        assert cm.dir.exists()


def test_is_s3_uri():
    """Test S3 URI detection."""
    assert s3_cache._is_s3_uri("s3://bucket/key")
    assert s3_cache._is_s3_uri("S3://BUCKET/KEY")
    assert not s3_cache._is_s3_uri("http://example.com")
    assert not s3_cache._is_s3_uri("/local/path")
    assert not s3_cache._is_s3_uri("file:///local/path")
    assert not s3_cache._is_s3_uri("")


def test_safe_slug():
    """Test filesystem-safe slug generation."""
    assert s3_cache._safe_slug("my-bucket") == "my-bucket"
    assert s3_cache._safe_slug("my_bucket") == "my_bucket"
    assert (
        s3_cache._safe_slug("my/path/with/slashes") == "my_path_with_slashes"
    )
    assert s3_cache._safe_slug("special!@#$%chars") == "special_chars"
    assert (
        s3_cache._safe_slug("___multiple___underscores___")
        == "multiple_underscores"
    )
    assert s3_cache._safe_slug("") == "root"

    # Test maxlen
    long_string = "a" * 200
    result = s3_cache._safe_slug(long_string, maxlen=50)
    assert len(result) == 50
    assert result == "a" * 50


def test_cache_name():
    """Test cache filename generation."""
    bucket = "my-bucket"
    key = "path/to/file.nii.gz"
    etag = "abc123def456"

    cache_name = s3_cache._cache_name(bucket, key, etag)

    # Should preserve original filename
    assert cache_name.endswith("file.nii.gz")

    # Should include bucket and path info
    assert "my-bucket" in cache_name
    assert "path_to" in cache_name

    # Should include hash
    assert len(cache_name.split("__")) >= 3  # bucket, path, hash, filename


def test_cache_name_root_key():
    """Test cache filename generation with root-level key."""
    bucket = "bucket"
    key = "file.txt"
    etag = "abc123"

    cache_name = s3_cache._cache_name(bucket, key, etag)

    # Should not include path component for root-level files
    assert "file.txt" in cache_name
    assert "bucket" in cache_name
    assert cache_name.count("__") == 2  # bucket, hash, filename


def test_get_local_path_for_resource_local_file(tmp_path):
    """Test getting local path for an existing local file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    result = s3_cache.get_local_path_for_resource(str(test_file))

    assert result.path == test_file.resolve()
    assert not result.from_cache
    assert result.source == "local"


def test_get_local_path_for_resource_missing_local_file(tmp_path):
    """Test error handling for missing local file."""
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Local path not found"):
        s3_cache.get_local_path_for_resource(str(missing_file))


def test_get_local_path_for_resource_s3_success(mock_s3_client, tmp_path):
    """Test successful S3 resource download."""
    test_data = {"test": "data"}
    mock_s3_client.data = test_data
    mock_s3_client.etag = "test-etag-123"
    mock_s3_client.content_length = 100

    uri = "s3://test-bucket/path/to/file.json"

    result = s3_cache.get_local_path_for_resource(
        uri, cache_dir=tmp_path, s3_client=mock_s3_client
    )

    assert result.from_cache
    assert result.source == "s3"
    assert result.path.exists()
    assert result.path.parent == tmp_path

    # Verify download was called
    assert len(mock_s3_client.downloads) == 1
    download_call = mock_s3_client.downloads[0]
    assert download_call[0] == "test-bucket"  # Bucket
    assert download_call[1] == "path/to/file.json"  # Key


def test_get_local_path_for_resource_s3_cache_hit(mock_s3_client, tmp_path):
    """Test S3 resource cache hit (file already exists)."""
    test_data = {"test": "data"}
    test_content = '{"test": "data"}'
    mock_s3_client.data = test_data
    mock_s3_client.etag = "test-etag-123"
    mock_s3_client.content_length = len(
        test_content.encode()
    )  # Exact byte count

    uri = "s3://test-bucket/file.json"

    # Create cache file first with exact size
    cache_name = s3_cache._cache_name(
        "test-bucket", "file.json", "test-etag-123"
    )
    cache_file = tmp_path / cache_name
    cache_file.write_text(test_content)

    # Verify the file size matches content_length
    assert cache_file.stat().st_size == mock_s3_client.content_length

    result = s3_cache.get_local_path_for_resource(
        uri, cache_dir=tmp_path, s3_client=mock_s3_client
    )

    assert result.from_cache
    assert result.source == "s3"
    assert result.path == cache_file

    # Should not have downloaded (cache hit)
    assert len(mock_s3_client.downloads) == 0


def test_get_local_path_for_resource_s3_head_blocked(mock_s3_client, tmp_path):
    """Test S3 resource access when HEAD is blocked but GET works."""
    test_data = {"test": "data"}
    mock_s3_client.data = test_data
    mock_s3_client.etag = "test-etag-456"
    mock_s3_client.content_length = 100
    mock_s3_client.simulate_head_blocked = True

    uri = "s3://test-bucket/file.json"

    result = s3_cache.get_local_path_for_resource(
        uri, cache_dir=tmp_path, s3_client=mock_s3_client
    )

    assert result.from_cache
    assert result.source == "s3"
    assert result.path.exists()

    # Should have fallen back to range GET and then downloaded
    assert len(mock_s3_client.downloads) == 1


def test_get_local_path_for_resource_s3_not_found(mock_s3_client, tmp_path):
    """Test S3 resource not found error."""
    mock_s3_client.simulate_errors = {"test-bucket/missing.json": 404}

    uri = "s3://test-bucket/missing.json"

    with pytest.raises(
        FileNotFoundError, match="S3 object not found or inaccessible"
    ):
        s3_cache.get_local_path_for_resource(
            uri, cache_dir=tmp_path, s3_client=mock_s3_client
        )


def test_get_local_path_for_resource_s3_access_denied(
    mock_s3_client, tmp_path
):
    """Test S3 resource access denied error."""
    mock_s3_client.simulate_errors = {"test-bucket/denied.json": 403}

    uri = "s3://test-bucket/denied.json"

    with pytest.raises(
        FileNotFoundError, match="S3 object not found or inaccessible"
    ):
        s3_cache.get_local_path_for_resource(
            uri, cache_dir=tmp_path, s3_client=mock_s3_client
        )


def test_get_local_path_for_resource_anonymous_access(tmp_path):
    """Test anonymous S3 access configuration."""
    with patch(
        "aind_zarr_utils.s3_cache._ensure_client"
    ) as mock_ensure_client:
        # Create a proper mock client instance
        from tests.conftest import UnifiedS3Client

        mock_client = UnifiedS3Client()
        mock_client.etag = "test-etag"
        mock_client.content_length = 50
        mock_ensure_client.return_value = mock_client

        uri = "s3://public-bucket/file.json"

        s3_cache.get_local_path_for_resource(
            uri, cache_dir=tmp_path, anonymous=True
        )

        # Verify _ensure_client was called with anonymous=True
        mock_ensure_client.assert_called_once()
        call_args = mock_ensure_client.call_args
        assert call_args[1]["anonymous"] is True


def test_get_local_path_for_resource_custom_config(tmp_path):
    """Test custom configuration parameters."""
    with patch(
        "aind_zarr_utils.s3_cache._ensure_client"
    ) as mock_ensure_client:
        # Create a proper mock client instance
        from tests.conftest import UnifiedS3Client

        mock_client = UnifiedS3Client()
        mock_client.etag = "test-etag"
        mock_client.content_length = 50
        mock_ensure_client.return_value = mock_client

        uri = "s3://bucket/large-file.bin"

        s3_cache.get_local_path_for_resource(
            uri,
            cache_dir=tmp_path,
            max_concurrency=32,
            multipart_threshold_bytes=16 * 1024 * 1024,  # 16MB
        )

        # Verify custom parameters were passed
        mock_ensure_client.assert_called_once()
        call_args = mock_ensure_client.call_args
        assert call_args[1]["max_concurrency"] == 32


def test_safe_atomic_replace(tmp_path):
    """Test atomic file replacement."""
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "subdir" / "dest.txt"

    src_file.write_text("test content")

    # Destination directory doesn't exist yet
    assert not dst_file.parent.exists()

    s3_cache._safe_atomic_replace(src_file, dst_file)

    # Should create parent directory and move file
    assert dst_file.exists()
    assert dst_file.read_text() == "test content"
    assert not src_file.exists()


def test_head_or_peek_byte_head_success(mock_s3_client):
    """Test successful HEAD operation."""
    mock_s3_client.etag = "test-etag-789"
    mock_s3_client.content_length = 12345

    etag, size = s3_cache._head_or_peek_byte(mock_s3_client, "bucket", "key")

    assert etag == "test-etag-789"
    assert size == 12345


def test_head_or_peek_byte_head_blocked_fallback(mock_s3_client):
    """Test fallback to range GET when HEAD is blocked."""
    mock_s3_client.etag = "fallback-etag"
    mock_s3_client.content_length = 9999
    mock_s3_client.simulate_head_blocked = True

    etag, size = s3_cache._head_or_peek_byte(mock_s3_client, "bucket", "key")

    assert etag == "fallback-etag"
    assert size == 9999


def test_default_cache_dir():
    """Test default cache directory creation."""
    cache_dir = s3_cache._default_cache_dir()
    assert cache_dir.name == "s3-ants-cache"
    assert cache_dir.exists()
    assert cache_dir.is_dir()
