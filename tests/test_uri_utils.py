"""Tests for uri_utils module."""

from pathlib import Path, PurePosixPath

import pytest

from aind_zarr_utils import uri_utils


class TestUrlDetection:
    """Test URL and file path detection functions."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://example.com", True),
            ("https://example.com", True),
            ("s3://bucket/key", True),
            ("ftp://example.com", False),
            ("file:///path/to/file", False),
            ("/local/path", False),
            ("./relative/path", False),
            ("", False),
            ("not-a-url", False),
        ],
    )
    def test_is_url(self, url: str, expected: bool):
        """Test URL detection for various schemes."""
        assert uri_utils.is_url(url) == expected

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("file:///path/to/file", True),
            ("/absolute/path", True),
            ("./relative/path", True),
            ("relative/path", True),
            ("", True),  # Empty path treated as file path
            ("http://example.com", False),
            ("https://example.com", False),
            ("s3://bucket/key", False),
        ],
    )
    def test_is_file_path(self, path: str, expected: bool):
        """Test file path detection for various inputs."""
        assert uri_utils.is_file_path(path) == expected

    def test_url_and_file_path_mutually_exclusive(self):
        """Test that URL and file path detection are mutually exclusive."""
        test_cases = [
            "http://example.com",
            "https://example.com",
            "s3://bucket/key",
            "/local/path",
            "./relative/path",
            "file:///path/to/file",
        ]

        for test_case in test_cases:
            is_url = uri_utils.is_url(test_case)
            is_file = uri_utils.is_file_path(test_case)
            # Exactly one should be True
            assert is_url != is_file, f"Failed for: {test_case}"


class TestS3UriParsing:
    """Test S3 URI parsing functionality."""

    @pytest.mark.parametrize(
        "s3_uri,expected_bucket,expected_key",
        [
            ("s3://bucket/key", "bucket", "key"),
            ("s3://bucket/path/to/file.txt", "bucket", "path/to/file.txt"),
            ("s3://bucket/", "bucket", ""),
            ("s3://bucket", "bucket", ""),
            (
                "s3://my-bucket/deep/nested/path/file.json",
                "my-bucket",
                "deep/nested/path/file.json",
            ),
        ],
    )
    def test_parse_s3_uri_success(
        self, s3_uri: str, expected_bucket: str, expected_key: str
    ):
        """Test successful S3 URI parsing."""
        bucket, key = uri_utils.parse_s3_uri(s3_uri)
        assert bucket == expected_bucket
        assert key == expected_key

    @pytest.mark.parametrize(
        "invalid_uri",
        [
            "http://example.com",
            "https://bucket.s3.amazonaws.com/key",
            "file:///path/to/file",
            "/local/path",
            "not-a-uri",
            "",
        ],
    )
    def test_parse_s3_uri_failure(self, invalid_uri: str):
        """Test S3 URI parsing with invalid inputs."""
        with pytest.raises(ValueError, match="Not a valid S3 URI"):
            uri_utils.parse_s3_uri(invalid_uri)


class TestPathConversion:
    """Test path conversion utilities."""

    def test_as_pathlike_s3_uri(self):
        """Test converting S3 URI to pathlike representation."""
        result = uri_utils.as_pathlike("s3://bucket/path/to/file.txt")
        kind, bucket, path = result

        assert kind == "s3"
        assert bucket == "bucket"
        assert isinstance(path, PurePosixPath)
        assert str(path) == "path/to/file.txt"

    def test_as_pathlike_s3_uri_root(self):
        """Test converting S3 URI with no path to pathlike representation."""
        result = uri_utils.as_pathlike("s3://bucket/")
        kind, bucket, path = result

        assert kind == "s3"
        assert bucket == "bucket"
        assert isinstance(path, PurePosixPath)
        assert str(path) == "."

    def test_as_pathlike_local_path(self):
        """Test converting local path to pathlike representation."""
        result = uri_utils.as_pathlike("/local/path/to/file.txt")
        kind, bucket, path = result

        assert kind == "file"
        assert bucket is None
        assert isinstance(path, Path)
        assert str(path) == "/local/path/to/file.txt"

    def test_as_pathlike_relative_path(self):
        """Test converting relative path to pathlike representation."""
        result = uri_utils.as_pathlike("relative/path/file.txt")
        kind, bucket, path = result

        assert kind == "file"
        assert bucket is None
        assert isinstance(path, Path)
        assert str(path) == "relative/path/file.txt"

    def test_as_string_s3_with_bucket(self):
        """Test converting S3 pathlike back to string with bucket."""
        path = PurePosixPath("path/to/file.txt")
        result = uri_utils.as_string("s3", "bucket", path)
        assert result == "s3://bucket/path/to/file.txt"

    def test_as_string_s3_no_bucket(self):
        """Test converting S3 pathlike back to string without bucket."""
        path = PurePosixPath("path/to/file.txt")
        result = uri_utils.as_string("s3", None, path)
        assert result == "s3://path/to/file.txt"

    def test_as_string_s3_empty_key(self):
        """Test converting S3 pathlike with empty key back to string."""
        path = PurePosixPath("")
        result = uri_utils.as_string("s3", "bucket", path)
        assert result == "s3://bucket/"

    def test_as_string_s3_root_path(self):
        """Test converting S3 pathlike with root path back to string."""
        path = PurePosixPath(".")
        result = uri_utils.as_string("s3", "bucket", path)
        assert result == "s3://bucket/"

    def test_as_string_file(self):
        """Test converting file pathlike back to string."""
        path = Path("/local/path/to/file.txt")
        result = uri_utils.as_string("file", None, path)
        assert result == "/local/path/to/file.txt"

    def test_as_string_unsupported_kind(self):
        """Test as_string with unsupported kind raises ValueError."""
        path = Path("/some/path")
        with pytest.raises(ValueError, match="Unsupported kind: ftp"):
            uri_utils.as_string("ftp", None, path)

    def test_round_trip_conversion_s3(self):
        """Test round-trip conversion for S3 URIs."""
        original = "s3://bucket/path/to/file.txt"
        kind, bucket, path = uri_utils.as_pathlike(original)
        result = uri_utils.as_string(kind, bucket, path)
        assert result == original

    def test_round_trip_conversion_file(self):
        """Test round-trip conversion for file paths."""
        original = "/local/path/to/file.txt"
        kind, bucket, path = uri_utils.as_pathlike(original)
        result = uri_utils.as_string(kind, bucket, path)
        assert result == original


class TestPathJoining:
    """Test path joining functionality."""

    def test_join_any_s3_uri(self):
        """Test joining path components with S3 URI base."""
        base = "s3://bucket/base/path"
        result = uri_utils.join_any(base, "subdir", "file.txt")
        assert result == "s3://bucket/base/path/subdir/file.txt"

    def test_join_any_s3_uri_single_part(self):
        """Test joining single path component with S3 URI base."""
        base = "s3://bucket/base"
        result = uri_utils.join_any(base, "file.txt")
        assert result == "s3://bucket/base/file.txt"

    def test_join_any_s3_uri_no_parts(self):
        """Test joining no path components with S3 URI base."""
        base = "s3://bucket/base/path"
        result = uri_utils.join_any(base)
        assert result == "s3://bucket/base/path"

    def test_join_any_local_path(self):
        """Test joining path components with local path base."""
        base = "/local/base/path"
        result = uri_utils.join_any(base, "subdir", "file.txt")
        expected = str(Path(base) / "subdir" / "file.txt")
        assert result == expected

    def test_join_any_local_path_single_part(self):
        """Test joining single path component with local path base."""
        base = "/local/base"
        result = uri_utils.join_any(base, "file.txt")
        expected = str(Path(base) / "file.txt")
        assert result == expected

    def test_join_any_relative_path(self):
        """Test joining path components with relative path base."""
        base = "relative/base"
        result = uri_utils.join_any(base, "subdir", "file.txt")
        expected = str(Path(base) / "subdir" / "file.txt")
        assert result == expected

    def test_join_any_empty_parts(self):
        """Test joining with empty string parts."""
        base = "s3://bucket/base"
        result = uri_utils.join_any(base, "", "file.txt", "")
        assert result == "s3://bucket/base/file.txt"

    def test_join_any_s3_normalizes_backslashes(self):
        """Test that S3 joins normalize backslashes to forward slashes."""
        base = "s3://bucket/base"
        result = uri_utils.join_any(base, "dir\\with\\backslashes")
        # Backslashes should be normalized to forward slashes for S3
        assert result == "s3://bucket/base/dir/with/backslashes"
        assert "\\" not in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        # Empty string is treated as file path
        assert uri_utils.is_file_path("")
        assert not uri_utils.is_url("")

        # Empty string as pathlike - Python Path("") becomes Path(".")
        kind, bucket, path = uri_utils.as_pathlike("")
        assert kind == "file"
        assert bucket is None
        assert str(path) == "."  # Python's Path("") behavior

    def test_special_characters_in_paths(self):
        """Test handling of special characters in paths."""
        # S3 URI with special characters
        s3_uri = "s3://bucket/path%20with%20spaces/file.txt"
        bucket, key = uri_utils.parse_s3_uri(s3_uri)
        assert bucket == "bucket"
        assert key == "path%20with%20spaces/file.txt"

        # Round-trip should preserve encoding
        kind, bucket, path = uri_utils.as_pathlike(s3_uri)
        result = uri_utils.as_string(kind, bucket, path)
        assert result == s3_uri

    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_path = "/path/with/unicode/файл.txt"
        kind, bucket, path = uri_utils.as_pathlike(unicode_path)
        result = uri_utils.as_string(kind, bucket, path)
        assert result == unicode_path

    def test_very_long_paths(self):
        """Test handling of very long paths."""
        long_component = "very" * 100  # 400 characters
        s3_uri = f"s3://bucket/{long_component}/file.txt"

        bucket, key = uri_utils.parse_s3_uri(s3_uri)
        assert bucket == "bucket"
        assert key == f"{long_component}/file.txt"

        # Round-trip test
        kind, bucket, path = uri_utils.as_pathlike(s3_uri)
        result = uri_utils.as_string(kind, bucket, path)
        assert result == s3_uri
