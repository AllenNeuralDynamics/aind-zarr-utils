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
            # Windows paths should not be URLs
            ("C:\\Windows\\System32", False),
            ("C:/Windows/System32", False),
            ("D:\\path\\to\\file.txt", False),
            ("\\\\server\\share\\path", False),
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
            # Windows paths should be file paths
            ("C:\\Windows\\System32", True),
            ("C:/Windows/System32", True),
            ("D:\\path\\to\\file.txt", True),
            ("\\\\server\\share\\path", True),
            ("E:", True),  # Drive letter only
            ("C:\\", True),  # Drive root
            ("C:/", True),   # Drive root with forward slash
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
            # Windows paths
            "C:\\Windows\\System32",
            "C:/Windows/System32",
            "D:\\path\\to\\file.txt",
            "\\\\server\\share\\path",
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
        with pytest.raises(ValueError, match="Unsupported kind: 'ftp'"):
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


class TestWindowsPathHandling:
    """Test Windows-specific path handling edge cases."""

    @pytest.mark.parametrize(
        "path,expected_is_file,expected_is_url",
        [
            # Drive letters with various separators
            ("C:\\", True, False),
            ("C:/", True, False),
            ("C:", True, False),
            ("D:\\Windows\\System32", True, False),
            ("E:/Users/test/file.txt", True, False),
            
            # UNC paths
            ("\\\\server\\share", True, False),
            ("\\\\server\\share\\", True, False),
            ("\\\\server\\share\\path\\file.txt", True, False),
            ("//server/share", True, False),
            ("//server/share/path", True, False),
            
            # Mixed separators
            ("C:\\Windows/System32", True, False),
            ("D:/Users\\test\\file.txt", True, False),
            ("\\\\server/share\\path", True, False),
            
            # Edge cases that might confuse urlparse
            ("C:relative\\path", True, False),
            ("C:file.txt", True, False),
            ("Z:\\", True, False),
            
            # Non-Windows paths that should not be confused
            ("s3://bucket/C:/fake/path", False, True),
            ("http://example.com/C:\\fake", False, True),
            ("ftp://C:/not/windows", False, False),
            
            # Relative paths that might look like drive letters
            ("c/not/windows", True, False),
            ("c:/not/windows/either", True, False),
        ],
    )
    def test_windows_path_detection(self, path: str, expected_is_file: bool, expected_is_url: bool):
        """Test Windows path detection for various edge cases."""
        is_file = uri_utils.is_file_path(path)
        is_url = uri_utils.is_url(path)
        
        assert is_file == expected_is_file, f"is_file_path('{path}') should be {expected_is_file}, got {is_file}"
        assert is_url == expected_is_url, f"is_url('{path}') should be {expected_is_url}, got {is_url}"
        
        # They should be mutually exclusive (except for edge cases)
        if expected_is_file or expected_is_url:
            assert is_file != is_url, f"is_file_path and is_url should be mutually exclusive for '{path}'"

    @pytest.mark.parametrize(
        "path,expected_kind,expected_bucket,expected_path_str",
        [
            # Standard Windows paths
            ("C:\\Windows\\System32", "file", None, "C:\\Windows\\System32"),
            ("D:/Users/test/file.txt", "file", None, "D:\\Users\\test\\file.txt"),
            ("E:\\", "file", None, "E:\\"),
            
            # UNC paths
            ("\\\\server\\share\\path", "file", None, "\\\\server\\share\\path"),
            ("//server/share/file.txt", "file", None, "\\\\server\\share\\file.txt"),
            
            # Mixed separators - should normalize appropriately
            ("C:\\Windows/System32", "file", None, "C:\\Windows\\System32"),
            ("D:/Users\\test", "file", None, "D:\\Users\\test"),
        ],
    )
    def test_windows_as_pathlike(self, path: str, expected_kind: str, expected_bucket: str | None, expected_path_str: str):
        """Test as_pathlike conversion for Windows paths."""
        kind, bucket, path_obj = uri_utils.as_pathlike(path)
        
        assert kind == expected_kind
        assert bucket == expected_bucket
        assert str(path_obj) == expected_path_str

    @pytest.mark.parametrize(
        "base,parts,expected_pattern",
        [
            # Windows drive paths
            ("C:\\base", ("sub", "file.txt"), "C:\\base\\sub\\file.txt"),
            ("D:/base", ("sub", "file.txt"), "D:/base/sub/file.txt"),
            ("E:\\", ("dir", "file.txt"), "E:\\dir\\file.txt"),
            
            # UNC paths
            ("\\\\server\\share", ("dir", "file.txt"), "\\\\server\\share\\dir\\file.txt"),
            ("//server/share", ("dir", "file.txt"), "//server/share/dir/file.txt"),
            
            # Mixed separators in parts
            ("C:\\base", ("sub\\dir", "file.txt"), "C:\\base\\sub\\dir\\file.txt"),
            ("D:/base", ("sub/dir", "file.txt"), "D:/base/sub/dir/file.txt"),
        ],
    )
    def test_windows_join_any(self, base: str, parts: tuple, expected_pattern: str):
        """Test join_any with Windows paths."""
        result = uri_utils.join_any(base, *parts)
        
        # For Windows paths, normalize separators for comparison
        # The main requirement is consistent separators, not specific separator types
        from pathlib import Path, PureWindowsPath
        
        # Normalize both to use the same separator for comparison
        def normalize_for_comparison(path_str: str) -> str:
            # Convert all separators to forward slashes for comparison
            return path_str.replace("\\", "/")
        
        result_normalized = normalize_for_comparison(result)
        expected_normalized = normalize_for_comparison(expected_pattern)
        
        # Check that the normalized paths are equivalent
        assert result_normalized == expected_normalized, f"join_any('{base}', {parts}) = '{result}', expected pattern like '{expected_pattern}'"
        
        # Additionally, ensure the result has consistent separators
        # (no mixing of / and \ within the same path)
        has_forward = "/" in result
        has_backward = "\\" in result
        if has_forward and has_backward:
            # Mixed separators are only acceptable for UNC paths like //server\share
            if not (result.startswith("//") or result.startswith("\\\\")):
                assert False, f"Result has mixed separators: '{result}'"

    def test_windows_round_trip_conversions(self):
        """Test round-trip conversions preserve Windows path semantics."""
        test_paths = [
            "C:\\Windows\\System32",
            "D:/Users/test/file.txt",
            "\\\\server\\share\\path",
            "//server/share/file.txt",
            "E:\\",
        ]
        
        for original in test_paths:
            kind, bucket, path = uri_utils.as_pathlike(original)
            result = uri_utils.as_string(kind, bucket, path)
            
            # For Windows paths, both should normalize to the same logical path
            # Use PureWindowsPath for consistent comparison
            from pathlib import PureWindowsPath
            
            # Both should represent the same logical Windows path
            try:
                original_normalized = PureWindowsPath(original)
                result_normalized = PureWindowsPath(result)
                assert original_normalized == result_normalized, f"Round-trip failed: '{original}' -> '{result}'"
            except (ValueError, OSError):
                # Fallback to string comparison if PureWindowsPath can't handle it
                # Normalize separators for comparison
                original_norm = original.replace("/", "\\")
                result_norm = result.replace("/", "\\")
                assert original_norm == result_norm, f"Round-trip failed: '{original}' -> '{result}'"

    def test_windows_urlparse_edge_cases(self):
        """Test edge cases where urlparse might misinterpret Windows paths."""
        # These are cases where urlparse might think the drive letter is a scheme
        edge_cases = [
            "C:relative\\path",  # Drive with relative path
            "C:file.txt",       # Drive with filename only
            "D:../parent",      # Drive with relative navigation
        ]
        
        for path in edge_cases:
            # These should all be detected as file paths, not URLs
            assert uri_utils.is_file_path(path), f"'{path}' should be detected as file path"
            assert not uri_utils.is_url(path), f"'{path}' should not be detected as URL"
            
            # Should be convertible via as_pathlike
            kind, bucket, path_obj = uri_utils.as_pathlike(path)
            assert kind == "file"
            assert bucket is None


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
