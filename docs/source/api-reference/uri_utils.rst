uri_utils module
================

.. automodule:: aind_zarr_utils.uri_utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``uri_utils`` module provides utilities for working with local file paths and S3 URIs. It enables scheme-aware path manipulation and provides helpers for detecting URL vs file paths, parsing S3 URIs, and joining paths.

Functions
---------

.. autosummary::
   :toctree: generated/
   
   is_url
   is_file_path
   parse_s3_uri
   as_pathlike
   as_string
   join_any

Examples
--------

Basic URL and path detection::

    from aind_zarr_utils.uri_utils import is_url, is_file_path, parse_s3_uri

    # URL detection
    assert is_url("https://example.com") == True
    assert is_file_path("/local/path") == True
    
    # S3 URI parsing
    bucket, key = parse_s3_uri("s3://my-bucket/path/to/file.txt")
    # Returns: ("my-bucket", "path/to/file.txt")

Path manipulation across schemes::

    from aind_zarr_utils.uri_utils import join_any, as_pathlike, as_string

    # Join paths (works for both local and S3)
    result = join_any("s3://bucket/base", "subdir", "file.txt")
    # Returns: "s3://bucket/base/subdir/file.txt"
    
    # Convert to normalized form and back
    kind, bucket, path = as_pathlike("s3://bucket/path/file.txt")
    uri = as_string(kind, bucket, path)
    # Round-trip preserves original URI