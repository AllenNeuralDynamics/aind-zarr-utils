"""
Utilities for working with local file paths and S3 URIs.

This module provides helpers to:
- detect whether a string is a URL or a local file path,
- parse S3 URIs into ``(bucket, key)`` parts,
- convert between a normalized tuple form and string form,
- join paths in a scheme-aware way (local filesystem vs. S3).

Functions
---------
is_url(path_or_url)
    Return True if the string parses as a URL (non-empty scheme) that is not
    a local file path (see Notes).
is_file_path(path_or_url)
    Return True if the string represents a local file path or a ``file://``
    URL.
parse_s3_uri(s3_uri)
    Split an S3 URI into ``(bucket, key)``.
as_pathlike(base)
    Parse a string into ``(kind, bucket, path)`` where ``kind`` is ``"s3"``
    or ``"file"``, ``bucket`` is None for local paths, and ``path`` is a
    ``PurePosixPath`` (S3) or ``Path`` (local).
as_string(kind, bucket, path)
    Convert the tuple back to ``"s3://bucket/key"`` or a local path string.
join_any(base, *parts)
    Join path components under either scheme and return a string.

Notes
-----
- S3 keys are treated as POSIX paths (forward slashes) regardless of host OS.
- For S3, ``as_string`` returns ``"s3://bucket/"`` when ``path`` is the root.
- ``join_any`` normalizes backslashes in ``parts`` to ``/`` for S3 keys.
- ``is_url`` delegates to ``urllib.parse.urlparse``; a bare path with no
  scheme is considered a file path.

Examples
--------
>>> parse_s3_uri('s3://my-bucket/dir/file.txt')
('my-bucket', 'dir/file.txt')

>>> kind, bucket, p = as_pathlike('s3://my-bucket/dir')
>>> as_string(kind, bucket, p)
's3://my-bucket/dir'

>>> join_any('s3://my-bucket', 'a', 'b.txt')
's3://my-bucket/a/b.txt'

>>> join_any('/home/user', 'a', 'b.txt')
'/home/user/a/b.txt'
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import ParseResult, urlparse


def _is_url_parsed(parsed: ParseResult) -> bool:
    """
    Check if a parsed URL is an HTTP, HTTPS, or S3 URL.

    Parameters
    ----------
    parsed : ParseResult
        The parsed URL object.

    Returns
    -------
    bool
        True if the URL is HTTP, HTTPS, or S3, False otherwise.
    """
    return parsed.scheme in ("http", "https", "s3")


def _is_file_parsed(parsed: ParseResult) -> bool:
    """
    Check if a parsed URL represents a file path.

    Parameters
    ----------
    parsed : ParseResult
        The parsed URL object.

    Returns
    -------
    bool
        True if the URL represents a file path, False otherwise.
    """
    is_file = not _is_url_parsed(parsed) and (
        parsed.scheme == "file"
        or (not parsed.scheme and parsed.path is not None)
    )
    return is_file


def is_url(path_or_url: str) -> bool:
    """
    Determine if a given string is a URL.

    Parameters
    ----------
    path_or_url : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a URL, False otherwise.
    """
    parsed = urlparse(path_or_url)
    return _is_url_parsed(parsed)


def is_file_path(path_or_url: str) -> bool:
    """
    Determine if a given string is a file path.

    Parameters
    ----------
    path_or_url : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a file path, False otherwise.
    """
    parsed = urlparse(path_or_url)
    return _is_file_parsed(parsed)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI into bucket and key components.

    Parameters
    ----------
    s3_uri : str
        The S3 URI to parse.

    Returns
    -------
    tuple
        A tuple containing the bucket name and the key.

    Raises
    ------
    ValueError
        If the URI is not a valid S3 URI.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError("Not a valid S3 URI")
    return parsed.netloc, parsed.path.lstrip("/")


def as_pathlike(base: str):
    u = urlparse(base)
    if u.scheme == "s3":
        # bucket + POSIX key
        return ("s3", u.netloc, PurePosixPath(u.path.lstrip("/")))
    else:
        # local path
        return ("file", None, Path(base))


def as_string(
    kind: str, bucket: str | None, path: Path | PurePosixPath
) -> str:
    """
    Convert a parsed (kind, bucket, path) tuple back to a string.

    - For S3 URIs: "s3://bucket/key"
    - For local paths: "/local/path"
    """
    if kind == "s3":
        key = path.as_posix().lstrip("/")
        # Handle root path case - empty key or "." should result in just bucket
        if not key or key == ".":
            return f"s3://{bucket}/" if bucket else "s3://"
        return f"s3://{bucket}/{key}" if bucket else f"s3://{key}"
    elif kind == "file":
        return str(path)
    else:
        raise ValueError(f"Unsupported kind: {kind}")


def join_any(base: str, *parts: str) -> str:
    """Join path components, returning a string URI for s3, or a filesystem
    path for local."""
    kind, bucket, p = as_pathlike(base)

    if kind == "s3":
        # For S3, normalize backslashes to forward slashes in parts
        normalized_parts = [part.replace("\\", "/") for part in parts]
        joined = p.joinpath(*normalized_parts)
        return f"s3://{bucket}/{joined.as_posix()}"
    else:
        joined = p.joinpath(*parts)
        return str(joined)
