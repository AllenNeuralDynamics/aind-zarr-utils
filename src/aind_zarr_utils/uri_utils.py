"""
Utilities for local paths, Windows paths, file:// URIs, and S3 URIs.

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
- Only ``http``, ``https``, and ``s3`` are treated as network URLs.
  ``file://`` is considered a file path.
- Windows drive-letter paths like ``C:\\...`` / ``C:/...`` and UNC paths
  like ``\\\\server\\share\\...`` are considered file paths on all OSes.
- S3 keys are treated as POSIX paths (forward slashes) regardless of host.
- For S3, ``as_string`` returns ``"s3://bucket/"`` when ``path`` is root.

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

from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import ParseResult, urlparse
from urllib.request import url2pathname

# Network URL schemes considered "URLs" (everything else -> file path).
_KNOWN_URL_SCHEMES = {"http", "https", "s3"}


def _looks_like_windows_path(s: str) -> bool:
    """
    Heuristic for Windows drive/UNC paths (works cross-platform).

    Parameters
    ----------
    s : str
        Input path-like string.

    Returns
    -------
    bool
        True if `s` looks like a Windows drive path (e.g., ``C:\\...`` or
        ``C:/...``) or a UNC path (e.g., ``\\\\server\\share\\...``).
    """
    # UNC paths: \\server\share or //server/share
    if s.startswith("\\\\") or (s.startswith("//") and "/" in s[2:]):
        return True

    # Windows drive patterns: C:, C:\, C:/, etc.
    try:
        pw = PureWindowsPath(s)
        if pw.drive:
            return True
    except (ValueError, OSError):
        pass

    # Additional heuristics for drive-relative paths like "C:file.txt"
    if len(s) >= 2 and s[1] == ":":
        # Check if first character is a letter (drive letter)
        return s[0].isalpha()

    return False


def _is_url_parsed(parsed: ParseResult) -> bool:
    """
    Check if a parsed URL is a known network URL with a netloc.

    Parameters
    ----------
    parsed : ParseResult
        Result of ``urllib.parse.urlparse``.

    Returns
    -------
    bool
        True if scheme is in {``http``, ``https``, ``s3``} and netloc is
        non-empty.
    """
    scheme = (parsed.scheme or "").lower()
    if scheme not in _KNOWN_URL_SCHEMES:
        return False
    return bool(parsed.netloc)


def _is_file_parsed(parsed: ParseResult) -> bool:
    """
    Check if a parsed string represents a file path.

    Parameters
    ----------
    parsed : ParseResult
        Result of ``urllib.parse.urlparse``.

    Returns
    -------
    bool
        True if the input is a local/Windows path or a ``file://`` URI.
    """
    if _is_url_parsed(parsed):
        return False

    # Explicit file:// scheme
    if parsed.scheme == "file":
        return True

    # UNC paths in the path component
    if parsed.path and parsed.path.startswith("\\\\"):
        return True

    # Windows drive letters: urlparse might interpret "C:" as scheme
    if len(parsed.scheme) == 1 and parsed.scheme.isalpha():
        # This looks like a Windows drive letter
        return True

    # No scheme means local path
    if not parsed.scheme and parsed.path is not None:
        return True

    return False


def is_url(path_or_url: str) -> bool:
    """
    Determine if a string is a network URL.

    Parameters
    ----------
    path_or_url : str
        Input string.

    Returns
    -------
    bool
        True if the string is an ``http``, ``https``, or ``s3`` URL with a
        non-empty netloc.
    """
    parsed = urlparse(path_or_url)
    return _is_url_parsed(parsed)


def is_file_path(path_or_url: str) -> bool:
    """
    Determine if a string is a local file path.

    Parameters
    ----------
    path_or_url : str
        Input string.

    Returns
    -------
    bool
        True if the string is a local path, a Windows drive/UNC path, or a
        ``file://`` URI.
    """
    parsed = urlparse(path_or_url)
    if _is_file_parsed(parsed):
        return True
    return _looks_like_windows_path(path_or_url)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI into bucket and key.

    Parameters
    ----------
    s3_uri : str
        URI like ``s3://bucket/key``. The key may be empty.

    Returns
    -------
    tuple of (str, str)
        ``(bucket, key)`` where key has no leading slash.

    Raises
    ------
    ValueError
        If `s3_uri` is not an ``s3://`` URI.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError("Not a valid S3 URI")
    return parsed.netloc, parsed.path.lstrip("/")


def as_pathlike(
    base: str,
) -> tuple[str, str | None, Path | PurePosixPath | PureWindowsPath]:
    """
    Parse a string into a normalized triplet.

    Parameters
    ----------
    base : str
        Input URI or path. Supports ``s3://``, ``file://``, POSIX, and
        Windows paths.

    Returns
    -------
    tuple
        ``(kind, bucket, path)`` where:
        - ``kind`` is ``"s3"`` or ``"file"``,
        - ``bucket`` is ``None`` for local,
        - ``path`` is ``PurePosixPath`` (S3), ``PureWindowsPath`` (Windows),
          or ``Path`` (local).

    Notes
    -----
    For ``file://server/share/path``, the netloc becomes a UNC prefix and
    the returned path is a native UNC path (``//server/share/path``).
    For Windows paths, uses ``PureWindowsPath`` for consistent
    cross-platform behavior.
    """
    u = urlparse(base)
    if u.scheme == "s3":
        return ("s3", u.netloc, PurePosixPath(u.path.lstrip("/")))
    if u.scheme == "file":
        p = u.path or ""
        if u.netloc:
            p = f"//{u.netloc}{p}"
        return ("file", None, Path(url2pathname(p)))
    # For Windows paths, use PureWindowsPath for consistent behavior
    # across platforms
    if _looks_like_windows_path(base):
        return ("file", None, PureWindowsPath(base))

    return ("file", None, Path(base))


def as_string(
    kind: str, bucket: str | None, path: Path | PurePosixPath | PureWindowsPath
) -> str:
    """
    Convert a normalized triplet back to a string.

    Parameters
    ----------
    kind : str
        Either ``"s3"`` or ``"file"``.
    bucket : str or None
        S3 bucket for ``kind == "s3"``. Ignored for local.
    path : Path, PurePosixPath, or PureWindowsPath
        Path object for the location.

    Returns
    -------
    str
        ``"s3://bucket/key"`` for S3 or a filesystem path for local.

    Raises
    ------
    ValueError
        If `kind` is not supported.
    """
    if kind == "s3":
        key = path.as_posix().lstrip("/")
        if not key or key == ".":
            return f"s3://{bucket}/" if bucket else "s3://"
        return f"s3://{bucket}/{key}" if bucket else f"s3://{key}"
    if kind == "file":
        return str(path)
    raise ValueError(f"Unsupported kind: {kind!r}")


def join_any(base: str, *parts: str) -> str:
    """
    Join path components under either scheme.

    Parameters
    ----------
    base : str
        Base URI or path.
    *parts : str
        Additional components to join.

    Returns
    -------
    str
        A string URI for S3 or a filesystem path for local.

    Notes
    -----
    For S3, backslashes in `parts` are normalized to forward slashes.
    For Windows paths, maintains consistent separator style with the base.
    """
    kind, bucket, p = as_pathlike(base)
    if kind == "s3":
        norm = [part.replace("\\", "/") for part in parts]
        joined = p.joinpath(*norm)
        return f"s3://{bucket}/{joined.as_posix()}"

    # For file paths, maintain separator consistency
    joined = p.joinpath(*parts)
    result = str(joined)

    # If the original base used backslashes and we're on a system that
    # might mix separators, ensure consistency
    if "\\" in base and "/" in result:
        # Convert forward slashes to backslashes to match base style
        result = result.replace("/", "\\")
    elif "/" in base and "\\" in result and not result.startswith("//"):
        # Convert backslashes to forward slashes, unless it's a UNC path
        result = result.replace("\\", "/")

    return result
