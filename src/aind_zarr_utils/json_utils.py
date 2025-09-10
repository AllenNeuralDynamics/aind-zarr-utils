"""S3 utilities for reading and writing JSON files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse
from warnings import warn

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config

from aind_zarr_utils.uri_utils import (
    _is_file_parsed,
    _is_url_parsed,
    parse_s3_uri,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def get_json_s3(
    bucket: str,
    key: str,
    s3_client: Optional["S3Client"] = None,
    anonymous: bool = False,
    anon: bool | None = None,
) -> dict:
    """
    Retrieve a JSON object from an S3 bucket.

    Parameters
    ----------
    bucket : str
        The name of the S3 bucket.
    key : str
        The key of the JSON object in the bucket.
    s3_client : boto3.client, optional
        An existing S3 client. If None, a new client is created.
    anonymous : bool, optional
        If True, the S3 client will be created in anonymous mode.

    deprecated parameters
    ---------------------
    anon : bool, optional
        If True, the S3 client will be created in anonymous mode.

    Returns
    -------
    dict
        The JSON object.
    """
    if s3_client is None:
        if anon is not None:
            # Deprecated parameter 'anon' is now 'anonymous'
            anonymous = anon
            warn(
                DeprecationWarning(
                    "The 'anon' parameter is deprecated, use "
                    "'anonymous' instead."
                )
            )
        if anonymous:
            s3_client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            )
        else:
            s3_client = boto3.client("s3")
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    json_data: dict = json.load(resp["Body"])
    return json_data


def get_json_s3_uri(
    uri: str,
    s3_client: Optional["S3Client"] = None,
    anonymous: bool = False,
) -> dict:
    """
    Retrieve a JSON object from an S3 URI.

    Parameters
    ----------
    uri : str
        The S3 URI of the JSON object.
    s3_client : boto3.client, optional
        An existing S3 client. If None, a new client is created.
    anonymous : bool, optional
        If True, the S3 client will be created in anonymous mode.

    Returns
    -------
    dict
        The JSON object.
    """
    bucket, key = parse_s3_uri(uri)
    return get_json_s3(bucket, key, s3_client=s3_client, anonymous=anonymous)


def get_json_url(url: str) -> dict:
    """
    Retrieve a JSON object from a URL.

    Parameters
    ----------
    url : str
        The URL of the JSON object.

    Returns
    -------
    dict
        The JSON object.

    Raises
    ------
    HTTPError
        If the HTTP request fails.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises an error if the download failed
    json_data: dict = response.json()
    return json_data


def get_json(
    file_url_or_bucket: str, key: str | None = None, *args, **kwargs
) -> dict:
    """
    Read a JSON file from local path, file:// URI, HTTP(S), or S3.

    Parameters
    ----------
    file_url_or_bucket : str
        Local path, ``file://`` URI, ``http(s)://`` URL, ``s3://`` URI, or
        S3 bucket name (when `key` is provided).
    key : str or None
        S3 key when `file_url_or_bucket` is a bucket name. Ignored when an
        ``s3://`` URI is provided.
    *args
        Extra args forwarded to S3/HTTP helpers.
    **kwargs
        Extra kwargs forwarded to S3/HTTP helpers.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If the input is not a supported path/URL or fetch fails.
    """
    # Case 1: explicit bucket + key form for S3
    if key is not None:
        return get_json_s3(file_url_or_bucket, key, *args, **kwargs)

    s = file_url_or_bucket
    parsed = urlparse(s)

    # Case 2: network URLs we recognize (http/https/s3)
    if _is_url_parsed(parsed):
        if parsed.scheme == "s3":
            return get_json_s3_uri(s, *args, **kwargs)
        return get_json_url(s)

    # Case 3: file:// URIs and all local paths (POSIX/Windows/UNC)
    if parsed.scheme == "file":
        # Convert file:// to a local path string (handles UNC too).
        p = parsed.path or ""
        if parsed.netloc:
            p = f"//{parsed.netloc}{p}"
        local = Path(url2pathname(p))
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: treat as local/Windows path (works cross-platform).
    if _is_file_parsed(parsed) or _looks_like_windows_path(s):
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported URL or file path: {s!r}")
