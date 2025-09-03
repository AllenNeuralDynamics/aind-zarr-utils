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
    file_url_or_bucket: str, key: Optional[str] = None, *args, **kwargs
) -> dict:
    """
    Read a JSON file from a local path, URL, or S3.

    Parameters
    ----------
    file_url_or_bucket : str
        The file path, URL, or S3 bucket name.
    key : str, optional
        The key for the S3 object. Required if reading from S3.
    *args : tuple
        Additional arguments for S3 client or HTTP requests.
    **kwargs : dict
        Additional keyword arguments for S3 client or HTTP requests.

    Returns
    -------
    dict
        The JSON object.

    Raises
    ------
    ValueError
        If the input is not a valid file path, URL, or S3 URI.
    """
    if key is None:
        parsed = urlparse(file_url_or_bucket)
        if _is_url_parsed(parsed):
            if parsed.scheme == "s3":
                data = get_json_s3_uri(file_url_or_bucket, *args, **kwargs)
            else:
                data = get_json_url(file_url_or_bucket)
        elif _is_file_parsed(parsed):
            with open(file_url_or_bucket, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported URL or file path: {file_url_or_bucket}"
            )
    else:
        data = get_json_s3(file_url_or_bucket, key, *args)
    return data
