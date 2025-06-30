"""S3 utilities for reading and writing JSON files."""

import json
from urllib.parse import urlparse

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config


def _is_url_parsed(parsed):
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


def _is_file_parsed(parsed):
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
        parsed.scheme == "file" or (not parsed.scheme and parsed.path)
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


def parse_s3_uri(s3_uri):
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


def get_json_s3(bucket, key, s3_client=None, anon=False):
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
    anon : bool, optional
        If True, the S3 client will be created in anonymous mode.

    Returns
    -------
    dict
        The JSON object.
    """
    if s3_client is None:
        if anon:
            s3_client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            )
        else:
            s3_client = boto3.client("s3")
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return json.load(resp["Body"])


def get_json_s3_uri(uri, s3_client=None):
    """
    Retrieve a JSON object from an S3 URI.

    Parameters
    ----------
    uri : str
        The S3 URI of the JSON object.
    s3_client : boto3.client, optional
        An existing S3 client. If None, a new client is created.

    Returns
    -------
    dict
        The JSON object.
    """
    bucket, key = parse_s3_uri(uri)
    return get_json_s3(bucket, key, s3_client=s3_client)


def get_json_url(url):
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
    return response.json()


def get_json(file_url_or_bucket, key=None, *args, **kwargs):
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
