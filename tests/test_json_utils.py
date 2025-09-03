import json
import os
import tempfile
from unittest import mock

import pytest

from aind_zarr_utils import json_utils


def test_is_url_and_is_file_path():
    from urllib.parse import urlparse

    from aind_zarr_utils.uri_utils import _is_file_parsed, _is_url_parsed

    assert _is_url_parsed(urlparse("http://example.com"))
    assert _is_url_parsed(urlparse("https://example.com"))
    assert _is_url_parsed(urlparse("s3://bucket/key"))
    assert not _is_url_parsed(urlparse("/tmp/file.json"))
    assert _is_file_parsed(urlparse("/tmp/file.json"))
    assert not _is_file_parsed(urlparse("http://example.com"))


def test_parse_s3_uri():
    from aind_zarr_utils.uri_utils import parse_s3_uri

    bucket, key = parse_s3_uri("s3://mybucket/mykey.json")
    assert bucket == "mybucket"
    assert key == "mykey.json"
    with pytest.raises(ValueError):
        parse_s3_uri("http://example.com/file.json")


def test_get_json_url(monkeypatch, mock_requests_get):
    monkeypatch.setattr(
        json_utils.requests, "get", lambda url: mock_requests_get(url)
    )
    result = json_utils.get_json_url("http://example.com/file.json")
    assert result == {"mocked": "data", "url": "http://example.com/file.json"}


def test_get_json_s3(mock_s3_client):
    data = {"hello": "world"}
    mock_s3_client.data = data
    # Patch json.load to read from UnifiedS3Client
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json_s3(
            "bucket", "key", s3_client=mock_s3_client
        )
    assert result == data


def test_get_json_s3_uri(mock_s3_client):
    data = {"a": 1}
    mock_s3_client.data = data
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json_s3_uri(
            "s3://bucket/key.json", s3_client=mock_s3_client
        )
    assert result == data


def test_get_json_local_file():
    data = {"x": 42}
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        json.dump(data, f)
        fname = f.name
    try:
        result = json_utils.get_json(fname)
        assert result == data
    finally:
        os.remove(fname)


def test_get_json_url_integration(monkeypatch, mock_requests_get):
    monkeypatch.setattr(
        json_utils.requests, "get", lambda url: mock_requests_get(url)
    )
    result = json_utils.get_json("http://example.com/file.json")
    assert result == {"mocked": "data", "url": "http://example.com/file.json"}


def test_get_json_s3_uri_integration(mock_s3_client):
    data = {"a": 1}
    mock_s3_client.data = data
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json(
            "s3://bucket/key.json", s3_client=mock_s3_client
        )
    assert result == data


def test_get_json_s3_bucket_key(mock_s3_client):
    data = {"b": 2}
    mock_s3_client.data = data
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json("bucket", "key", mock_s3_client)
    assert result == data


def test_get_json_invalid():
    with pytest.raises(FileNotFoundError):
        json_utils.get_json(":not_a_valid_path:")
