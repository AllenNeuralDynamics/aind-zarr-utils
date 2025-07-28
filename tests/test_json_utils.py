import json
import os
import tempfile
from unittest import mock

import pytest

from aind_zarr_utils import json_utils


class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception("HTTP Error")


class DummyS3Client:
    def __init__(self, data):
        self.data = data

    def get_object(self, Bucket, Key):
        return {"Body": self}

    def read(self, *args, **kwargs):
        return json.dumps(self.data).encode()

    def __iter__(self):
        return iter([json.dumps(self.data).encode()])

    def __next__(self):
        raise StopIteration

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def readlines(self):
        return [json.dumps(self.data).encode()]

    def readline(self):
        return json.dumps(self.data).encode()

    def seek(self, *args, **kwargs):
        pass

    def tell(self):
        return 0

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def __bool__(self):
        return True

    def __len__(self):
        return 1

    def __contains__(self, item):
        return item in self.data

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.data))

    def __call__(self, *args, **kwargs):
        return self.data

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __setattr__(self, key, value):
        if key == "data":
            object.__setattr__(self, key, value)
        else:
            setattr(self.data, key, value)


def test_is_url_and_is_file_path():
    assert json_utils.is_url("http://example.com")
    assert json_utils.is_url("https://example.com")
    assert json_utils.is_url("s3://bucket/key")
    assert not json_utils.is_url("/tmp/file.json")
    assert json_utils.is_file_path("/tmp/file.json")
    assert not json_utils.is_file_path("http://example.com")


def test_parse_s3_uri():
    bucket, key = json_utils.parse_s3_uri("s3://mybucket/mykey.json")
    assert bucket == "mybucket"
    assert key == "mykey.json"
    with pytest.raises(ValueError):
        json_utils.parse_s3_uri("http://example.com/file.json")


def test_get_json_url(monkeypatch):
    data = {"foo": "bar"}
    monkeypatch.setattr(
        json_utils.requests, "get", lambda url: DummyResponse(data)
    )
    result = json_utils.get_json_url("http://example.com/file.json")
    assert result == data


def test_get_json_s3():
    data = {"hello": "world"}
    client = DummyS3Client(data)
    # Patch json.load to read from DummyS3Client
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json_s3("bucket", "key", s3_client=client)
    assert result == data


def test_get_json_s3_uri():
    data = {"a": 1}
    client = DummyS3Client(data)
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json_s3_uri(
            "s3://bucket/key.json", s3_client=client
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


def test_get_json_url_integration(monkeypatch):
    data = {"foo": "bar"}
    monkeypatch.setattr(
        json_utils.requests, "get", lambda url: DummyResponse(data)
    )
    result = json_utils.get_json("http://example.com/file.json")
    assert result == data


def test_get_json_s3_uri_integration():
    data = {"a": 1}
    client = DummyS3Client(data)
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json("s3://bucket/key.json", s3_client=client)
    assert result == data


def test_get_json_s3_bucket_key():
    data = {"b": 2}
    client = DummyS3Client(data)
    with mock.patch("json.load", return_value=data):
        result = json_utils.get_json("bucket", "key", client)
    assert result == data


def test_get_json_invalid():
    with pytest.raises(FileNotFoundError):
        json_utils.get_json(":not_a_valid_path:")
