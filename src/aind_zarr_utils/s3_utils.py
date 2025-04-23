"""S3 utilities for reading and writing JSON files."""

import json
from urllib.parse import urlparse

import boto3


def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError("Not a valid S3 URI")
    return parsed.netloc, parsed.path.lstrip("/")


def get_s3_json(bucket, key, s3_client=None):
    if s3_client is None:
        s3_client = boto3.client("s3")
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return json.load(resp["Body"])


def get_s3_json_uri(uri, s3_client=None):
    bucket, key = parse_s3_uri(uri)
    return get_s3_json(bucket, key, s3_client=s3_client)
