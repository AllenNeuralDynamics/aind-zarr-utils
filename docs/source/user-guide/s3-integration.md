# S3 Integration Guide

This guide covers working with S3 resources in aind-zarr-utils, including anonymous access, caching, and credential management.

## Overview

aind-zarr-utils provides seamless integration with AWS S3 for:

- **Anonymous access** to public buckets like `aind-open-data`
- **Efficient caching** with ETag-based validation
- **Flexible authentication** supporting various credential methods
- **Multi-source JSON loading** from S3, HTTP, and local files

## Quick Start

### Anonymous Access to Public Data

Most AIND data is publicly accessible without credentials:

```python
from aind_zarr_utils.json_utils import get_json

# Access public AIND data (no credentials needed)
metadata = get_json("s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/metadata.json")
zarr_uri = "s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/exaspim.ome.zarr/0"

print(f"Session ID: {metadata['session_id']}")
```

### Basic S3 URI Patterns

```python
# AIND public data bucket
s3_uri = "s3://aind-open-data/dataset_name/file.json"

# Parse S3 URIs
from aind_zarr_utils.uri_utils import parse_s3_uri
bucket, key = parse_s3_uri(s3_uri)
print(f"Bucket: {bucket}, Key: {key}")
```

## JSON Loading from S3

### Unified get_json Function

The `get_json` function automatically detects and handles S3 URIs:

```python
from aind_zarr_utils.json_utils import get_json

# Automatically uses anonymous S3 access for public buckets
data = get_json("s3://aind-open-data/path/to/metadata.json")

# Also works with HTTP URLs and local files
data = get_json("https://example.com/data.json")
data = get_json("/local/path/data.json")
```

### Explicit S3 Functions

For more control over S3 access:

```python
from aind_zarr_utils.json_utils import get_json_s3, get_json_s3_uri

# Direct bucket/key access
data = get_json_s3(
    bucket="aind-open-data",
    key="dataset/metadata.json",
    anonymous=True  # Explicit anonymous access
)

# S3 URI with custom client
data = get_json_s3_uri(
    uri="s3://aind-open-data/dataset/metadata.json",
    anonymous=True
)
```

## S3 Caching

### Automatic Caching

aind-zarr-utils provides intelligent caching to avoid repeated downloads:

```python
from aind_zarr_utils.s3_cache import get_local_path_for_resource

# First call downloads file
result = get_local_path_for_resource("s3://aind-open-data/large-file.json")
print(f"Downloaded to: {result.path}")
print(f"From cache: {result.from_cache}")  # False on first call

# Second call uses cached version
result = get_local_path_for_resource("s3://aind-open-data/large-file.json")
print(f"From cache: {result.from_cache}")  # True on subsequent calls
```

### Cache Management

#### Temporary Cache (Auto-cleanup)

```python
from aind_zarr_utils.s3_cache import CacheManager

# Temporary cache - automatically cleaned up
with CacheManager(persistent=False) as cm:
    result = get_local_path_for_resource(
        "s3://aind-open-data/file.json",
        cache_dir=cm.dir
    )
    # Use result.path...
# Cache directory automatically deleted
```

#### Persistent Cache

```python
# Persistent cache in home directory
cache_dir = "~/.aind_zarr_cache"
with CacheManager(cache_dir=cache_dir) as cm:
    result = get_local_path_for_resource(
        "s3://aind-open-data/file.json",
        cache_dir=cm.dir
    )
    # Files persist after context exit
```

#### Custom Cache Directory

```python
import tempfile

# Custom temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    result = get_local_path_for_resource(
        "s3://aind-open-data/file.json",
        cache_dir=temp_dir
    )
    print(f"Cached in: {temp_dir}")
```

### ETag-based Validation

The caching system uses ETags to avoid unnecessary downloads:

```python
# First download
result = get_local_path_for_resource("s3://bucket/file.json")
print(f"Source: {result.source}")  # "s3"
print(f"From cache: {result.from_cache}")  # False

# File unchanged on S3 - uses cache
result = get_local_path_for_resource("s3://bucket/file.json")  
print(f"From cache: {result.from_cache}")  # True

# File changed on S3 - re-downloads automatically
# (ETag validation happens transparently)
```

## Authentication Methods

### Anonymous Access (Default)

For public buckets, no credentials are needed:

```python
from aind_zarr_utils.json_utils import get_json

# Automatic anonymous access for public buckets
data = get_json("s3://aind-open-data/public/data.json")
```

### Custom S3 Client

For private buckets or custom configurations:

```python
import boto3
from aind_zarr_utils.json_utils import get_json_s3

# Create custom S3 client with credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id='your-access-key',
    aws_secret_access_key='your-secret-key',
    region_name='us-west-2'
)

# Use custom client
data = get_json_s3(
    bucket="private-bucket",
    key="data.json",
    s3_client=s3_client
)
```

### AWS Credential Chain

boto3 automatically uses the standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM roles (when running on EC2)
4. AWS SSO

```python
# No explicit client needed - uses credential chain
data = get_json_s3(
    bucket="private-bucket",
    key="data.json",
    anonymous=False  # Use credential chain
)
```

## Advanced Usage

### Batch Processing with Caching

Efficiently process multiple S3 files:

```python
from aind_zarr_utils.s3_cache import CacheManager, get_local_path_for_resource

s3_files = [
    "s3://aind-open-data/dataset1/metadata.json",
    "s3://aind-open-data/dataset2/metadata.json", 
    "s3://aind-open-data/dataset3/metadata.json"
]

with CacheManager(persistent=True, cache_dir="~/.aind_cache") as cm:
    for s3_uri in s3_files:
        result = get_local_path_for_resource(s3_uri, cache_dir=cm.dir)
        
        # Process local file (much faster than repeated S3 access)
        with open(result.path) as f:
            data = json.load(f)
            
        print(f"Processed {s3_uri} (cached: {result.from_cache})")
```

### Custom Download Configuration

```python
import boto3
from botocore.config import Config

# Custom configuration for large files
config = Config(
    retries={'max_attempts': 5},
    max_pool_connections=50
)

s3_client = boto3.client('s3', config=config)

result = get_local_path_for_resource(
    "s3://bucket/large-file.zarr",
    s3_client=s3_client,
    cache_dir="/fast/storage/cache"
)
```

### URI Manipulation

```python
from aind_zarr_utils.uri_utils import join_any, as_pathlike, as_string

# Build S3 paths programmatically
base_uri = "s3://aind-open-data/experiment_123"
metadata_uri = join_any(base_uri, "processed", "metadata.json")
zarr_uri = join_any(base_uri, "processed", "data.ome.zarr", "0")

print(f"Metadata: {metadata_uri}")
print(f"ZARR: {zarr_uri}")

# Parse and reconstruct URIs
kind, bucket, path = as_pathlike(metadata_uri)
new_uri = as_string(kind, bucket, path / "additional_file.json")
```

## Performance Optimization

### Cache Strategy

Choose cache strategy based on usage patterns:

```python
# Short-term processing - use temporary cache
with CacheManager(persistent=False) as cm:
    # Process files, cache auto-deleted
    pass

# Repeated analysis - use persistent cache  
with CacheManager(cache_dir="~/.aind_cache") as cm:
    # Files remain cached between sessions
    pass

# High-performance storage - use fast local disk
with CacheManager(cache_dir="/nvme/cache") as cm:
    # Use fastest available storage
    pass
```

### Parallel Downloads

For multiple files, consider concurrent downloads:

```python
import concurrent.futures
from aind_zarr_utils.s3_cache import get_local_path_for_resource

def download_file(s3_uri):
    return get_local_path_for_resource(s3_uri, cache_dir="~/.cache")

s3_uris = [
    "s3://bucket/file1.json",
    "s3://bucket/file2.json", 
    "s3://bucket/file3.json"
]

# Download files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(download_file, s3_uris))

for uri, result in zip(s3_uris, results):
    print(f"{uri} -> {result.path}")
```

## Error Handling

### Common S3 Errors

```python
from botocore.exceptions import ClientError, NoCredentialsError
from aind_zarr_utils.json_utils import get_json

try:
    data = get_json("s3://private-bucket/file.json")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'NoSuchBucket':
        print("Bucket does not exist")
    elif error_code == 'NoSuchKey':
        print("File not found in bucket")
    elif error_code == 'AccessDenied':
        print("Access denied - check permissions")
    else:
        print(f"S3 error: {error_code}")
except NoCredentialsError:
    print("AWS credentials not found")
```

### Validation and Fallbacks

```python
from aind_zarr_utils.uri_utils import is_url
from aind_zarr_utils.json_utils import get_json

def safe_load_json(uri_or_path):
    """Safely load JSON with fallback handling."""
    try:
        if is_url(uri_or_path):
            # Try S3/HTTP first
            return get_json(uri_or_path)
        else:
            # Local file
            with open(uri_or_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load {uri_or_path}: {e}")
        return None

# Usage
data = safe_load_json("s3://bucket/file.json")
if data is None:
    # Handle error case
    pass
```

## Best Practices

1. **Use Anonymous Access**: For `aind-open-data` and other public buckets
2. **Enable Persistent Caching**: For repeated analysis workflows  
3. **Choose Appropriate Cache Location**: Fast local storage for large files
4. **Handle Errors Gracefully**: Network issues are common with cloud storage
5. **Monitor Cache Size**: Clean up caches periodically to save disk space
6. **Use Concurrent Downloads**: For processing multiple files
7. **Validate URIs**: Check URI format before making requests

## Cache Management

### Cache Location

```python
# Check default cache locations
import tempfile
import os

print(f"Temp directory: {tempfile.gettempdir()}")
print(f"Home directory: {os.path.expanduser('~')}")

# Recommended locations
cache_locations = {
    "temporary": tempfile.gettempdir(),
    "persistent": os.path.expanduser("~/.aind_cache"), 
    "high_performance": "/nvme/cache",  # If available
}
```

### Cache Cleanup

```python
import shutil
from pathlib import Path

def cleanup_cache(cache_dir, max_age_days=30):
    """Remove cache files older than max_age_days."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return
        
    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    
    for file_path in cache_path.rglob("*"):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
            print(f"Removed old cache file: {file_path}")

# Clean up old cache files
cleanup_cache("~/.aind_cache", max_age_days=30)
```