# S3 Integration Guide

aind-zarr-utils accepts S3 URIs directly. Public AIND data normally works with
anonymous access; private data can use a configured boto3 client.

## Public Data

```python
from aind_zarr_utils import Asset, Points, Space
from aind_s3_cache.json_utils import get_json

dataset = "s3://aind-open-data/dataset"

asset = Asset.from_root(dataset, anonymous=True)
ng_state = get_json(f"{dataset}/neuroglancer_state.json")

points = Points.from_neuroglancer(ng_state)
ccf = asset.transform(points, to=Space.CCF_MM)
```

`Asset.from_root()` loads `metadata.nd.json` and `processing.json`, resolves the
alignment-channel Zarr, and pre-opens it. Later calls to `image()`, `stub()`,
and `transform()` reuse the same opened Zarr.

## Images From S3

```python
asset = Asset.from_zarr("s3://aind-open-data/dataset/image.ome.zarr/0")

ants_img = asset.image(level=3, library="ants")
stub, size_ijk = asset.stub(level=0)
pipeline_stub, native_size_ijk = asset.stub(pipeline=True)
```

Use `stub()` for coordinate work when pixel data is not needed.

## Private Buckets

```python
import boto3
from aind_zarr_utils import Asset

s3_client = boto3.client("s3")

asset = Asset.from_root(
    "s3://private-bucket/dataset",
    anonymous=False,
    s3_client=s3_client,
    cache_dir="~/.aind-cache",
)
```

## Cache Directory

`cache_dir` is forwarded to transform-chain resolution and S3 resource access.
Use a persistent directory for repeated workflows:

```python
asset = Asset.from_zarr(
    "s3://aind-open-data/dataset/image.ome.zarr/0",
    cache_dir="~/.aind-cache",
)

# First access resolves/downloads transform files; subsequent accesses reuse
# the cached `TransformPaths` object on the Asset.
paths = asset.transforms
```

## Lower-Level JSON Loading

Metadata or Neuroglancer state can still be loaded directly with
`aind-s3-cache`:

```python
from aind_s3_cache.json_utils import get_json

metadata = get_json("s3://aind-open-data/dataset/metadata.nd.json")
processing = get_json("s3://aind-open-data/dataset/processing.json")
```
