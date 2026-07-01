# Quick Start

This guide covers the recommended `Asset` / `Points` API.

## Install

```bash
pip install aind-zarr-utils
```

## Build An Asset

Use `Asset.from_zarr()` when you have any acquisition Zarr URI. The constructor
loads `metadata.nd.json` and `processing.json`, resolves the alignment-channel
Zarr, and pre-opens it so later operations reuse the same handle.

```python
from aind_zarr_utils import Asset

zarr_uri = "s3://aind-open-data/dataset/image.ome.zarr/0"
asset = Asset.from_zarr(zarr_uri)

print(asset.alignment_zarr_uri)
print(asset.source_zarr_uri)
```

Use `Asset.from_root()` when you already have the asset root:

```python
asset = Asset.from_root("s3://aind-open-data/dataset")
```

If metadata is already loaded, use the no-I/O constructor:

```python
asset = Asset(
    alignment_zarr_uri="s3://bucket/dataset/alignment.ome.zarr/0",
    metadata=metadata,
    processing=processing,
)
```

## Images And Stubs

```python
from aind_zarr_utils import Origin

# SimpleITK by default
sitk_img = asset.image(level=3)

# ANTs when requested
ants_img = asset.image(level=3, library="ants")

# Header-only SimpleITK image for coordinate work
stub, size_ijk = asset.stub(level=0)

# Pipeline-corrected spatial domain from processing.json
pipeline_stub, native_size_ijk = asset.stub(pipeline=True)
pipeline_img = asset.image(level=3, library="sitk", pipeline=True)

# Explicit origin control for non-pipeline images
anchored = asset.image(
    level=3,
    origin=Origin.at_corner("RAS", (0.0, 0.0, 0.0)),
)
```

`origin` and `pipeline=True` are intentionally exclusive. Pipeline outputs use
the corrected origin implied by the pipeline metadata.

## Neuroglancer To CCF

```python
from aind_zarr_utils import Asset, Points, Space
from aind_s3_cache.json_utils import get_json

ng_state = get_json("s3://aind-open-data/dataset/neuroglancer_state.json")

asset = Asset.from_neuroglancer(ng_state)
points = Points.from_neuroglancer(ng_state)

ccf_points = asset.transform(points, to=Space.CCF_MM)

for layer, values in ccf_points.values.items():
    print(layer, values.shape)
```

The same `Points` can be projected to intermediate spaces:

```python
raw_lps = asset.transform(points, to=Space.LS_ANATOMICAL_MM)
pipeline_lps = asset.transform(points, to=Space.LS_PIPELINE_ANATOMICAL_MM)
back_to_indices = asset.transform(ccf_points, to=Space.ZARR_INDICES)
```

## Manual Points

```python
import numpy as np
from aind_zarr_utils import Points, Space

indices = Points(
    {
        "soma": np.array(
            [
                [100, 200, 50],
                [120, 180, 60],
            ]
        )
    },
    Space.ZARR_INDICES,
)

ccf = asset.transform(indices, to=Space.CCF_MM)
```

`Points` validates every array as `(N, 3)` and coerces values to `float`.

## SWC Coordinates

```python
swc_points = Points.from_swc(
    swc_array,
    axis_order="zyx",
    units="micrometer",
)

ccf = asset.transform(swc_points, to=Space.CCF_MM)
indices = asset.transform(swc_points, to=Space.ZARR_INDICES)
```

The SWC constructor preserves continuous sub-voxel coordinates. It does not
round to integer Zarr indices.

## S3 And Caching

`Asset.from_zarr()`, `Asset.from_root()`, and `Asset.from_neuroglancer()` accept
the same S3 access knobs:

```python
asset = Asset.from_zarr(
    "s3://aind-open-data/dataset/image.ome.zarr/0",
    anonymous=True,
    cache_dir="~/.aind-cache",
)
```

For private buckets, pass a configured boto3 S3 client:

```python
import boto3

s3_client = boto3.client("s3")
asset = Asset.from_root(
    "s3://private-bucket/dataset",
    anonymous=False,
    s3_client=s3_client,
)
```

## Legacy Functions

The lower-level functions are still available when you need explicit metadata
arguments:

```python
from aind_zarr_utils.zarr import zarr_to_sitk

sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3)
```

The auto-metadata helpers in `pipeline_transformed` are deprecated. Prefer
`Asset.from_zarr(...)` plus `Asset.transform(...)` for new code.
