# Examples

These examples use the recommended `Asset` / `Points` API.

## Inspect An Asset

```python
from aind_zarr_utils import Asset

asset = Asset.from_root("s3://aind-open-data/dataset")

print(asset.alignment_zarr_uri)
print(asset.source_zarr_uri)
print(asset.metadata.keys())
print(asset.processing.keys())
```

## Load A Working-Resolution Image

```python
asset = Asset.from_zarr("s3://aind-open-data/dataset/image.ome.zarr/0")

sitk_img = asset.image(level=3)
ants_img = asset.image(level=3, library="ants")

print(sitk_img.GetSize())
print(ants_img.shape)
```

## Work With Coordinates Without Pixels

```python
stub, native_size_ijk = asset.stub(level=0)

index = (100, 200, 50)
physical = stub.TransformIndexToPhysicalPoint(index)

print(native_size_ijk)
print(physical)
```

For pipeline-corrected coordinates:

```python
pipeline_stub, native_size_ijk = asset.stub(pipeline=True)
pipeline_physical = pipeline_stub.TransformIndexToPhysicalPoint(index)
```

## Transform Neuroglancer Annotations To CCF

```python
from aind_zarr_utils import Asset, Points, Space
from aind_s3_cache.json_utils import get_json

ng_state = get_json("s3://aind-open-data/dataset/neuroglancer_state.json")

asset = Asset.from_neuroglancer(ng_state)
points = Points.from_neuroglancer(ng_state)

ccf = asset.transform(points, to=Space.CCF_MM)

for layer, values in ccf.values.items():
    print(f"{layer}: {values.shape[0]} points")
```

## Transform Manual Points

```python
import numpy as np
from aind_zarr_utils import Points, Space

points = Points(
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

raw_lps = asset.transform(points, to=Space.LS_ANATOMICAL_MM)
pipeline_lps = asset.transform(points, to=Space.LS_PIPELINE_ANATOMICAL_MM)
ccf = asset.transform(points, to=Space.CCF_MM)
```

## Transform SWC Coordinates

```python
swc_points = Points.from_swc(
    swc_array,
    axis_order="zyx",
    units="micrometer",
)

indices = asset.transform(swc_points, to=Space.ZARR_INDICES)
ccf = asset.transform(swc_points, to=Space.CCF_MM)
```

## Anchor A Non-Pipeline Origin

```python
from aind_zarr_utils import Origin

img = asset.image(
    level=3,
    origin=Origin.at_corner("RAS", (0.0, 0.0, 0.0)),
)
```

Pipeline outputs derive their origin from `processing.json`, so `origin` is only
valid when `pipeline=False`.

## Use A Private S3 Bucket

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

## Legacy Explicit-Metadata Calls

The lower-level functions are still available for workflows that already manage
metadata explicitly:

```python
from aind_zarr_utils.zarr import zarr_to_sitk

sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3)
```

New code should prefer `Asset` so opened Zarr state and transform paths are
cached across the workflow.
