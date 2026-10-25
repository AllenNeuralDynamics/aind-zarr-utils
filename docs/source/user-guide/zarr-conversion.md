# Zarr Conversion Guide

This guide covers image and stub construction with the recommended
`Asset` API. The lower-level `zarr_to_*` functions remain available for code
that already manages metadata explicitly.

## Build An Asset

```python
from aind_zarr_utils import Asset

asset = Asset.from_zarr("s3://aind-open-data/dataset/image.ome.zarr/0")
```

`Asset.from_zarr()` discovers the asset root, loads `metadata.nd.json` and
`processing.json`, resolves the alignment-channel Zarr, and pre-opens it. Use
`Asset.from_root()` when the asset root is already known.

## Convert To Images

```python
# SimpleITK by default
sitk_img = asset.image(level=3)

# ANTs when requested
ants_img = asset.image(level=3, library="ants")
```

Resolution levels follow the OME-Zarr multiscale convention:

- **Level 0**: full resolution
- **Level 3**: typical working resolution
- **Level 5+**: preview resolution

Every level is placed in one physical frame. A level voxel is the mean of the
level-0 voxels beneath it, so its center sits where theirs does: at level `L`,
voxel `i` is centered on level-0 index `2**L * i + (2**L - 1) / 2`, and the
level's origin is offset from level 0's by half a level voxel less half a
level-0 voxel. The Zarr's own `translation` is not read, because the SmartSPIM
fusion writes it missing or in the wrong units.

## Read Concurrency

Reading a level from an S3-backed Zarr is one dask `compute()` call, and it is
latency-bound: each chunk is a separate S3 request. A serial read of a large
level can take minutes, while 16–32 workers typically cut it by an order of
magnitude. The blosc/zstd codec releases the GIL, so threads help decompression
too. Past ~32 workers the gains reverse.

Pass `compute_kwargs` — forwarded verbatim to dask's `compute()`:

```python
# Per asset: applies to every read
asset = Asset.from_zarr(
    "s3://aind-open-data/dataset/image.ome.zarr/0",
    compute_kwargs={"scheduler": "threads", "num_workers": 16},
)
sitk_img = asset.image(level=3)

# Per call: replaces the asset-level default
sitk_img = asset.image(level=3, compute_kwargs={"num_workers": 32})
```

`compute_kwargs` is also accepted by `zarr_to_ants`, `zarr_to_sitk`,
`zarr_to_numpy`, and the `mimic_pipeline_zarr_to_*` /
`base_and_pipeline_zarr_to_*` functions.

Setting `num_workers` alone selects the `"threads"` scheduler, so
`{"num_workers": 16}` is enough.

Deployments that cannot edit their call sites can set the options in the
environment instead:

```bash
export AIND_ZARR_COMPUTE_WORKERS=16      # -> num_workers, threads scheduler
export AIND_ZARR_COMPUTE_SCHEDULER=threads
```

Explicit `compute_kwargs` always win over the environment; the variables only
fill in keys the caller left unset. Passing nothing and setting nothing
reproduces dask's defaults.

## Header-Only Stubs

Use stubs for coordinate transformations without loading pixel data:

```python
stub_img, native_size_ijk = asset.stub(level=0)

print(stub_img.GetSize())       # (1, 1, 1)
print(native_size_ijk)          # level-0 image dimensions
print(stub_img.GetSpacing())
print(stub_img.GetOrigin())
```

For pipeline-corrected coordinates:

```python
pipeline_stub, native_size_ijk = asset.stub(pipeline=True)
```

Pipeline stubs use `processing.json` to reproduce the spatial domain used by
registration.

## Origin Control

Use `Origin` for explicit non-pipeline origins:

```python
from aind_zarr_utils import Origin

at_origin = asset.image(level=3, origin=Origin.at((0.0, 0.0, 0.0)))

ras_anchored = asset.image(
    level=3,
    origin=Origin.at_corner("RAS", (0.0, 0.0, 0.0)),
)
```

An `Origin` places the full-resolution grid: `Origin.at` is the first level-0
voxel's center, and `Origin.at_corner` anchors a level-0 corner voxel's center.
The level read is then placed over its level-0 sources, so the same origin
gives the same frame at every level.

`origin` is rejected when `pipeline=True`, because pipeline outputs must use the
pipeline-corrected origin.

## Coordinate System Details

All physical coordinates are ITK LPS millimeters by default:

- **L**: left direction is positive X
- **P**: posterior direction is positive Y
- **S**: superior direction is positive Z

SimpleITK and ANTs expose image size/shape differently, but the physical domain
represented by `asset.image(..., library="sitk")` and
`asset.image(..., library="ants")` is the same.

## Legacy Explicit-Metadata API

If you need to pass metadata directly, the original functions remain available:

```python
from aind_zarr_utils.zarr import zarr_to_ants, zarr_to_sitk, zarr_to_sitk_stub

ants_img = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit="millimeter")
sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3, scale_unit="millimeter")
stub_img, native_size_ijk = zarr_to_sitk_stub(
    zarr_uri,
    metadata,
    level=0,
    scale_unit="millimeter",
)
```

New code should prefer `Asset` so metadata, opened Zarr state, and transform
paths are cached in one object.
