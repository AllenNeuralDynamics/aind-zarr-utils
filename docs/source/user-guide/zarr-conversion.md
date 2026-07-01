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
