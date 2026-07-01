# Coordinate Systems Guide

aind-zarr-utils represents point coordinates with two pieces of information:

- a named `(N, 3)` array stored in `Points.values`
- an explicit `Space` tag describing what those three columns mean

`Asset.transform(points, to=...)` moves points between supported spaces.

## Supported Spaces

```python
from aind_zarr_utils import Space
```

| Space | Meaning |
|-------|---------|
| `Space.ZARR_INDICES` | Continuous level-0 `(z, y, x)` Zarr indices |
| `Space.LS_SCALED_MM` | Light-sheet coordinates after multiplying indices by spacing |
| `Space.LS_ANATOMICAL_MM` | Raw Zarr anatomical LPS millimeters |
| `Space.LS_PIPELINE_ANATOMICAL_MM` | Pipeline-corrected anatomical LPS millimeters |
| `Space.CCF_MM` | Allen CCF LPS millimeters |

## Basic Transform

```python
import numpy as np
from aind_zarr_utils import Asset, Points, Space

asset = Asset.from_zarr("s3://aind-open-data/dataset/image.ome.zarr/0")

points = Points(
    {"soma": np.array([[100, 200, 50]])},
    Space.ZARR_INDICES,
)

raw_lps = asset.transform(points, to=Space.LS_ANATOMICAL_MM)
pipeline_lps = asset.transform(points, to=Space.LS_PIPELINE_ANATOMICAL_MM)
ccf = asset.transform(points, to=Space.CCF_MM)
```

`Points` validates every array as `(N, 3)` and coerces values to floating
point, so sub-voxel coordinates are preserved.

## LPS Physical Coordinates

Physical outputs use ITK LPS millimeters:

- positive X moves left
- positive Y moves posterior
- positive Z moves superior

Raw anatomical space and pipeline anatomical space are both LPS millimeter
spaces. They differ in the header used to interpret Zarr indices:

- `Space.LS_ANATOMICAL_MM` uses the raw Zarr metadata header
- `Space.LS_PIPELINE_ANATOMICAL_MM` uses the pipeline-corrected header from
  `processing.json`

## Neuroglancer

Neuroglancer annotations enter the graph as Zarr indices:

```python
from aind_zarr_utils import Asset, Points, Space

asset = Asset.from_neuroglancer(ng_state)
points = Points.from_neuroglancer(ng_state)

ccf = asset.transform(points, to=Space.CCF_MM)
```

Descriptions from annotation layers are preserved on the returned `Points`
object when available.

## SWC

SWC coordinates enter as spacing-scaled light-sheet millimeters:

```python
points = Points.from_swc(
    swc_array,
    axis_order="zyx",
    units="micrometer",
)

indices = asset.transform(points, to=Space.ZARR_INDICES)
ccf = asset.transform(points, to=Space.CCF_MM)
```

Unlike the legacy SWC helper, `Points.from_swc()` does not round to integer
indices.

## Images And Stubs

Use images when you need pixels:

```python
sitk_img = asset.image(level=3)
ants_img = asset.image(level=3, library="ants")
```

Use stubs when you need only the coordinate system:

```python
stub, native_size_ijk = asset.stub(level=0)
pipeline_stub, native_size_ijk = asset.stub(pipeline=True)
```

SimpleITK and ANTs expose image size/shape differently, but the physical domain
is the same for matching asset, level, and pipeline settings.

## Legacy Helpers

The older helpers are still available for explicit-metadata workflows:

```python
from aind_zarr_utils.annotations import annotation_indices_to_anatomical
from aind_zarr_utils.zarr import zarr_to_sitk_stub

stub, native_size_ijk = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
raw_lps = annotation_indices_to_anatomical(stub, {"soma": indices})
```

New code should prefer `Points` and `Asset.transform()` so the coordinate space
is carried with the data.
