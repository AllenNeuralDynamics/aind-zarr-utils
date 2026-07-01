# aind-zarr-utils

Utilities for working with AIND Zarr assets, metadata, Neuroglancer
annotations, and SmartSPIM pipeline coordinate transforms.

The recommended API is asset-centric:

```python
from aind_zarr_utils import Asset, Points, Space
from aind_s3_cache.json_utils import get_json

ng_state = get_json("s3://aind-open-data/dataset/neuroglancer_state.json")
asset = Asset.from_zarr("s3://aind-open-data/dataset/image.ome.zarr/0")

points = Points.from_neuroglancer(ng_state)
ccf_points = asset.transform(points, to=Space.CCF_MM)
```

`Asset` owns metadata discovery, Zarr opening, transform-chain resolution, and
pipeline overlays. `Points` carries named `(N, 3)` arrays together with an
explicit coordinate-space tag, so one `asset.transform(...)` call replaces the
older family of `*_to_ccf` / `ccf_to_*` helper calls.

## Key Features

- **Asset discovery**: Build an `Asset` from a Zarr URI, asset root, or
  Neuroglancer state.
- **Image construction**: Create SimpleITK or ANTs images through
  `Asset.image()`, with optional pipeline corrections.
- **Header-only stubs**: Use `Asset.stub()` for coordinate operations without
  loading image pixels.
- **Point transforms**: Move `Points` between Zarr indices, light-sheet
  anatomical space, pipeline-corrected anatomical space, and Allen CCF.
- **Legacy compatibility**: Lower-level `zarr`, `neuroglancer`, and
  `pipeline_transformed` functions remain available for explicit workflows.

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started/index
user-guide/index
tutorials/index
api-reference/index
contributing/index
reference/index
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
