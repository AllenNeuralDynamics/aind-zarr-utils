# aind-zarr-utils

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

Utilities for working with AIND Zarr assets, metadata, Neuroglancer
annotations, and SmartSPIM pipeline coordinate transforms.

## Recommended API

The primary entry point is `Asset`. It discovers an asset's alignment-channel
Zarr, loads `metadata.nd.json` and `processing.json`, opens the Zarr once, and
reuses that state for image, stub, and point-transform workflows.

```python
from aind_zarr_utils import Asset, Points, Space
from aind_s3_cache.json_utils import get_json

zarr_uri = "s3://aind-open-data/dataset/image.ome.zarr/0"
ng_state = get_json("s3://aind-open-data/dataset/neuroglancer_state.json")

asset = Asset.from_zarr(zarr_uri)
points = Points.from_neuroglancer(ng_state)

ccf = asset.transform(points, to=Space.CCF_MM)
print(ccf.values)
```

If you already have metadata and processing dictionaries loaded, use the no-I/O
constructor. `alignment_zarr_uri` should be the Zarr used by the alignment
pipeline; `source_zarr_uri` is optional provenance for the Zarr you started
from.

```python
from aind_zarr_utils import Asset

asset = Asset(
    alignment_zarr_uri="s3://bucket/asset/alignment.ome.zarr/0",
    metadata=metadata,
    processing=processing,
    source_zarr_uri="s3://bucket/asset/acquisition.ome.zarr/0",
)
```

## Images And Stubs

`Asset.image()` returns a SimpleITK image by default and can also return an ANTs
image. `Asset.stub()` returns a header-only SimpleITK image for coordinate
operations without loading pixel data.

```python
from aind_zarr_utils import Asset, Origin

asset = Asset.from_root("s3://aind-open-data/dataset")

sitk_img = asset.image(level=3)
ants_img = asset.image(level=3, library="ants")

stub, size_ijk = asset.stub(level=0)
pipeline_stub, native_size_ijk = asset.stub(pipeline=True)

anchored = asset.image(
    level=3,
    origin=Origin.at_corner("RAS", (0.0, 0.0, 0.0)),
)
```

`origin` is only accepted when `pipeline=False`. Pipeline images and stubs use
the pipeline-corrected origin from `processing.json`.

## Coordinate Spaces

`Points` stores named `(N, 3)` arrays plus a `Space` tag. Constructors validate
shape and coerce arrays to floating point.

```python
import numpy as np
from aind_zarr_utils import Asset, Points, Space

asset = Asset.from_zarr("s3://bucket/asset/image.ome.zarr/0")

indices = Points(
    {"soma": np.array([[100, 200, 50], [120, 180, 60]])},
    Space.ZARR_INDICES,
)

pipeline_mm = asset.transform(indices, to=Space.LS_PIPELINE_ANATOMICAL_MM)
ccf_mm = asset.transform(indices, to=Space.CCF_MM)
round_trip = asset.transform(ccf_mm, to=Space.ZARR_INDICES)
```

Supported spaces are:

- `Space.ZARR_INDICES`: continuous level-0 `(z, y, x)` Zarr indices
- `Space.LS_SCALED_MM`: spacing-scaled light-sheet coordinates
- `Space.LS_ANATOMICAL_MM`: raw Zarr anatomical LPS millimeters
- `Space.LS_PIPELINE_ANATOMICAL_MM`: pipeline-corrected LPS millimeters
- `Space.CCF_MM`: Allen CCF LPS millimeters

SWC coordinates can enter the same graph without opening image data first:

```python
swc_points = Points.from_swc(swc_array, axis_order="zyx", units="micrometer")
ccf_points = asset.transform(swc_points, to=Space.CCF_MM)
```

## Legacy Functions

The lower-level modules remain available for compatibility:

- `aind_zarr_utils.zarr`: `zarr_to_ants`, `zarr_to_sitk`, `zarr_to_sitk_stub`
- `aind_zarr_utils.neuroglancer`: Neuroglancer annotation readers
- `aind_zarr_utils.pipeline_transformed`: explicit metadata transform helpers

The auto-metadata convenience helpers in `pipeline_transformed` are deprecated
in favor of `Asset.from_zarr(...)` / `Asset.from_root(...)` plus
`Asset.transform(...)`.

## Installation

```bash
pip install aind-zarr-utils
```

For development:

```bash
git clone https://github.com/AllenNeuralDynamics/aind-zarr-utils.git
cd aind-zarr-utils
uv sync
```

## Development

Run the core checks with:

```bash
uv run ruff format
uv run ruff check
uv run mypy
uv run pytest
uv run --group docs sphinx-build docs/source docs/build/html -W --keep-going
```

Pull requests use Angular-style commit messages:

```text
<type>(<scope>): <short summary>
```
