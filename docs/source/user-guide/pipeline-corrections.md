# Pipeline Corrections Guide

SmartSPIM pipeline versions can differ in the spatial domain used for
registration. `Asset` exposes the pipeline-corrected domain without requiring
callers to manually thread `metadata.nd.json`, `processing.json`, opened Zarr
handles, and overlay selectors through each function.

## Pipeline-Corrected Stubs

```python
from aind_zarr_utils import Asset

asset = Asset.from_root("s3://aind-open-data/dataset")

stub, native_size_ijk = asset.stub(pipeline=True)

print(stub.GetSpacing())
print(stub.GetOrigin())
print(stub.GetDirection())
```

The returned SimpleITK image is header-only. It carries the same
pipeline-corrected spacing, origin, and direction that the registration
pipeline used, but it does not load image pixels.

## Pipeline-Corrected Images

```python
sitk_img = asset.image(level=3, pipeline=True)
ants_img = asset.image(level=3, library="ants", pipeline=True)
```

`origin` is intentionally rejected when `pipeline=True`; the origin comes from
the corrected pipeline header.

The registration placed the pyramid level it registered (level 3), so the
overlays anchor that level's grid. Level 0 and every other level are derived
from it, each over its level-0 sources. Anchoring a corner of the level-0 grid
instead misses by the level's half-voxel offset plus the voxels the pyramid
truncated at that end: 3.5–10.5 level-0 voxels per axis at level 3 (6–42 µm
for 1.8–4 µm voxels), depending on the acquisition's orientation and size.

## Points In Pipeline Space

```python
from aind_zarr_utils import Points, Space

points = Points.from_neuroglancer(ng_state)

pipeline_lps = asset.transform(points, to=Space.LS_PIPELINE_ANATOMICAL_MM)
ccf = asset.transform(points, to=Space.CCF_MM)
```

`Space.LS_PIPELINE_ANATOMICAL_MM` is the light-sheet anatomical space after
pipeline overlays. This is the space used immediately before the ANTs
registration chain to CCF.

## Overlay System

The lower-level overlay system remains available for explicit testing and
custom rules:

```python
from aind_anatomical_utils.anatomical_volume import AnatomicalHeader
from aind_zarr_utils.domain.selector import apply_overlays, get_selector

selector = get_selector()
overlays = selector.select(version=4, meta=asset.metadata)

# Overlays run on the registered level's grid, so the level is 0 relative to it.
base_stub, size_ijk = asset.stub(level=3, pipeline=False)
base_header = AnatomicalHeader.from_sitk(base_stub, size_ijk)

corrected_header, applied = apply_overlays(
    base_header,
    overlays,
    dict(asset.metadata),
    registration_multiscale_no=0,
    zarr_import_version=4,
)
```

Most workflows should use `Asset.stub(pipeline=True)` or
`Asset.image(pipeline=True)` instead.

## Legacy Compatibility

The explicit helper remains available:

```python
from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_anatomical_stub

stub, native_size_ijk = mimic_pipeline_zarr_to_anatomical_stub(
    asset.alignment_zarr_uri,
    dict(asset.metadata),
    dict(asset.processing),
)
```

New code should prefer `Asset.stub(pipeline=True)` so the same opened Zarr and
metadata are reused throughout the workflow.
