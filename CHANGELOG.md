## v0.15.0 (2026-04-28)

### Feat

- New asset-centric API. The package's recommended entry point is now
  `Asset` (with `Asset.from_zarr`, `Asset.from_root`,
  `Asset.from_neuroglancer`, plus a no-I/O `Asset(zarr_uri, metadata,
  processing)` constructor). A single `Asset` instance opens its Zarr
  exactly once and resolves transform-chain paths exactly once,
  regardless of how many methods participate in a workflow.
- `Points` + `Space` make coordinate spaces first-class. Five spaces
  are recognised: `ZARR_INDICES`, `LS_SCALED_MM`, `LS_ANATOMICAL_MM`,
  `LS_PIPELINE_ANATOMICAL_MM`, `CCF_MM`. `Asset.transform(points,
  to=Space.X)` walks a small graph of named edges to project them.
  Every previous combination of `*_to_ccf` / `ccf_to_*` /
  `*_to_indices` is now expressible as one `transform` call.
- `Origin` value type replaces the legacy
  `(set_origin, set_corner, set_corner_lps)` triple on the image
  builders: `Origin.default()`, `Origin.at(point)`,
  `Origin.at_corner(code, lps_point)`.
- `Points.from_neuroglancer` and `Points.from_swc` provide one-line
  conversion from NG state and raw SWC arrays into the package's
  canonical representation.

### Refactor

- The package source is reorganised into focused subpackages:
  `io/` (Zarr opening, metadata parsing, processing-JSON parsing,
  transform-chain resolution, URI helpers), `domain/` (overlay
  protocol, selector, concrete overlays), `formats/` (Neuroglancer
  and SWC readers), and the new top-level modules `asset.py`,
  `points.py`, `image.py`, `origin.py`. The legacy modules
  (`zarr.py`, `neuroglancer.py`, `pipeline_domain_selector.py`,
  `pipeline_transformed.py`) keep working as re-export shims and
  continue to pass all existing tests.
- The SimpleITK and ANTs pipeline-overlay paths (~320 LOC of
  duplication) collapse into a single type-dispatching
  `apply_pipeline_overlays` plus a small pure helper
  (`_to_ants_convention`) that encodes the level > 0 spacing-reverse
  + opposite-corner origin recompute. The level > 0 ANTs math is now
  pinned by dedicated regression tests.

### Deprecations

The following auto-discovery convenience helpers emit
`DeprecationWarning` and will be removed in a future release. Each
points to its `Asset`-based replacement:

- `mimic_pipeline_zarr_to_anatomical_stub` (still callable but
  superseded by `Asset(...).stub(pipeline=True)`; not warned this
  release because of internal use).
- `neuroglancer_to_ccf_auto_metadata` →
  `Asset.from_neuroglancer(ng).transform(Points.from_neuroglancer(ng), to=Space.CCF_MM)`
- `swc_data_to_ccf_auto_metadata` →
  `Asset.from_root(uri).transform(Points.from_swc(...), to=Space.CCF_MM)`
- `indices_to_ccf_auto_metadata` →
  `Asset.from_zarr(uri).transform(Points(idx, Space.ZARR_INDICES), to=Space.CCF_MM)`
- `ccf_to_indices_auto_metadata` →
  `Asset.from_zarr(uri).transform(Points(ccf, Space.CCF_MM), to=Space.ZARR_INDICES)`

The lower-level free functions in `pipeline_transformed` and `zarr`
(e.g. `neuroglancer_to_ccf`, `ccf_to_indices`, `zarr_to_sitk`,
`zarr_to_sitk_stub`, `mimic_pipeline_zarr_to_anatomical_stub`) are
unchanged for now and remain part of the package surface.

## v0.14.0 (2026-02-11)

### Feat

- allow template base to be paths, as well as strings

## v0.13.2 (2025-12-10)

### Fix

- return numpy-ordered indices when converting from ccf

## v0.13.1 (2025-12-10)

### Fix

- float conversion

## v0.13.0 (2025-12-10)

### Feat

- transform CCF coordinates to zarr indices (#52)

## v0.12.0 (2025-12-03)

### Feat

- Refactor/global to scaled (#51)

## v0.11.6 (2025-12-03)

### Fix

- add auto-metadata indices to ccf function

## v0.11.5 (2025-11-07)

### Fix

- sanitize zarr urls

## v0.11.4 (2025-11-06)

### Fix

- allow image sources to be dicts

## v0.11.3 (2025-11-04)

### Fix

- convert to native endianness before creating images

## v0.11.2 (2025-10-24)

### Fix

- Modify existing anatomical images to be like pipeline

## v0.11.1 (2025-10-21)

### Fix

- loosen zarr dep range

## v0.11.0 (2025-10-19)

### BREAKING CHANGE

- Removed `Header` from `pipeline_domain_selector`

### Refactor

- Remove Header to aind-anatomical-utils

## v0.10.4 (2025-10-14)

### Fix

- remove prefixes like "zarr://" from neuroglancer sources

## v0.10.3 (2025-10-10)

### Fix

- add compat for aind-registration-utils 0.4

## v0.10.2 (2025-10-09)

### Fix

- Add connection kwargs to metadata convenience functions

## v0.10.1 (2025-10-09)

### Fix

- update domain selector to newest pipeline version

## v0.10.0 (2025-10-09)


- move compute_origin_for_corner to aind_anatomical_utils

## v0.9.0 (2025-10-08)

### Feat

- return ijk size from `mimic_pipeline_zarr_to_anatomical_stub`

## v0.8.0 (2025-09-29)

### Feat

- add pipeline sitk and ants images

## v0.7.0 (2025-09-28)

### Feat

- add image transform accessors (#27)

## v0.6.0 (2025-09-19)

### BREAKING CHANGE

- Rename `neuroglancer_to_ccf_pipeline_files` to
`neuroglancer_to_ccf_auto_metadata`, as the old name was nearly uninterpretable.

### Feat

- Add support for SWCs (#26)

## v0.5.1 (2025-09-18)

### Fix

- Return arrays not dicts for ccf points

## v0.5.0 (2025-09-17)

### Feat

- define a public-facing API

## v0.4.0 (2025-09-17)

### BREAKING CHANGE

- s3_cache, json_utils, and uri_utils have been removed

### Feat

- Remove s3_cache, json_utils, and uri_utils

## v0.3.1 (2025-09-11)

### Fix

- Attempt to fix file parsing on windows (#21)

## v0.3.0 (2025-09-10)

### Feat

- add convenience function to load all files from neuroglancer

## v0.2.0 (2025-09-09)

### Feat

- **domain**: add pipeline domain/transform utils; S3 caching; docs

## v0.1.4 (2025-08-07)

## v0.1.3 (2025-07-29)

### Fix

- Typing (#10)

## v0.1.2 (2025-07-01)

## v0.1.1 (2025-06-30)

## v0.1.0 (2025-06-30)

### Feat

- add annotations

## v0.0.11 (2025-06-30)

## v0.0.10 (2025-06-26)

## v0.0.9 (2025-06-26)

## v0.0.8 (2025-06-26)

## v0.0.7 (2025-06-26)

## v0.0.6 (2025-06-26)

## v0.0.5 (2025-06-13)

## v0.0.4 (2025-04-24)

### Fix

- some problems with variable names

## v0.0.3 (2025-04-23)

## v0.0.2 (2025-04-23)

## v0.0.1 (2025-04-23)

## v0.0.0 (2025-04-23)
