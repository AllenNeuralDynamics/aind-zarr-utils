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
