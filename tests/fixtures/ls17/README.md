# LS-17 metadata fixtures

These are unchanged copies of the reader's public processing excerpts and full
NGFF attribute captures for SmartSPIM assets 771432 and 837772. Capture locators,
dates, hashes, and excerpt boundaries are recorded in
[`../metadata_reader/production-captures.json`](../metadata_reader/production-captures.json)
and the reader's fixture documentation.

771432 has no pyramid translations and fusion component version 0.0.2. 837772 has
voxel-count translations in micrometer fields and fusion version 0.0.4. The
integration tests construct small, synthetic block-mean pyramids using the first
four captured metadata levels. Their explicit evidence describes that synthetic
construction; tests do not claim to rerun the production pixel measurements.

No tests read live stores. The original fixture files and generated Zarr attributes
remain unchanged when the correction is applied.
