# Metadata reader integration fixtures

These files are copied from the reader's retained fixtures so consumer tests run
without a sibling test directory or live services. `production-captures.json`
records the original locators, capture dates, response hashes and excerpt boundaries.

- `771432-docdb-historical-v1.json` and `771432-docdb-converted-v2.json` are
  field-preserving excerpts of paired public DocDB records captured on 2026-11-10.
  Their acquisition and processing sections retain distinct source representations.
  The pair does not certify conversion lineage or source authority.
- `791132-s3-processing.json` is the complete public processing file for
  `s3://aind-open-data/SmartSPIM_791132_2026-09-18_14-10-48_stitched_2026-10-29_21-23-13`.
  Its registration producer is 0.0.37 and its processing core is 2.3.3.
- `v2.9.0-exaspim_acquisition.json` and `v3.0.0-exaspim_acquisition.json` were
  serialized and validated with their respective upstream release models from
  [v2.9.0](https://github.com/AllenInstitute/biodata-schema/blob/v2.9.0/examples/exaspim_acquisition.py)
  and [v3.0.0](https://github.com/AllenInstitute/biodata-schema/blob/v3.0.0/examples/exaspim_acquisition.py).
  They are model examples; production v3 coverage remains pending adoption.
