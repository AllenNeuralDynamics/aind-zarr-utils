# S3 Integration Guide

This guide covers using aind-zarr-utils with S3 data sources.

## Overview

aind-zarr-utils functions accept S3 URIs directly and handle authentication automatically for public AIND data.

## Basic Usage

All aind-zarr-utils functions work seamlessly with S3 URIs:

```python
from aind_zarr_utils.zarr import zarr_to_ants
from aind_s3_cache.json_utils import get_json

# Load metadata and ZARR data from S3
metadata = get_json("s3://aind-open-data/dataset/metadata.json")
zarr_uri = "s3://aind-open-data/dataset/data.ome.zarr/0"

# Convert directly from S3 ZARR
ants_img = zarr_to_ants(zarr_uri, metadata, level=3)
```

## Public Data Access

AIND public data requires no authentication:

```python
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

# Public bucket - no credentials needed
ng_data = get_json("s3://aind-open-data/dataset/neuroglancer_state.json")
metadata = get_json("s3://aind-open-data/dataset/metadata.json")
zarr_uri = "s3://aind-open-data/dataset/data.ome.zarr/0"

# Transform annotations from public data
points, descriptions = neuroglancer_annotations_to_anatomical(
    ng_data, zarr_uri, metadata
)
```

## Pipeline Functions with S3

Pipeline correction functions work with S3 processing metadata:

```python
from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_anatomical_stub

# Load processing metadata from S3
processing_data = get_json("s3://aind-open-data/dataset/processing.json")
metadata = get_json("s3://aind-open-data/dataset/metadata.json")

# Create pipeline-corrected stub from S3 data
stub = mimic_pipeline_zarr_to_anatomical_stub(
    zarr_uri="s3://aind-open-data/dataset/data.ome.zarr/0",
    metadata=metadata,
    processing_data=processing_data
)
```

## Authentication

For private buckets, configure AWS credentials using standard methods:

- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- AWS credentials file: `~/.aws/credentials`
- IAM roles (when running on EC2)

## Performance Notes

- S3 data is automatically cached locally for efficient repeated access
- Use appropriate resolution levels (`level` parameter) to balance speed and quality
- Consider using `zarr_to_sitk_stub()` for coordinate transformations without loading full image data

## Example Workflow

```python
from aind_zarr_utils.pipeline_transformed import neuroglancer_to_ccf
from aind_s3_cache.json_utils import get_json

# Complete workflow with S3 data
dataset_base = "s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44"

ng_data = get_json(f"{dataset_base}/neuroglancer_state.json")
metadata = get_json(f"{dataset_base}/metadata.json")
processing_data = get_json(f"{dataset_base}/processing.json")

# Transform Neuroglancer annotations to CCF coordinates
ccf_points, descriptions = neuroglancer_to_ccf(
    neuroglancer_data=ng_data,
    zarr_uri=f"{dataset_base}/exaspim.ome.zarr/0",
    zarr_metadata=metadata,
    processing_metadata=processing_data,
    template_used="SmartSPIM-template_2024-05-16_11-26-14"
)

print(f"Transformed {len(ccf_points)} annotation layers to CCF coordinates")
```