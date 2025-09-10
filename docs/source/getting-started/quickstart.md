# Quick Start

Get up and running with aind-zarr-utils in minutes. This guide covers the most common use cases.

## 5-Minute Start

### 1. Install

```bash
pip install aind-zarr-utils
```

### 2. Load ZARR as Image

```python
from aind_zarr_utils.zarr import zarr_to_ants
from aind_zarr_utils.json_utils import get_json

# Load metadata and convert ZARR to ANTs image
metadata = get_json("s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/metadata.json")
zarr_uri = "s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44/exaspim.ome.zarr/0"

# Create ANTs image for analysis
ants_img = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit="millimeter")

print(f"Image shape: {ants_img.shape}")
print(f"Spacing: {ants_img.spacing} mm")
print(f"Origin: {ants_img.origin} mm")
```

### 3. Process Neuroglancer Annotations

```python
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

# Load Neuroglancer state (example structure)
ng_data = {
    "layers": {
        "region1": {
            "annotations": [
                {"point": [100, 200, 50]},  # [z, y, x] coordinates
                {"point": [120, 180, 60]}
            ]
        }
    }
}

# Convert to physical coordinates
physical_points, descriptions = neuroglancer_annotations_to_anatomical(
    ng_data, zarr_uri, metadata, scale_unit="millimeter"
)

print(f"Region1 points: {physical_points['region1']} mm (LPS coordinates)")
```

## Common Workflows

### Workflow 1: ZARR Analysis

```python
from aind_zarr_utils.zarr import zarr_to_sitk, zarr_to_sitk_stub
from aind_zarr_utils.json_utils import get_json
import numpy as np

# Step 1: Load metadata
metadata = get_json("s3://aind-open-data/dataset/metadata.json")
zarr_uri = "s3://aind-open-data/dataset/data.ome.zarr/0"

# Step 2: Check image properties without loading full data
stub_img, original_size = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
print(f"Full resolution: {original_size}")
print(f"Spacing: {stub_img.GetSpacing()} mm")

# Step 3: Load working resolution for analysis
sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3, scale_unit="millimeter")
print(f"Working resolution: {sitk_img.GetSize()}")

# Step 4: Basic analysis
array = sitk.GetArrayFromImage(sitk_img)
print(f"Intensity range: {array.min()} - {array.max()}")
```

### Workflow 2: Coordinate Transformation

```python
from aind_zarr_utils.annotations import annotation_indices_to_anatomical

# Your annotation data (voxel indices)
annotations = {
    "soma_locations": np.array([
        [100, 200, 50],   # [z, y, x] indices
        [120, 180, 60],
        [95, 210, 45]
    ]),
    "dendrite_tips": np.array([
        [105, 195, 55],
        [115, 175, 65]
    ])
}

# Transform to anatomical coordinates
physical_coords = annotation_indices_to_anatomical(
    annotations, zarr_uri, metadata, scale_unit="millimeter"
)

# Result: coordinates in LPS millimeters
for region, coords in physical_coords.items():
    print(f"{region}: {coords.shape[0]} points")
    print(f"  Range X: {coords[:, 0].min():.2f} - {coords[:, 0].max():.2f} mm")
    print(f"  Range Y: {coords[:, 1].min():.2f} - {coords[:, 1].max():.2f} mm") 
    print(f"  Range Z: {coords[:, 2].min():.2f} - {coords[:, 2].max():.2f} mm")
```

### Workflow 3: S3 Data Processing

```python
from aind_zarr_utils.s3_cache import CacheManager, get_local_path_for_resource
from aind_zarr_utils.json_utils import get_json
import json

# Process multiple datasets efficiently with caching
datasets = [
    "s3://aind-open-data/dataset1/metadata.json",
    "s3://aind-open-data/dataset2/metadata.json", 
    "s3://aind-open-data/dataset3/metadata.json"
]

results = []
with CacheManager(persistent=True, cache_dir="~/.aind_cache") as cm:
    for dataset_uri in datasets:
        # Download with caching
        result = get_local_path_for_resource(dataset_uri, cache_dir=cm.dir)
        
        # Process local file (much faster)
        with open(result.path) as f:
            metadata = json.load(f)
            
        session_id = metadata.get('session_id', 'unknown')
        results.append({
            "uri": dataset_uri,
            "session_id": session_id,
            "from_cache": result.from_cache
        })
        
        print(f"Processed {session_id} (cached: {result.from_cache})")

print(f"Processed {len(results)} datasets")
```

### Workflow 4: Pipeline Data Processing

```python
from aind_zarr_utils.pipeline_transformed import neuroglancer_to_ccf
from aind_zarr_utils.json_utils import get_json

# Load pipeline-processed data
zarr_metadata = get_json("s3://bucket/zarr_metadata.json")
processing_metadata = get_json("s3://bucket/processing.json")
neuroglancer_state = get_json("s3://bucket/neuroglancer_state.json")

# Transform annotations to CCF coordinates
ccf_points, descriptions = neuroglancer_to_ccf(
    neuroglancer_data=neuroglancer_state,
    zarr_uri="s3://bucket/data.ome.zarr/0",
    zarr_metadata=zarr_metadata,
    processing_metadata=processing_metadata,
    template_used="SmartSPIM-template_2024-05-16_11-26-14"
)

# Result: annotations in Allen CCF space
for layer, points in ccf_points.items():
    print(f"{layer}: {points.shape[0]} points in Allen CCF")
    
    # CCF coordinates are in RAS millimeters
    print(f"  X range: {points[:, 0].min():.1f} - {points[:, 0].max():.1f} mm")
    print(f"  Y range: {points[:, 1].min():.1f} - {points[:, 1].max():.1f} mm")
    print(f"  Z range: {points[:, 2].min():.1f} - {points[:, 2].max():.1f} mm")
```

## Essential Concepts

### Resolution Levels

ZARR files contain multiple resolution levels:

```python
# Check available levels
for level in range(6):
    try:
        stub, size = zarr_to_sitk_stub(zarr_uri, metadata, level=level)
        spacing = stub.GetSpacing()
        print(f"Level {level}: {size} voxels, {spacing} mm spacing")
    except:
        break

# Choose appropriate level:
# - Level 0: Full resolution (large, slow)
# - Level 3: Good for most analysis (recommended)
# - Level 5+: Quick previews only
```

### Coordinate Systems

aind-zarr-utils uses **LPS (Left-Posterior-Superior)** coordinates:

```python
# All functions return LPS coordinates
ants_img = zarr_to_ants(zarr_uri, metadata)
physical_points = neuroglancer_annotations_to_anatomical(ng_data, zarr_uri, metadata)

# LPS means:
# +X = Left direction
# +Y = Posterior direction  
# +Z = Superior direction

print(f"Point in LPS: {physical_points['layer1'][0]}")  # [left, posterior, superior]
```

### Units

Specify output units explicitly:

```python
# Micrometers (original acquisition units)
img_um = zarr_to_ants(zarr_uri, metadata, scale_unit="micrometer")

# Millimeters (standard for medical imaging)
img_mm = zarr_to_ants(zarr_uri, metadata, scale_unit="millimeter")

print(f"Micrometer spacing: {img_um.spacing}")
print(f"Millimeter spacing: {img_mm.spacing}")
```

## Data Sources

### Public AIND Data

Access public datasets without credentials:

```python
# Browse available datasets
public_bucket = "s3://aind-open-data/"

# Example datasets
datasets = [
    "exaspim_708373_2024-02-02_11-26-44",
    "smartspim_679848_2024-01-12_14-33-35",
    # ... more datasets
]

# No AWS credentials needed for public data
for dataset in datasets:
    metadata_uri = f"{public_bucket}{dataset}/metadata.json"
    metadata = get_json(metadata_uri)  # Works automatically
    print(f"Dataset: {metadata.get('session_id', dataset)}")
```

### Private S3 Data

Configure AWS credentials for private data:

```python
import boto3

# Option 1: Custom S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id='your_key',
    aws_secret_access_key='your_secret'
)

from aind_zarr_utils.json_utils import get_json_s3
data = get_json_s3("private-bucket", "data.json", s3_client=s3_client)

# Option 2: Environment variables (AWS_ACCESS_KEY_ID, etc.)
# get_json() will automatically use configured credentials
```

## Performance Tips

### 1. Use Appropriate Resolution

```python
# For development/testing: use level 3
img = zarr_to_ants(zarr_uri, metadata, level=3)

# For final analysis: use level 0 (if memory allows)
img = zarr_to_ants(zarr_uri, metadata, level=0)

# For coordinate system only: use stub
stub, size = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
```

### 2. Enable Caching

```python
from aind_zarr_utils.s3_cache import CacheManager

# Use persistent cache for repeated access
with CacheManager(cache_dir="~/.aind_cache") as cm:
    # All S3 operations will be cached
    metadata = get_json("s3://bucket/data.json", cache_dir=cm.dir)
```

### 3. Process Multiple Files Efficiently

```python
import concurrent.futures

def process_dataset(uri):
    metadata = get_json(f"{uri}/metadata.json") 
    return analyze_dataset(metadata)

# Parallel processing
uris = ["s3://bucket/dataset1", "s3://bucket/dataset2"]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_dataset, uris))
```

## Next Steps

Now that you're up and running:

1. **Explore User Guides**: Detailed documentation for specific features
   - [ZARR Conversion](../user-guide/zarr-conversion.md)
   - [S3 Integration](../user-guide/s3-integration.md)
   - [Coordinate Systems](../user-guide/coordinate-systems.md)

2. **Check API Reference**: Complete function documentation
   - [zarr module](../api-reference/zarr.rst)
   - [json_utils module](../api-reference/json_utils.rst)

3. **See Examples**: Real-world usage scenarios
   - [Examples Overview](examples.md)

4. **Get Help**: 
   - [Troubleshooting](../reference/troubleshooting.md)
   - [GitHub Issues](https://github.com/AllenNeuralDynamics/aind-zarr-utils/issues)

## Common Next Steps

**For Image Analysis**:
```python
# Convert ZARR to your preferred format
ants_img = zarr_to_ants(zarr_uri, metadata, level=3)
sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3)

# Use with your analysis pipeline
registered_img = ants.registration(ants_img, template)
```

**For Annotation Processing**:
```python
# Transform annotations to physical space
physical_coords = neuroglancer_annotations_to_anatomical(
    ng_data, zarr_uri, metadata
)

# Use coordinates for further analysis
distance_matrix = calculate_distances(physical_coords['neurons'])
```

**For Pipeline Integration**:
```python
# Get pipeline-corrected spatial domain
ccf_coords = neuroglancer_to_ccf(
    ng_data, zarr_uri, zarr_metadata, processing_metadata, template
)

# Register to Allen CCF atlas
atlas_regions = query_atlas(ccf_coords)
```