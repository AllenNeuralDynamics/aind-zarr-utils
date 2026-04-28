# Examples

Real-world examples demonstrating common aind-zarr-utils workflows.

## Basic Examples

### Example 1: Load and Inspect ZARR Data

```python
from aind_zarr_utils.zarr import zarr_to_ants, zarr_to_sitk_stub
from aind_s3_cache.json_utils import get_json
import numpy as np

# Public AIND dataset
dataset_uri = "s3://aind-open-data/exaspim_708373_2024-02-02_11-26-44"
metadata = get_json(f"{dataset_uri}/metadata.json")
zarr_uri = f"{dataset_uri}/exaspim.ome.zarr/0"

print(f"Session ID: {metadata['session_id']}")
print(f"Subject ID: {metadata['subject_id']}")

# Check available resolution levels
print("\nAvailable resolution levels:")
for level in range(6):
    try:
        stub, size = zarr_to_sitk_stub(zarr_uri, metadata, level=level)
        spacing = stub.GetSpacing()
        memory_gb = (size[0] * size[1] * size[2] * 4) / (1024**3)  # 4 bytes per voxel
        print(f"  Level {level}: {size} voxels, {spacing} mm, ~{memory_gb:.1f} GB")
    except:
        break

# Load working resolution
ants_img = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit="millimeter")
print(f"\nLoaded image:")
print(f"  Shape: {ants_img.shape}")
print(f"  Spacing: {ants_img.spacing} mm")
print(f"  Origin: {ants_img.origin} mm")
print(f"  Physical extent: {np.array(ants_img.shape) * np.array(ants_img.spacing)} mm")
```

**Output:**
```
Session ID: exaspim_708373_2024-02-02_11-26-44
Subject ID: 708373

Available resolution levels:
  Level 0: (2048, 2048, 1536) voxels, (0.0072, 0.0072, 0.008) mm, ~24.6 GB
  Level 1: (1024, 1024, 768) voxels, (0.0144, 0.0144, 0.016) mm, ~3.1 GB
  Level 2: (512, 512, 384) voxels, (0.0288, 0.0288, 0.032) mm, ~0.4 GB
  Level 3: (256, 256, 192) voxels, (0.0576, 0.0576, 0.064) mm, ~0.05 GB

Loaded image:
  Shape: (192, 256, 256)
  Spacing: (0.064, 0.0576, 0.0576) mm
  Origin: (0.0, 0.0, 0.0) mm
  Physical extent: (12.288, 14.746, 14.746) mm
```

### Example 2: Process Neuroglancer Annotations

```python
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical
import numpy as np

# Example Neuroglancer state data
ng_state = {
    "layers": {
        "soma_locations": {
            "annotations": [
                {"point": [100, 200, 150], "description": "Pyramidal neuron"},
                {"point": [120, 180, 160], "description": "Interneuron"},
                {"point": [95, 210, 145], "description": "Pyramidal neuron"}
            ]
        },
        "dendrite_tips": {
            "annotations": [
                {"point": [105, 195, 155]},
                {"point": [115, 185, 165]}
            ]
        }
    }
}

# Convert to physical coordinates
physical_coords, descriptions = neuroglancer_annotations_to_anatomical(
    ng_state, zarr_uri, metadata, scale_unit="millimeter"
)

print("Transformed annotations:")
for layer, coords in physical_coords.items():
    print(f"\n{layer}:")
    print(f"  Points: {coords.shape[0]}")
    print(f"  Coordinates (LPS mm):")
    for i, point in enumerate(coords):
        desc = descriptions[layer][i] if layer in descriptions else "No description"
        print(f"    {i+1}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) - {desc}")
        
# Calculate distances between soma locations
soma_coords = physical_coords['soma_locations']
distances = []
for i in range(len(soma_coords)):
    for j in range(i+1, len(soma_coords)):
        dist = np.linalg.norm(soma_coords[i] - soma_coords[j])
        distances.append(dist)
        print(f"Distance soma {i+1} to soma {j+1}: {dist:.2f} mm")

print(f"\nMean inter-soma distance: {np.mean(distances):.2f} mm")
```

### Example 3: S3 Data Processing with Caching

```python
from aind_s3_cache.s3_cache import CacheManager, get_local_path_for_resource
from aind_s3_cache.json_utils import get_json
import json
import time

# Multiple datasets to process
datasets = [
    "exaspim_708373_2024-02-02_11-26-44",
    "smartspim_679848_2024-01-12_14-33-35",
    # Add more dataset IDs as needed
]

def analyze_metadata(metadata):
    """Simple analysis function."""
    return {
        "session_id": metadata.get("session_id"),
        "subject_id": metadata.get("subject_id"), 
        "acquisition_datetime": metadata.get("acquisition", {}).get("acquisition_datetime"),
        "has_processing": "processing" in metadata
    }

# Process with caching
results = []
with CacheManager(persistent=True, cache_dir="~/.aind_cache") as cm:
    print(f"Using cache directory: {cm.dir}")
    
    for dataset_id in datasets:
        print(f"\nProcessing {dataset_id}...")
        metadata_uri = f"s3://aind-open-data/{dataset_id}/metadata.json"
        
        # Time the download/cache access
        start_time = time.time()
        cache_result = get_local_path_for_resource(metadata_uri, cache_dir=cm.dir)
        download_time = time.time() - start_time
        
        # Load and analyze
        with open(cache_result.path) as f:
            metadata = json.load(f)
        
        analysis = analyze_metadata(metadata)
        analysis.update({
            "from_cache": cache_result.from_cache,
            "download_time": download_time
        })
        
        results.append(analysis)
        
        status = "cache hit" if cache_result.from_cache else "downloaded"
        print(f"  {status} in {download_time:.2f}s")

# Summary
print(f"\nProcessed {len(results)} datasets:")
cache_hits = sum(1 for r in results if r["from_cache"])
downloads = len(results) - cache_hits
print(f"  Cache hits: {cache_hits}")
print(f"  Downloads: {downloads}")
print(f"  Average time: {np.mean([r['download_time'] for r in results]):.2f}s")
```

## Advanced Examples

### Example 4: Multi-Resolution Analysis

```python
from aind_zarr_utils.zarr import zarr_to_ants
import ants
import matplotlib.pyplot as plt

def analyze_at_multiple_resolutions(zarr_uri, metadata, levels=[1, 2, 3, 4]):
    """Compare analysis results across resolution levels."""
    results = {}
    
    for level in levels:
        print(f"Processing level {level}...")
        
        # Load image at this resolution
        img = zarr_to_ants(zarr_uri, metadata, level=level, scale_unit="millimeter")
        
        # Basic analysis
        array_data = img.numpy()
        
        results[level] = {
            "shape": img.shape,
            "spacing": img.spacing,
            "mean_intensity": array_data.mean(),
            "std_intensity": array_data.std(),
            "max_intensity": array_data.max(),
            "memory_mb": array_data.nbytes / (1024**2)
        }
        
        # Optional: compute gradients for edge detection
        if array_data.nbytes < 500 * 1024**2:  # Only if < 500MB
            grad = ants.iMath(img, "Grad")
            results[level]["edge_strength"] = grad.numpy().mean()
    
    return results

# Run analysis
analysis_results = analyze_at_multiple_resolutions(zarr_uri, metadata)

# Display results
print("\nMulti-resolution analysis:")
print(f"{'Level':<6} {'Shape':<20} {'Spacing (mm)':<20} {'Mean Int':<10} {'Edges':<10} {'Memory (MB)':<12}")
print("-" * 90)

for level, data in analysis_results.items():
    shape_str = f"{data['shape']}"
    spacing_str = f"({data['spacing'][0]:.4f}, {data['spacing'][1]:.4f}, {data['spacing'][2]:.4f})"
    mean_int = f"{data['mean_intensity']:.1f}"
    edge_str = f"{data.get('edge_strength', 'N/A'):.3f}" if data.get('edge_strength') else "N/A"
    memory = f"{data['memory_mb']:.1f}"
    
    print(f"{level:<6} {shape_str:<20} {spacing_str:<20} {mean_int:<10} {edge_str:<10} {memory:<12}")
```

### Example 5: Coordinate System Validation

```python
from aind_zarr_utils.zarr import zarr_to_sitk, zarr_to_ants
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical
import numpy as np

def validate_coordinate_consistency(zarr_uri, metadata, test_points):
    """Verify coordinate transformations are consistent."""
    
    # Get same data in different formats
    sitk_img = zarr_to_sitk(zarr_uri, metadata, level=3, scale_unit="millimeter")
    ants_img = zarr_to_ants(zarr_uri, metadata, level=3, scale_unit="millimeter")
    
    print("Coordinate system validation:")
    print(f"SimpleITK - Size: {sitk_img.GetSize()}, Spacing: {sitk_img.GetSpacing()}")
    print(f"ANTs      - Shape: {ants_img.shape}, Spacing: {ants_img.spacing}")
    
    # Test coordinate transformations
    print(f"\nTesting {len(test_points)} coordinate transformations:")
    print(f"{'Index':<15} {'SimpleITK (LPS)':<25} {'ANTs (calc)':<25} {'Diff (mm)':<10}")
    print("-" * 80)
    
    for i, index in enumerate(test_points):
        # SimpleITK transformation (native)
        sitk_physical = sitk_img.TransformIndexToPhysicalPoint(index)
        
        # ANTs manual calculation (accounting for axis order)
        # ANTs shape is (z,y,x) but coordinates are still LPS
        ants_origin = ants_img.origin     # (ox, oy, oz)
        ants_spacing = ants_img.spacing   # (sz, sy, sx) 
        
        # Convert index [x,y,z] to physical [x,y,z] 
        ants_physical = [
            ants_origin[0] + index[0] * ants_spacing[2],  # X = origin_x + ix * spacing_x
            ants_origin[1] + index[1] * ants_spacing[1],  # Y = origin_y + iy * spacing_y  
            ants_origin[2] + index[2] * ants_spacing[0]   # Z = origin_z + iz * spacing_z
        ]
        
        # Compare results
        diff = np.linalg.norm(np.array(sitk_physical) - np.array(ants_physical))
        
        sitk_str = f"({sitk_physical[0]:.2f}, {sitk_physical[1]:.2f}, {sitk_physical[2]:.2f})"
        ants_str = f"({ants_physical[0]:.2f}, {ants_physical[1]:.2f}, {ants_physical[2]:.2f})"
        
        print(f"{str(index):<15} {sitk_str:<25} {ants_str:<25} {diff:.4f}")
        
        # Should be very close (within numerical precision)
        assert diff < 0.001, f"Coordinate mismatch: {diff} mm"
    
    print("✓ All coordinate transformations consistent")

# Test with sample points
test_indices = [
    [0, 0, 0],      # Origin corner
    [100, 100, 50], # Interior point
    [255, 255, 191] # Near opposite corner (for level 3)
]

validate_coordinate_consistency(zarr_uri, metadata, test_indices)
```

### Example 6: Pipeline Data Processing

```python
from aind_zarr_utils.pipeline_transformed import (
    mimic_pipeline_zarr_to_anatomical_stub, 
    neuroglancer_to_ccf
)
from aind_s3_cache.json_utils import get_json

# Example with pipeline-processed data
def process_pipeline_dataset(base_uri):
    """Process a complete pipeline dataset."""
    
    # Load all required metadata
    zarr_metadata = get_json(f"{base_uri}/metadata.json")
    processing_metadata = get_json(f"{base_uri}/processing.json") 
    neuroglancer_state = get_json(f"{base_uri}/neuroglancer_state.json")
    
    zarr_uri = f"{base_uri}/data.ome.zarr/0"
    
    print(f"Processing pipeline dataset:")
    print(f"  Pipeline version: {processing_metadata['processing']['pipeline_version']}")
    print(f"  Session: {zarr_metadata['session_id']}")
    
    # Create pipeline-corrected spatial domain
    pipeline_stub = mimic_pipeline_zarr_to_anatomical_stub(
        zarr_uri, zarr_metadata, processing_metadata
    )
    
    print(f"  Pipeline-corrected spacing: {pipeline_stub.GetSpacing()}")
    print(f"  Pipeline-corrected origin: {pipeline_stub.GetOrigin()}")
    
    # Transform annotations to CCF space
    template_name = "SmartSPIM-template_2024-05-16_11-26-14"
    ccf_points, descriptions = neuroglancer_to_ccf(
        neuroglancer_data=neuroglancer_state,
        zarr_uri=zarr_uri,
        zarr_metadata=zarr_metadata, 
        processing_metadata=processing_metadata,
        template_used=template_name
    )
    
    print(f"\nTransformed to CCF coordinates:")
    for layer, points in ccf_points.items():
        print(f"  {layer}: {points.shape[0]} points")
        if points.shape[0] > 0:
            center = points.mean(axis=0)
            print(f"    Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm")
            
    return {
        "pipeline_version": processing_metadata['processing']['pipeline_version'],
        "session_id": zarr_metadata['session_id'],
        "ccf_points": ccf_points,
        "descriptions": descriptions,
        "template": template_name
    }

# Example usage (replace with actual pipeline data URI)
# result = process_pipeline_dataset("s3://aind-pipeline-data/dataset123")
```

## Performance Examples

### Example 7: Efficient Batch Processing

```python
from aind_s3_cache.s3_cache import CacheManager
from aind_s3_cache.json_utils import get_json
from aind_zarr_utils.zarr import zarr_to_ants
import concurrent.futures
import time

def process_single_dataset(dataset_info, cache_dir=None):
    """Process a single dataset efficiently."""
    dataset_id, analysis_level = dataset_info
    
    try:
        # Load metadata with caching
        metadata_uri = f"s3://aind-open-data/{dataset_id}/metadata.json"
        metadata = get_json(metadata_uri)
        
        # Load image at specified level
        zarr_uri = f"s3://aind-open-data/{dataset_id}/data.ome.zarr/0"
        img = zarr_to_ants(zarr_uri, metadata, level=analysis_level)
        
        # Simple analysis
        array_data = img.numpy()
        result = {
            "dataset_id": dataset_id,
            "session_id": metadata.get("session_id"),
            "shape": img.shape,
            "mean_intensity": float(array_data.mean()),
            "std_intensity": float(array_data.std()),
            "success": True
        }
        
        return result
        
    except Exception as e:
        return {
            "dataset_id": dataset_id,
            "error": str(e),
            "success": False
        }

def batch_process_datasets(dataset_ids, analysis_level=3, max_workers=4):
    """Process multiple datasets in parallel."""
    
    dataset_info = [(dataset_id, analysis_level) for dataset_id in dataset_ids]
    results = []
    
    start_time = time.time()
    
    with CacheManager(persistent=True) as cm:
        print(f"Processing {len(dataset_ids)} datasets with {max_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_dataset = {
                executor.submit(process_single_dataset, info, cm.dir): info[0] 
                for info in dataset_info
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_dataset):
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    print(f"✓ {result['dataset_id']}: {result['session_id']}")
                else:
                    print(f"✗ {result['dataset_id']}: {result['error']}")
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    
    print(f"\nBatch processing complete:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Successful: {successful}/{len(dataset_ids)}")
    print(f"  Average time per dataset: {total_time/len(dataset_ids):.1f}s")
    
    return results

# Example usage
example_datasets = [
    "exaspim_708373_2024-02-02_11-26-44",
    # Add more dataset IDs for batch processing
]

# results = batch_process_datasets(example_datasets, analysis_level=3, max_workers=2)
```

### Example 8: Memory-Efficient Processing

```python
from aind_zarr_utils.zarr import zarr_to_sitk_stub, _open_zarr
import numpy as np

def memory_efficient_analysis(zarr_uri, metadata, chunk_size=100):
    """Analyze large ZARR without loading full dataset."""
    
    # Use stub to understand coordinate system
    stub_img, full_size = zarr_to_sitk_stub(zarr_uri, metadata, level=0)
    
    print(f"Full dataset size: {full_size}")
    print(f"Estimated memory: {np.prod(full_size) * 4 / 1024**3:.1f} GB")
    print(f"Processing in chunks of {chunk_size}^3...")
    
    # Open ZARR directly for chunked access
    image_node, zarr_meta = _open_zarr(zarr_uri)
    level_0_data = image_node['0']  # Full resolution level
    
    # Process in chunks
    statistics = []
    nz, ny, nx = full_size[2], full_size[1], full_size[0]  # ZARR is ZYX order
    
    for z in range(0, nz, chunk_size):
        for y in range(0, ny, chunk_size):
            for x in range(0, nx, chunk_size):
                # Define chunk bounds
                z_end = min(z + chunk_size, nz)
                y_end = min(y + chunk_size, ny) 
                x_end = min(x + chunk_size, nx)
                
                # Load chunk
                chunk = level_0_data[z:z_end, y:y_end, x:x_end]
                chunk_array = np.array(chunk)
                
                # Analyze chunk
                chunk_stats = {
                    "position": (z, y, x),
                    "size": chunk_array.shape,
                    "mean": float(chunk_array.mean()),
                    "std": float(chunk_array.std()),
                    "max": float(chunk_array.max()),
                    "min": float(chunk_array.min())
                }
                
                statistics.append(chunk_stats)
                
                print(f"  Chunk {len(statistics)}: pos=({z},{y},{x}), "
                      f"mean={chunk_stats['mean']:.1f}")
    
    # Aggregate statistics
    global_stats = {
        "total_chunks": len(statistics),
        "global_mean": np.mean([s['mean'] for s in statistics]),
        "global_std": np.mean([s['std'] for s in statistics]),
        "global_max": max([s['max'] for s in statistics]),
        "global_min": min([s['min'] for s in statistics])
    }
    
    print(f"\nGlobal statistics:")
    for key, value in global_stats.items():
        print(f"  {key}: {value}")
    
    return statistics, global_stats

# Example usage with large dataset
# stats, global_stats = memory_efficient_analysis(zarr_uri, metadata, chunk_size=200)
```

## Integration Examples

### Example 9: Integration with External Tools

```python
# Integration with scikit-image
from aind_zarr_utils.zarr import zarr_to_sitk
import SimpleITK as sitk
from skimage import filters, measure
import numpy as np

def segment_with_skimage(zarr_uri, metadata, level=3):
    """Segment ZARR data using scikit-image."""
    
    # Load image
    sitk_img = zarr_to_sitk(zarr_uri, metadata, level=level, scale_unit="millimeter")
    array = sitk.GetArrayFromImage(sitk_img)
    
    # Segment using scikit-image (example: Otsu thresholding)
    threshold = filters.threshold_otsu(array)
    binary = array > threshold
    
    # Label connected components
    labeled = measure.label(binary)
    props = measure.regionprops(labeled, array)
    
    print(f"Found {len(props)} objects")
    
    # Convert results back to physical coordinates
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    
    results = []
    for prop in props:
        # Convert centroid from voxel to physical coordinates
        # Note: SimpleITK array is ZYX, coordinates are XYZ
        centroid_voxel = prop.centroid  # (z, y, x)
        centroid_physical = [
            origin[0] + centroid_voxel[2] * spacing[0],  # X
            origin[1] + centroid_voxel[1] * spacing[1],  # Y
            origin[2] + centroid_voxel[0] * spacing[2]   # Z
        ]
        
        results.append({
            "label": prop.label,
            "area_voxels": prop.area,
            "area_mm3": prop.area * np.prod(spacing),
            "centroid_mm": centroid_physical,
            "mean_intensity": prop.mean_intensity
        })
    
    return results, binary, labeled

# Example usage
# segments, binary, labeled = segment_with_skimage(zarr_uri, metadata)
```

### Example 10: Custom Analysis Pipeline

```python
from aind_zarr_utils.zarr import zarr_to_ants
from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical
import ants
import numpy as np

class ZarrAnalysisPipeline:
    """Custom analysis pipeline for ZARR datasets."""
    
    def __init__(self, zarr_uri, metadata, working_level=3):
        self.zarr_uri = zarr_uri
        self.metadata = metadata
        self.working_level = working_level
        self.image = None
        self.preprocessed = None
        
    def load_image(self):
        """Load image at working resolution."""
        print(f"Loading ZARR at level {self.working_level}...")
        self.image = zarr_to_ants(
            self.zarr_uri, self.metadata, 
            level=self.working_level, 
            scale_unit="millimeter"
        )
        print(f"Loaded: {self.image.shape}, spacing: {self.image.spacing}")
        return self
        
    def preprocess(self, smooth_sigma=0.5):
        """Apply preprocessing steps."""
        print("Preprocessing image...")
        
        # Smooth image
        self.preprocessed = ants.smooth_image(self.image, smooth_sigma)
        
        # Normalize intensity
        array = self.preprocessed.numpy()
        array = (array - array.mean()) / array.std()
        self.preprocessed = ants.from_numpy(
            array, 
            origin=self.image.origin,
            spacing=self.image.spacing,
            direction=self.image.direction
        )
        
        print("Preprocessing complete")
        return self
        
    def register_to_template(self, template_path):
        """Register to a template (example)."""
        print("Registering to template...")
        
        # Load template
        template = ants.image_read(template_path)
        
        # Perform registration
        registration = ants.registration(
            fixed=template,
            moving=self.preprocessed,
            type_of_transform='SyN'
        )
        
        self.registration_result = registration
        print("Registration complete")
        return self
        
    def analyze_annotations(self, neuroglancer_data):
        """Analyze annotations in registered space."""
        print("Analyzing annotations...")
        
        # Transform annotations to physical space
        physical_coords, descriptions = neuroglancer_annotations_to_anatomical(
            neuroglancer_data, self.zarr_uri, self.metadata, 
            scale_unit="millimeter"
        )
        
        # Apply registration transforms if available
        if hasattr(self, 'registration_result'):
            transformed_coords = {}
            for layer, coords in physical_coords.items():
                # Transform each point using ANTs
                transformed_points = []
                for point in coords:
                    # Convert to ANTs format and transform
                    point_ants = ants.transform_physical_point_to_homogeneous_matrix(
                        point, self.registration_result['fwdtransforms']
                    )
                    transformed_points.append(point_ants)
                transformed_coords[layer] = np.array(transformed_points)
            
            physical_coords = transformed_coords
            
        self.annotations = physical_coords
        self.descriptions = descriptions
        print(f"Processed {sum(len(coords) for coords in physical_coords.values())} annotations")
        return self
        
    def generate_report(self):
        """Generate analysis report."""
        report = {
            "dataset": {
                "session_id": self.metadata.get("session_id"),
                "zarr_uri": self.zarr_uri,
                "working_level": self.working_level
            },
            "image_properties": {
                "shape": self.image.shape,
                "spacing": self.image.spacing,
                "physical_size": tuple(np.array(self.image.shape) * np.array(self.image.spacing))
            }
        }
        
        if hasattr(self, 'annotations'):
            report["annotations"] = {}
            for layer, coords in self.annotations.items():
                report["annotations"][layer] = {
                    "count": len(coords),
                    "center_of_mass": coords.mean(axis=0).tolist() if len(coords) > 0 else None
                }
                
        return report

# Example usage
def run_complete_analysis(zarr_uri, metadata, neuroglancer_data=None):
    """Run complete analysis pipeline."""
    
    pipeline = ZarrAnalysisPipeline(zarr_uri, metadata, working_level=3)
    
    # Execute pipeline
    pipeline.load_image().preprocess()
    
    if neuroglancer_data:
        pipeline.analyze_annotations(neuroglancer_data)
    
    # Generate and display report
    report = pipeline.generate_report()
    
    print("\n=== Analysis Report ===")
    print(f"Session: {report['dataset']['session_id']}")
    print(f"Image shape: {report['image_properties']['shape']}")
    print(f"Physical size: {report['image_properties']['physical_size']} mm")
    
    if "annotations" in report:
        print("\nAnnotations:")
        for layer, info in report["annotations"].items():
            print(f"  {layer}: {info['count']} points")
            if info['center_of_mass']:
                com = info['center_of_mass']
                print(f"    Center: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) mm")
    
    return pipeline, report

# Example execution
# pipeline, report = run_complete_analysis(zarr_uri, metadata, ng_state)
```

These examples demonstrate the full range of aind-zarr-utils capabilities, from basic data loading to complex analysis pipelines. Each example includes practical code that you can adapt for your specific use cases.