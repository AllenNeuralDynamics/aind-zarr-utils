annotations module
==================

.. automodule:: aind_zarr_utils.annotations
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``annotations`` module provides utilities for transforming point annotations from image space to anatomical space. It works with SimpleITK images to handle coordinate transformations while maintaining proper LPS (Left-Posterior-Superior) coordinate conventions.


Coordinate Systems
------------------

This module works exclusively with LPS (Left-Posterior-Superior) coordinates, which is the standard for medical imaging and ITK/SimpleITK:

- **L**: Left direction is positive X
- **P**: Posterior direction is positive Y  
- **S**: Superior direction is positive Z

Examples
--------

Transform annotation points from image indices to anatomical coordinates::

    import SimpleITK as sitk
    import numpy as np
    from aind_zarr_utils.annotations import annotation_indices_to_anatomical

    # Create sample annotation points (as image indices)
    points = {
        "region1": np.array([[10, 20, 30], [40, 50, 60]]),
        "region2": np.array([[100, 200, 300]])
    }
    
    # Transform to anatomical space using SimpleITK image header
    anatomical_points = annotation_indices_to_anatomical(sitk_image, points)
    
    # Result is in LPS coordinates (millimeters)
    print(anatomical_points["region1"])  # Physical coordinates in LPS