points module
=============

.. automodule:: aind_zarr_utils.points
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Overview
--------

``Points`` stores named ``(N, 3)`` arrays and a ``Space`` tag. The constructor
validates shape and coerces values to floating point.

Supported spaces:

* ``Space.ZARR_INDICES``
* ``Space.LS_SCALED_MM``
* ``Space.LS_ANATOMICAL_MM``
* ``Space.LS_PIPELINE_ANATOMICAL_MM``
* ``Space.CCF_MM``

Examples
--------

Manual points::

    import numpy as np
    from aind_zarr_utils import Points, Space

    points = Points(
        {"soma": np.array([[100, 200, 50]])},
        Space.ZARR_INDICES,
    )

Neuroglancer points::

    points = Points.from_neuroglancer(ng_state)

SWC points::

    points = Points.from_swc(
        swc_array,
        axis_order="zyx",
        units="micrometer",
    )

Transform through an asset::

    ccf = asset.transform(points, to=Space.CCF_MM)
