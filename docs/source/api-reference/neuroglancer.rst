neuroglancer module
===================

.. automodule:: aind_zarr_utils.neuroglancer
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``neuroglancer`` module contains legacy readers and coordinate helpers for
Neuroglancer annotation state. New code should usually use
``Points.from_neuroglancer`` and then transform through an ``Asset``:

.. code-block:: python

    from aind_zarr_utils import Asset, Points, Space

    asset = Asset.from_neuroglancer(ng_state)
    points = Points.from_neuroglancer(ng_state)
    ccf = asset.transform(points, to=Space.CCF_MM)

Coordinate Notes
----------------

Neuroglancer annotation points are read as level-0 ``(z, y, x)`` Zarr indices.
Physical outputs from ``Asset.transform`` use the destination ``Space`` tag.

Legacy Examples
---------------

Extract annotation points from Neuroglancer JSON::

    from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_indices

    annotations, descriptions = neuroglancer_annotations_to_indices(
        ng_state,
        layer_names=["my_annotations"],
        return_description=True,
    )

Convert to raw anatomical coordinates with explicit metadata::

    from aind_zarr_utils.neuroglancer import neuroglancer_annotations_to_anatomical

    physical_points, descriptions = neuroglancer_annotations_to_anatomical(
        ng_state,
        zarr_uri,
        metadata,
        scale_unit="millimeter",
        layer_names=["region_annotations", "landmarks"],
    )
