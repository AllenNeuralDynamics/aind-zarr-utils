origin module
=============

.. automodule:: aind_zarr_utils.origin
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

``Origin`` replaces the legacy ``set_origin`` / ``set_corner`` /
``set_corner_lps`` argument group with one explicit value.

Examples
--------

Default origin::

    img = asset.image(level=3, origin=Origin.default())

Set origin directly::

    img = asset.image(level=3, origin=Origin.at((0.0, 0.0, 0.0)))

Anchor a labelled corner::

    img = asset.image(
        level=3,
        origin=Origin.at_corner("RAS", (0.0, 0.0, 0.0)),
    )

``Origin`` is only accepted when ``pipeline=False``. Pipeline images and stubs
use the corrected origin from the pipeline metadata.
