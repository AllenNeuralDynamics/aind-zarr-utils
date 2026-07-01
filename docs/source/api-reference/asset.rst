asset module
============

.. automodule:: aind_zarr_utils.asset
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Overview
--------

``Asset`` is the recommended user-facing entry point. It owns the
alignment-channel Zarr URI, parsed metadata, parsed processing metadata, the
opened Zarr cache, and transform-chain cache.

Common constructors::

    from aind_zarr_utils import Asset

    asset = Asset.from_zarr("s3://bucket/dataset/image.ome.zarr/0")
    asset = Asset.from_root("s3://bucket/dataset")
    asset = Asset.from_neuroglancer(ng_state)

No-I/O construction is available when metadata is already loaded::

    asset = Asset(
        alignment_zarr_uri="s3://bucket/dataset/alignment.ome.zarr/0",
        metadata=metadata,
        processing=processing,
    )

Examples
--------

Build images and stubs::

    from aind_zarr_utils import Origin

    sitk_img = asset.image(level=3)
    ants_img = asset.image(level=3, library="ants")
    stub, size_ijk = asset.stub(level=0)
    pipeline_stub, native_size_ijk = asset.stub(pipeline=True)

    anchored = asset.image(
        level=3,
        origin=Origin.at_corner("RAS", (0.0, 0.0, 0.0)),
    )

Transform points::

    from aind_zarr_utils import Points, Space

    points = Points.from_neuroglancer(ng_state)
    ccf = asset.transform(points, to=Space.CCF_MM)
