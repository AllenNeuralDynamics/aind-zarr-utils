"""I/O primitives for aind-zarr-utils.

This subpackage contains the low-level building blocks used by the higher-level
``Asset`` API: opening OME-Zarr stores, parsing AIND ND/processing metadata,
resolving asset URIs, and downloading ANTs transform chains. These modules are
intended to be small, focused, and free of cross-coupling with each other
beyond the dependency directions established here.
"""
