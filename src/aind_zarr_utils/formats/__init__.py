"""Format-specific readers/converters for input data sources.

The modules here understand the *file formats* the package consumes —
Neuroglancer JSON state, SWC neuron tracings — and turn them into the
neutral ``(N, 3)`` arrays / per-layer dicts that the rest of the package
operates on. They deliberately do not know about ZARR opening, S3, or
the asset-level orchestration: those concerns live in
:mod:`aind_zarr_utils.io` and (eventually) the ``Asset`` façade.
"""
