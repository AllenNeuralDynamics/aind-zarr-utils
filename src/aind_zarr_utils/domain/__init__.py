"""Pipeline-domain corrections for SmartSPIM ↔ CCF mappings.

Some historical SmartSPIM pipeline versions produce spatial domains that
disagree with what the LS → template ANTs transforms were trained against.
This subpackage contains the *overlay system* used to mimic those buggy
domains so that voxel indices from a Zarr can be projected to the same
LPS world that the transforms expect.

The split here is dependency-driven:

* :mod:`aind_zarr_utils.domain.overlays` — concrete overlay classes
  (axis permutation/flip, spacing fixes, corner anchoring) plus the
  cardinal-direction helpers and the pipeline-multiscale heuristic.
* :mod:`aind_zarr_utils.domain.selector` — the :class:`Overlay` protocol,
  :class:`OverlayRule`, :class:`OverlaySelector`, ``apply_overlays``,
  and the default-rules / shared-singleton machinery.
"""
