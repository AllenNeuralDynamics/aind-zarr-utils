compute module
==============

.. automodule:: aind_zarr_utils.io.compute
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``io.compute`` module holds the options that control the single dask
``compute()`` call which materialises Zarr pixel data. Reading an S3-backed
Zarr is latency-bound rather than bandwidth-bound, so the concurrency of that
call dominates the wall-clock cost of ``Asset.image()`` and the legacy
``zarr_to_ants`` / ``zarr_to_sitk`` functions.

Every function that reads pixels accepts a ``compute_kwargs`` mapping which is
forwarded verbatim to ``compute()``. Passing nothing preserves dask's default
behaviour.

Environment Fallbacks
---------------------

For deployments that cannot change their call sites (a capsule pinned to a
released version, for instance), two environment variables fill in options the
caller did not set:

``AIND_ZARR_COMPUTE_WORKERS``
  Positive integer used as ``num_workers``; also selects the ``"threads"``
  scheduler unless a scheduler was requested explicitly.

``AIND_ZARR_COMPUTE_SCHEDULER``
  Dask scheduler name used as ``scheduler`` (``"threads"``,
  ``"synchronous"``, ``"processes"``).

Explicit ``compute_kwargs`` always take precedence. Malformed values are
ignored with a ``UserWarning`` rather than raising.
