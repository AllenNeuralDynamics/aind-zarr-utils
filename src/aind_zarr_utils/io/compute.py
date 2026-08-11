"""Caller-configurable options for the dask ``compute()`` that reads pixels.

Reading one pyramid level of an OME-Zarr is a single ``dask`` ``.compute()``
call on the array ``ome-zarr``'s reader hands back. Historically that call
took no arguments, which means it ran with dask's defaults and no caller —
not even one that knows the read is latency-bound on S3 — could influence
its concurrency.

Every function in this package that materialises pixel data now accepts a
``compute_kwargs`` mapping that is forwarded verbatim to ``.compute()``, e.g.
``{"scheduler": "threads", "num_workers": 16}``. Passing nothing preserves
the historical behaviour exactly.

Two environment variables provide a fallback for deployments that cannot
easily change their call sites (a capsule pinned to a released version, for
instance):

``AIND_ZARR_COMPUTE_WORKERS``
    Positive integer, used as ``num_workers``. Setting it also selects the
    ``"threads"`` scheduler unless a scheduler was requested explicitly.

``AIND_ZARR_COMPUTE_SCHEDULER``
    Dask scheduler name, used as ``scheduler`` (e.g. ``"threads"``,
    ``"synchronous"``, ``"processes"``).

Explicit ``compute_kwargs`` always win over the environment: the variables
only fill in keys the caller did not set. Malformed values are ignored with
a :class:`UserWarning` rather than raising, so a typo in a deployment's
environment cannot break a read that would otherwise have succeeded.

Notes
-----
For S3-backed stores the read is latency-bound, not bandwidth-bound, and the
blosc/zstd codec releases the GIL — so threads help both the fetch and the
decompression. Benchmarks against ``aind-open-data`` from outside AWS showed
roughly a 12x speed-up going from 1 to 16-32 workers, with 64 workers
regressing.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Environment variable naming the ``num_workers`` fallback.
COMPUTE_WORKERS_ENV_VAR = "AIND_ZARR_COMPUTE_WORKERS"

#: Environment variable naming the ``scheduler`` fallback.
COMPUTE_SCHEDULER_ENV_VAR = "AIND_ZARR_COMPUTE_SCHEDULER"

#: Type of the ``compute_kwargs`` argument threaded through the public API.
ComputeKwargs = Mapping[str, Any]


def _env_num_workers() -> int | None:
    """
    Read ``AIND_ZARR_COMPUTE_WORKERS`` as a positive integer.

    Returns
    -------
    int or None
        The requested worker count, or ``None`` when the variable is unset,
        empty, or malformed. Malformed values emit a :class:`UserWarning`.
    """
    raw = os.environ.get(COMPUTE_WORKERS_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        workers = int(raw)
    except ValueError:
        warnings.warn(
            f"Ignoring {COMPUTE_WORKERS_ENV_VAR}={raw!r}: not an integer.",
            stacklevel=3,
        )
        return None
    if workers < 1:
        warnings.warn(
            f"Ignoring {COMPUTE_WORKERS_ENV_VAR}={raw!r}: must be >= 1.",
            stacklevel=3,
        )
        return None
    return workers


def _env_scheduler() -> str | None:
    """
    Read ``AIND_ZARR_COMPUTE_SCHEDULER``.

    Returns
    -------
    str or None
        The requested scheduler name, or ``None`` when unset or empty. The
        name is not validated here; dask raises on an unknown scheduler.
    """
    raw = os.environ.get(COMPUTE_SCHEDULER_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def resolve_compute_kwargs(
    compute_kwargs: ComputeKwargs | None = None,
) -> dict[str, Any]:
    """
    Merge caller-supplied compute options with the environment fallbacks.

    Parameters
    ----------
    compute_kwargs : Mapping, optional
        Keyword arguments destined for ``dask``'s ``compute()``. Keys set
        here are never overridden by the environment.

    Returns
    -------
    dict
        The keyword arguments to pass to ``compute()``. Empty when neither
        the caller nor the environment asked for anything, which reproduces
        dask's default behaviour.

    Examples
    --------
    >>> resolve_compute_kwargs()  # no caller options, no env vars
    {}
    >>> resolve_compute_kwargs({"num_workers": 16})
    {'num_workers': 16, 'scheduler': 'threads'}
    """
    resolved = dict(compute_kwargs) if compute_kwargs else {}

    if "num_workers" not in resolved:
        env_workers = _env_num_workers()
        if env_workers is not None:
            resolved["num_workers"] = env_workers

    if "scheduler" not in resolved:
        env_scheduler = _env_scheduler()
        if env_scheduler is not None:
            resolved["scheduler"] = env_scheduler
        elif "num_workers" in resolved:
            # ``num_workers`` is only meaningful to a scheduler that has
            # workers; be explicit rather than relying on dask's per-collection
            # default.
            resolved["scheduler"] = "threads"

    return resolved


def compute_dask(
    array: Any,
    compute_kwargs: ComputeKwargs | None = None,
) -> NDArray:
    """
    Materialise a dask array using the resolved compute options.

    Parameters
    ----------
    array : dask.array.Array
        The lazy array to materialise (typically ``image_node.data[level]``).
    compute_kwargs : Mapping, optional
        Options forwarded to ``array.compute()`` after being merged with the
        environment fallbacks by :func:`resolve_compute_kwargs`.

    Returns
    -------
    numpy.ndarray
        The materialised array.
    """
    materialised: NDArray = array.compute(**resolve_compute_kwargs(compute_kwargs))
    return materialised
