"""Tests for caller-configurable dask compute options on Zarr pixel reads."""

from __future__ import annotations

from typing import Any

import pytest

from aind_zarr_utils import zarr as zarr_mod
from aind_zarr_utils.asset import Asset
from aind_zarr_utils.io.compute import (
    COMPUTE_SCHEDULER_ENV_VAR,
    COMPUTE_WORKERS_ENV_VAR,
    compute_dask,
    resolve_compute_kwargs,
)


@pytest.fixture(autouse=True)
def clear_compute_env(monkeypatch):
    """Keep the developer's own environment out of these tests."""
    monkeypatch.delenv(COMPUTE_WORKERS_ENV_VAR, raising=False)
    monkeypatch.delenv(COMPUTE_SCHEDULER_ENV_VAR, raising=False)


def _make_asset(node, metadata: dict[str, Any], **kwargs) -> Asset:
    """Build an Asset with a pre-opened mock Zarr and no other I/O."""
    processing: dict[str, Any] = {"processing_pipeline": {"pipeline_version": "3.1.0", "data_processes": []}}
    asset = Asset(
        alignment_zarr_uri="s3://bucket/data/asset/image.zarr/",
        metadata=metadata,
        processing=processing,
        **kwargs,
    )
    object.__setattr__(asset, "_opened_zarr", (node, node.metadata))
    return asset


# ------------------------------------------------------- resolve_compute_kwargs


class TestResolveComputeKwargs:
    """Merging of caller options with the environment fallbacks."""

    def test_default_is_empty(self):
        """No caller options and no env vars reproduces dask's own defaults."""
        assert resolve_compute_kwargs() == {}
        assert resolve_compute_kwargs(None) == {}
        assert resolve_compute_kwargs({}) == {}

    def test_explicit_kwargs_pass_through(self):
        """Arbitrary caller options are forwarded unchanged."""
        assert resolve_compute_kwargs({"scheduler": "synchronous"}) == {"scheduler": "synchronous"}
        assert resolve_compute_kwargs({"optimize_graph": False}) == {"optimize_graph": False}

    def test_explicit_workers_implies_threads(self):
        """``num_workers`` only means something to a scheduler with workers."""
        assert resolve_compute_kwargs({"num_workers": 16}) == {
            "num_workers": 16,
            "scheduler": "threads",
        }

    def test_explicit_workers_keeps_explicit_scheduler(self):
        """An explicit scheduler is never overwritten by the threads default."""
        assert resolve_compute_kwargs({"num_workers": 4, "scheduler": "processes"}) == {
            "num_workers": 4,
            "scheduler": "processes",
        }

    def test_caller_mapping_not_mutated(self):
        """The caller's mapping is copied, not modified in place."""
        caller = {"num_workers": 8}
        resolve_compute_kwargs(caller)
        assert caller == {"num_workers": 8}

    def test_env_workers(self, monkeypatch):
        """``AIND_ZARR_COMPUTE_WORKERS`` fills in num_workers + threads."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "24")
        assert resolve_compute_kwargs() == {"num_workers": 24, "scheduler": "threads"}

    def test_env_scheduler(self, monkeypatch):
        """``AIND_ZARR_COMPUTE_SCHEDULER`` fills in scheduler alone."""
        monkeypatch.setenv(COMPUTE_SCHEDULER_ENV_VAR, "synchronous")
        assert resolve_compute_kwargs() == {"scheduler": "synchronous"}

    def test_env_both(self, monkeypatch):
        """Both variables can be set together."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "12")
        monkeypatch.setenv(COMPUTE_SCHEDULER_ENV_VAR, "processes")
        assert resolve_compute_kwargs() == {"num_workers": 12, "scheduler": "processes"}

    def test_caller_wins_over_env(self, monkeypatch):
        """Explicit kwargs are never overridden by the environment."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "24")
        monkeypatch.setenv(COMPUTE_SCHEDULER_ENV_VAR, "processes")
        assert resolve_compute_kwargs({"num_workers": 2, "scheduler": "threads"}) == {
            "num_workers": 2,
            "scheduler": "threads",
        }

    def test_env_fills_only_missing_keys(self, monkeypatch):
        """The environment fills in keys the caller left unset."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "24")
        assert resolve_compute_kwargs({"scheduler": "processes"}) == {
            "num_workers": 24,
            "scheduler": "processes",
        }

    def test_empty_env_ignored(self, monkeypatch):
        """Empty / whitespace values behave as if unset, with no warning."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "  ")
        monkeypatch.setenv(COMPUTE_SCHEDULER_ENV_VAR, "")
        assert resolve_compute_kwargs() == {}

    def test_non_integer_env_warns_and_is_ignored(self, monkeypatch):
        """A typo in the deployment environment must not break the read."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "sixteen")
        with pytest.warns(UserWarning, match="not an integer"):
            assert resolve_compute_kwargs() == {}

    @pytest.mark.parametrize("value", ["0", "-4"])
    def test_non_positive_env_warns_and_is_ignored(self, monkeypatch, value):
        """Worker counts below 1 are meaningless; warn and fall back."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, value)
        with pytest.warns(UserWarning, match="must be >= 1"):
            assert resolve_compute_kwargs() == {}


class TestComputeDask:
    """``compute_dask`` forwards the resolved options to ``.compute()``."""

    def test_forwards_resolved_kwargs(self, mock_zarr_node):
        """Caller options reach the dask array untouched."""
        data = mock_zarr_node.data[0]
        compute_dask(data, {"scheduler": "synchronous"})
        assert data.compute_calls == [{"scheduler": "synchronous"}]

    def test_default_is_bare_compute(self, mock_zarr_node):
        """Passing nothing calls ``compute()`` with no arguments."""
        data = mock_zarr_node.data[0]
        compute_dask(data)
        assert data.compute_calls == [{}]


# ------------------------------------------------------------- plumbing tests


class TestZarrFunctionsThreadComputeKwargs:
    """The free functions in ``zarr`` forward ``compute_kwargs`` downward."""

    def test_zarr_to_numpy_anatomical(self, mock_zarr_node, mock_nd_metadata):
        """The private helper named in the TODO accepts compute options."""
        zarr_mod._zarr_to_numpy_anatomical(
            "s3://bucket/image.zarr",
            mock_nd_metadata,
            level=0,
            opened_zarr=(mock_zarr_node, mock_zarr_node.metadata),
            compute_kwargs={"scheduler": "threads", "num_workers": 8},
        )
        assert mock_zarr_node.data[0].compute_calls == [{"scheduler": "threads", "num_workers": 8}]

    def test_zarr_to_sitk(self, mock_zarr_node, mock_nd_metadata):
        """``zarr_to_sitk`` threads options through to the read."""
        zarr_mod.zarr_to_sitk(
            "s3://bucket/image.zarr",
            mock_nd_metadata,
            level=0,
            opened_zarr=(mock_zarr_node, mock_zarr_node.metadata),
            compute_kwargs={"num_workers": 16},
        )
        assert mock_zarr_node.data[0].compute_calls == [{"num_workers": 16, "scheduler": "threads"}]

    def test_zarr_to_ants(self, mock_zarr_node, mock_nd_metadata):
        """``zarr_to_ants`` threads options through to the read."""
        zarr_mod.zarr_to_ants(
            "s3://bucket/image.zarr",
            mock_nd_metadata,
            level=0,
            opened_zarr=(mock_zarr_node, mock_zarr_node.metadata),
            compute_kwargs={"num_workers": 16},
        )
        assert mock_zarr_node.data[0].compute_calls == [{"num_workers": 16, "scheduler": "threads"}]

    def test_no_kwargs_preserves_bare_compute(self, mock_zarr_node, mock_nd_metadata):
        """Callers that say nothing get exactly the historical behaviour."""
        zarr_mod.zarr_to_sitk(
            "s3://bucket/image.zarr",
            mock_nd_metadata,
            level=0,
            opened_zarr=(mock_zarr_node, mock_zarr_node.metadata),
        )
        assert mock_zarr_node.data[0].compute_calls == [{}]

    def test_env_var_reaches_the_read(self, monkeypatch, mock_zarr_node, mock_nd_metadata):
        """A deployment can tune reads without touching the call site."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "32")
        zarr_mod.zarr_to_sitk(
            "s3://bucket/image.zarr",
            mock_nd_metadata,
            level=0,
            opened_zarr=(mock_zarr_node, mock_zarr_node.metadata),
        )
        assert mock_zarr_node.data[0].compute_calls == [{"num_workers": 32, "scheduler": "threads"}]


class TestAssetComputeKwargs:
    """``Asset`` carries a default and honours a per-call override."""

    def test_asset_level_default(self, mock_zarr_node, mock_nd_metadata):
        """Options given to the constructor apply to every read."""
        asset = _make_asset(
            mock_zarr_node,
            mock_nd_metadata,
            compute_kwargs={"num_workers": 16, "scheduler": "threads"},
        )
        asset.image(level=0)
        assert mock_zarr_node.data[0].compute_calls == [{"num_workers": 16, "scheduler": "threads"}]

    def test_per_call_override(self, mock_zarr_node, mock_nd_metadata):
        """``image(compute_kwargs=...)`` replaces the asset-level default."""
        asset = _make_asset(mock_zarr_node, mock_nd_metadata, compute_kwargs={"num_workers": 16})
        asset.image(level=0, compute_kwargs={"scheduler": "synchronous"})
        assert mock_zarr_node.data[0].compute_calls == [{"scheduler": "synchronous"}]

    def test_ants_backend(self, mock_zarr_node, mock_nd_metadata):
        """The ANTs backend gets the same treatment as SimpleITK."""
        asset = _make_asset(mock_zarr_node, mock_nd_metadata, compute_kwargs={"num_workers": 4})
        asset.image(level=0, library="ants")
        assert mock_zarr_node.data[0].compute_calls == [{"num_workers": 4, "scheduler": "threads"}]

    def test_default_is_no_options(self, mock_zarr_node, mock_nd_metadata):
        """An asset built without compute options reads exactly as before."""
        asset = _make_asset(mock_zarr_node, mock_nd_metadata)
        assert asset.compute_kwargs is None
        asset.image(level=0)
        assert mock_zarr_node.data[0].compute_calls == [{}]

    def test_constructor_mapping_is_detached(self, mock_zarr_node, mock_nd_metadata):
        """Mutating the caller's dict afterwards does not affect the asset."""
        caller = {"num_workers": 16}
        asset = _make_asset(mock_zarr_node, mock_nd_metadata, compute_kwargs=caller)
        caller["num_workers"] = 1
        asset.image(level=0)
        assert mock_zarr_node.data[0].compute_calls == [{"num_workers": 16, "scheduler": "threads"}]


class TestRealDaskAcceptsOptions:
    """The generated options are valid for a real dask array, not just mocks."""

    def test_explicit_kwargs(self, real_ome_zarr, mock_nd_metadata):
        """Threaded reads of a real store return the same data."""
        arr, _, _, _ = zarr_mod._zarr_to_numpy_anatomical(
            real_ome_zarr,
            mock_nd_metadata,
            level=0,
            compute_kwargs={"scheduler": "threads", "num_workers": 2},
        )
        baseline, _, _, _ = zarr_mod._zarr_to_numpy_anatomical(real_ome_zarr, mock_nd_metadata, level=0)
        assert arr.shape == (10, 10, 10)
        assert (arr == baseline).all()

    def test_env_var(self, monkeypatch, real_ome_zarr):
        """The env-var path produces options dask actually accepts."""
        monkeypatch.setenv(COMPUTE_WORKERS_ENV_VAR, "2")
        arr, _, _ = zarr_mod.zarr_to_numpy(real_ome_zarr, level=0)
        assert arr.shape == (1, 1, 10, 10, 10)

    def test_synchronous_scheduler(self, real_ome_zarr):
        """``num_workers`` with the synchronous scheduler is tolerated."""
        arr, _, _ = zarr_mod.zarr_to_numpy(
            real_ome_zarr,
            level=0,
            compute_kwargs={"scheduler": "synchronous", "num_workers": 4},
        )
        assert arr.shape == (1, 1, 10, 10, 10)
