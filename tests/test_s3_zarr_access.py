"""Explicit S3 access survives format detection and child-location opening."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from zarr import storage

from aind_zarr_utils.asset import Asset
from aind_zarr_utils.io import zarr as io


@pytest.mark.parametrize("anonymous", [True, False])
def test_explicit_s3_store_reads_pixels_and_preserves_access_on_children(real_ome_zarr, monkeypatch, anonymous):
    multiscales = io._open_zarr(real_ome_zarr)[1]["multiscales"]
    metadata_access = []
    stores = []

    def local_path(uri):
        relative = uri.split(".zarr", 1)[1].lstrip("/")
        return Path(real_ome_zarr) / relative

    def filesystem(*, anon):
        metadata_access.append(anon)

        def read_metadata(uri):
            return local_path(uri).read_bytes()

        return Mock(cat_file=read_metadata)

    monkeypatch.setattr(io.s3fs, "S3FileSystem", filesystem)
    modern = getattr(storage, "FsspecStore", None)
    if modern is not None:

        def store(uri, *, storage_options, read_only):
            stores.append((uri, storage_options["anon"], read_only))
            return storage.LocalStore(str(local_path(uri)), read_only=True)

        monkeypatch.setattr(modern, "from_url", store)
    else:
        legacy_store = storage.FSStore

        class Store(legacy_store):
            def __init__(self, uri, *, mode, anon):
                stores.append((uri, anon, mode == "r"))
                super().__init__(str(local_path(uri)), mode="r")

        monkeypatch.setattr(storage, "FSStore", Store)

    uri = "s3://bucket/image.zarr"
    node, metadata = io._open_zarr(uri, anonymous=anonymous)
    assert np.array_equal(node.data[0].compute(), np.arange(1000).reshape(1, 1, 10, 10, 10))
    assert metadata["multiscales"] == multiscales
    child = node.zarr.create("../other.zarr")
    assert child.subpath() == "s3://bucket/other.zarr"
    assert metadata_access and all(value == anonymous for value in metadata_access)
    assert stores[0] == (uri, anonymous, True)
    assert stores[-1] == ("s3://bucket/other.zarr", anonymous, True)
    assert all(access == anonymous and read_only for _, access, read_only in stores)


@pytest.mark.parametrize("anonymous", [True, False])
def test_lazy_asset_forwards_access_choice_without_global_configuration(anonymous):
    asset = Asset("s3://bucket/image.zarr", {}, {}, anonymous=anonymous)
    with patch("aind_zarr_utils.asset._open_zarr", return_value=("node", {})) as opener:
        assert asset.opened_zarr[0] == "node"
        opener.assert_called_once_with(asset.alignment_zarr_uri, anonymous=anonymous)
