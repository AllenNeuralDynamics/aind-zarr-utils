"""Low-level OME-Zarr opening and per-level metadata extraction.

These helpers wrap ``ome-zarr-py``'s ``Reader``/``Node`` interface and turn its
multiscale metadata into the spacing/size/axis tuples the rest of the package
operates on. They do not depend on AIND-specific metadata; for that, see
:mod:`aind_zarr_utils.io.metadata`.
"""

from __future__ import annotations

import json
import posixpath

import numpy as np
import s3fs  # type: ignore[import-untyped]
from numpy.typing import NDArray
from ome_zarr.format import CurrentFormat, Format, detect_format  # type: ignore[import-untyped]
from ome_zarr.io import ZarrLocation, parse_url  # type: ignore[import-untyped]
from ome_zarr.reader import Node, Reader  # type: ignore[import-untyped]
from zarr import storage  # type: ignore[import-untyped,unused-ignore]

from aind_zarr_utils.io.compute import ComputeKwargs, compute_dask
from aind_zarr_utils.io.metadata import _unit_conversion
from aind_zarr_utils.io.ngff import level_transform


def ensure_native_endian(a: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    """Return ``a`` with native-endian dtype.

    - View if already native or byteorder-agnostic.
    - Copy only when endianness must change or fields are mixed.
    - With ``inplace=True``, mutate array when it's safe.
    """
    a = np.asarray(a)
    dt = a.dtype

    # An explicit byteorder can still be native: zarr 3 reports '<' on little-endian hosts.
    if dt.isnative:
        return a

    # -------- Plain (non-structured) dtype --------
    if dt.fields is None:
        if inplace:
            if not a.flags.writeable:
                raise ValueError("Array is not writeable; cannot convert in-place.")
            a.byteswap(inplace=True)
            a.dtype = dt.newbyteorder("=")  # type: ignore[misc]
            return a
        return a.astype(dt.newbyteorder("="), copy=False)

    # -------- Structured dtype --------
    # A whole-array byteswap would corrupt any native field. Nested and subarray fields report '|' whatever
    # their contents, so only flat scalar fields are trusted to be safe to swap.
    swappable = all(
        subdt.fields is None and subdt.subdtype is None and (not subdt.isnative or subdt.byteorder == "|")
        for (subdt, *_) in dt.fields.values()
    )
    if not swappable:
        return a.astype(dt.newbyteorder("="), copy=True)

    if inplace:
        if not a.flags.writeable:
            raise ValueError("Array is not writeable; cannot convert in-place.")
        a.byteswap(inplace=True)
        a.dtype = dt.newbyteorder("=")  # type: ignore[misc]
        return a

    return a.astype(dt.newbyteorder("="), copy=False)


class _S3ZarrLocation(ZarrLocation):
    """Preserve per-open authentication across OME-Zarr format detection and children."""

    def __init__(self, uri: str, anonymous: bool, fmt: Format | None = None) -> None:
        self._uri = uri.rstrip("/")
        self._anonymous = anonymous
        fs = s3fs.S3FileSystem(anon=anonymous)
        attributes = {}
        for name in (".zattrs", "zarr.json"):
            try:
                metadata = json.loads(fs.cat_file(self._uri + "/" + name))
            except FileNotFoundError:
                continue
            attributes = metadata if name == ".zattrs" else metadata.get("attributes", {})
            break
        # Older OME-Zarr rebuilds the store when detection changes the format,
        # discarding credentials. Detect first, using the same configured access.
        fmt = detect_format(attributes.get("ome", attributes), fmt or CurrentFormat())
        store_type = getattr(storage, "FsspecStore", None)
        if store_type is not None:
            store = store_type.from_url(self._uri, storage_options={"anon": anonymous}, read_only=True)
        else:
            store = getattr(storage, "FSStore")(self._uri, mode="r", anon=anonymous)
        super().__init__(store, mode="r", fmt=fmt)

    def subpath(self, subpath: str = "") -> str:
        """Keep child paths rooted at the original S3 URI rather than a store prefix."""
        if not subpath:
            return self._uri
        bucket, _, key = self._uri.removeprefix("s3://").partition("/")
        child = posixpath.normpath(posixpath.join(key, subpath.lstrip("/")))
        return "s3://" + bucket + ("/" + child if child != "." else "")

    def create(self, path: str) -> _S3ZarrLocation:
        """Open children with this location's access settings and detected format."""
        return _S3ZarrLocation(self.subpath(path), self._anonymous, self.fmt)


def _open_zarr(uri: str, *, anonymous: bool | None = None) -> tuple[Node, dict]:
    """
    Open a ZARR file and retrieve its metadata.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    anonymous : bool or None, optional
        Explicit S3 access mode. None preserves the storage backend's default.

    Returns
    -------
    image_node : ome_zarr.reader.Node
        The image node of the ZARR file.
    zarr_meta : dict
        Metadata of the ZARR file.
    """
    location = _S3ZarrLocation(uri, anonymous) if uri.startswith("s3://") and anonymous is not None else parse_url(uri)
    reader = Reader(location)

    # nodes may include images, labels etc
    nodes = list(reader())

    # first node will be the image pixel data
    image_node = nodes[0]
    zarr_meta = dict(image_node.metadata)
    attributes = getattr(getattr(image_node, "zarr", None), "root_attrs", None)
    if isinstance(attributes, dict) and "multiscales" in attributes:
        zarr_meta["multiscales"] = attributes["multiscales"]
    return image_node, zarr_meta


def zarr_to_numpy(
    uri: str,
    level: int = 3,
    ensure_native_endianness: bool = False,
    *,
    compute_kwargs: ComputeKwargs | None = None,
) -> tuple[NDArray, dict, int]:
    """
    Convert a ZARR file to a NumPy array.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    level : int, optional
        Resolution level to read, by default 3.
    ensure_native_endianness : bool, optional
        Whether to ensure native endianness of the returned array, by default
        False.
    compute_kwargs : Mapping, optional
        Options forwarded to dask's ``compute()`` when materialising the
        pixel data, e.g. ``{"scheduler": "threads", "num_workers": 16}``.
        Defaults to the environment-derived options resolved by
        :func:`~aind_zarr_utils.io.compute.resolve_compute_kwargs`, which is
        dask's own default unless ``AIND_ZARR_COMPUTE_WORKERS`` /
        ``AIND_ZARR_COMPUTE_SCHEDULER`` are set.

    Returns
    -------
    arr_data : ndarray
        NumPy array of the image data.
    zarr_meta : dict
        Metadata of the ZARR file.
    level : int
        Resolution level used.
    """
    image_node, zarr_meta = _open_zarr(uri)
    arr_data = compute_dask(image_node.data[level], compute_kwargs)
    if ensure_native_endianness:
        arr_data = ensure_native_endian(arr_data, inplace=True)
    return arr_data, zarr_meta, level


def _zarr_to_scaled(
    uri: str,
    *,
    level: int = 3,
    scale_unit: str = "millimeter",
    opened_zarr: tuple[Node, dict] | None = None,
) -> tuple[Node, set[int], list[str], list[float], list[int]]:
    """
    Extract scaled coordinate information from a ZARR file.

    Parameters
    ----------
    uri : str
        URI of the ZARR file.
    level : int, optional
        Resolution level to read, by default 3.
    scale_unit : str, optional
        Unit for scaling, by default "millimeter".
    opened_zarr : tuple, optional
        Pre-opened ZARR file (image_node, zarr_meta), by default None. If
        provided, this will be used instead of opening the ZARR file again.

    Returns
    -------
    image_node : ome_zarr.reader.Node
        The image node of the ZARR file.
    rej_axes : set
        Rejected axes indices.
    spacing : list
        List of spacing values.
    size : list
        List of size values.
    original_to_subset_axes_map : dict
        Mapping from original axes to subset axes.
    """
    # Create the zarr reader
    if opened_zarr is None:
        image_node, zarr_meta = _open_zarr(uri)
    else:
        image_node, zarr_meta = opened_zarr
    scale, _ = level_transform(zarr_meta, level)
    original_zarr_axes = zarr_meta["axes"]
    spatial_dims = {"x", "y", "z"}
    original_to_subset_axes_map = {}  # sorted
    i = 0
    for j, ax in enumerate(original_zarr_axes):
        ax_name = ax["name"]
        if ax_name in spatial_dims:
            original_to_subset_axes_map[j] = i
            i += 1
    rej_axes = set(range(len(original_zarr_axes))) - set(original_to_subset_axes_map.keys())
    spacing = []
    size = []
    kept_zarr_axes = []
    dask_shape = image_node.data[level].shape
    for i in original_to_subset_axes_map:
        kept_zarr_axes.append(original_zarr_axes[i]["name"])
        scale_factor = _unit_conversion(original_zarr_axes[i]["unit"], scale_unit)
        spacing.append(scale_factor * scale[i])
        size.append(dask_shape[i])
    return image_node, rej_axes, kept_zarr_axes, spacing, size
