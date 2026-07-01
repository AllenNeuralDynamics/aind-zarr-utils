"""Concrete header overlays and cardinal-direction helpers.

Each overlay is a small, pure transformation of an
:class:`~aind_anatomical_utils.anatomical_volume.AnatomicalHeader`. They
satisfy the :class:`~aind_zarr_utils.domain.selector.Overlay` protocol
structurally — there is no inheritance relationship — so this module
intentionally has no upward dependency on
:mod:`aind_zarr_utils.domain.selector`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from aind_anatomical_utils.anatomical_volume import (
    AnatomicalHeader,
    fix_corner_compute_origin,
)
from packaging.specifiers import SpecifierSet
from packaging.version import Version

Vec3 = tuple[float, float, float]
_PIPELINE_MULTISCALE_FACTOR = 2


@dataclass(frozen=True, slots=True)
class SpacingScaleOverlay:
    """
    Multiply the index-order spacing by a scalar.

    Parameters
    ----------
    scale : float
        Scalar to multiply each of ``(si, sj, sk)`` by.
    name : str, default "spacing_scale"
        Overlay name (for logs).
    priority : int, default 50
        Execution priority. Should run after axis permutations/flips but before
        anchoring.
    """

    scale: float
    name: str = "spacing_scale"
    priority: int = 50

    def __call__(
        self,
        h: AnatomicalHeader,
        meta: dict[str, Any],
        multiscale_no: int,
        zarr_import_version: str | None = None,
    ) -> AnatomicalHeader:
        """
        Apply the scaling to the spacing.

        Parameters
        ----------
        h : AnatomicalHeader
            Input header.
        meta : dict
            Unused.
        multiscale_no : int
            Unused.
        zarr_import_version : str or None, optional
            Unused.

        Returns
        -------
        AnatomicalHeader
            New header with spacing multiplied by ``scale``.
        """
        i, j, k = h.spacing
        return replace(h, spacing=(i * self.scale, j * self.scale, k * self.scale))


@dataclass(frozen=True, slots=True)
class FlipIndexAxesOverlay:
    """Flip one or more **index axes** by negating the corresponding columns of the direction matrix.

    Parameters
    ----------
    flip_i, flip_j, flip_k : bool, optional
        If ``True``, flip that index axis.
    name : str, default "flip_index_axes"
        Overlay name.
    priority : int, default 40
        Execution priority. Typical order: permute (30) → flip (40) → spacing
        fixes (50–60) → anchor (90).
    """

    flip_i: bool = False
    flip_j: bool = False
    flip_k: bool = False
    name: str = "flip_index_axes"
    priority: int = 40

    def __call__(
        self,
        h: AnatomicalHeader,
        meta: dict[str, Any],
        multiscale_no: int,
        zarr_import_version: str | None = None,
    ) -> AnatomicalHeader:
        """
        Negate selected columns of the direction matrix.

        Returns
        -------
        AnatomicalHeader
            AnatomicalHeader with updated direction matrix.
        """
        D = h.direction.copy()
        if self.flip_i:
            D[:, 0] *= -1.0
        if self.flip_j:
            D[:, 1] *= -1.0
        if self.flip_k:
            D[:, 2] *= -1.0
        return replace(h, direction=D)


@dataclass(frozen=True, slots=True)
class PermuteIndexAxesOverlay:
    """Permute the **index axes** (i, j, k) and carry spacing/size/direction along.

    Parameters
    ----------
    order : tuple of int
        A permutation of ``(0, 1, 2)`` describing the new order of index axes.
    name : str, default "permute_index_axes"
        Overlay name.
    priority : int, default 30
        Execution priority. Should run before flips and spacing changes.
    """

    order: tuple[int, int, int]  # permutation of (0,1,2)
    name: str = "permute_index_axes"
    priority: int = 30

    def __call__(
        self,
        h: AnatomicalHeader,
        meta: dict[str, Any],
        multiscale_no: int,
        zarr_import_version: str | None = None,
    ) -> AnatomicalHeader:
        """Reorder direction columns, spacing elements, and size_ijk entries according to ``order``.

        Returns
        -------
        AnatomicalHeader
            AnatomicalHeader with permuted index axes.
        """
        i0, i1, i2 = self.order
        D = h.direction[:, [i0, i1, i2]]
        S: Vec3 = (h.spacing[i0], h.spacing[i1], h.spacing[i2])
        N: tuple[int, int, int] = (
            h.size_ijk[i0],
            h.size_ijk[i1],
            h.size_ijk[i2],
        )
        return AnatomicalHeader(origin=h.origin, spacing=S, direction=D, size_ijk=N)


@dataclass(frozen=True, slots=True)
class ForceCornerAnchorOverlay:
    """
    Set the origin so a particular anatomical corner lands at a target point.

    Uses :func:`aind_anatomical_utils.anatomical_volume.fix_corner_compute_origin`
    to compute the required origin from the current header, a corner code
    (e.g., ``"RAS"``), and a target point expressed in a labeled frame.

    Parameters
    ----------
    corner_code : str
        3-letter anatomical code identifying the corner to anchor
        (e.g., ``"LPI"``, ``"RAS"``).
    target_point_labeled : tuple of float
        Target coordinates (mm) of that corner in ``target_frame``.
    target_frame : str, default "LPS"
        Frame label of ``target_point_labeled`` (e.g., ``"RAS"`` or
        ``"LPS"``).
    use_outer_box : bool, default False
        If ``True``, anchor using bounding-box corners ``(-0.5, size-0.5)``;
        otherwise use voxel-center corners ``(0, size-1)``.
    name : str, default "force_corner_anchor"
        Overlay name.
    priority : int, default 90
        Execution priority. Should run after spacing/axis overlays.
    """

    corner_code: str
    target_point_labeled: tuple[float, float, float]
    target_frame: str = "LPS"
    use_outer_box: bool = False
    name: str = "force_corner_anchor"
    priority: int = 90

    def __call__(
        self,
        h: AnatomicalHeader,
        meta: dict[str, Any],
        multiscale_no: int,
        zarr_import_version: str | None = None,
    ) -> AnatomicalHeader:
        """Compute and set the origin such that the specified corner aligns with the target point.

        Returns
        -------
        AnatomicalHeader
            AnatomicalHeader with updated origin.
        """
        origin_lps, _, _ = fix_corner_compute_origin(
            size=h.size_ijk,
            spacing=h.spacing,
            direction=h.direction,
            target_point=self.target_point_labeled,
            corner_code=self.corner_code,
            target_frame=self.target_frame,
            use_outer_box=self.use_outer_box,
        )
        return replace(h, origin=origin_lps)


@dataclass(frozen=True, slots=True)
class ForceCornerAnchorOverlayWithVersionWarning(ForceCornerAnchorOverlay):
    """
    Extends ForceCornerAnchorOverlay with version-aware warnings.

    Emits a warning if the zarr_import_version parameter exceeds the last
    verified buggy version, indicating the correction is being applied
    optimistically to a version that may have fixed the bug.

    Parameters
    ----------
    last_verified_buggy_version : str, default "0.0.34"
        The last zarr import version known to have the alignment bug.
        If the runtime version exceeds this, a warning is emitted.

    See Also
    --------
    ForceCornerAnchorOverlay : Base class for corner anchor correction.
    """

    last_verified_buggy_version: str = "0.0.34"

    def __call__(
        self,
        h: AnatomicalHeader,
        meta: dict[str, Any],
        multiscale_no: int,
        zarr_import_version: str | None = None,
    ) -> AnatomicalHeader:
        """Compute and set the origin, emitting a warning if the version is beyond the last verified buggy version.

        Parameters
        ----------
        h : AnatomicalHeader
            Input header.
        meta : dict
            Acquisition metadata (unused by this overlay).
        multiscale_no : int
            Multiscale level (unused by this overlay).
        zarr_import_version : str or None, optional
            Zarr import process version. If provided and exceeds
            ``last_verified_buggy_version``, a warning is emitted.

        Returns
        -------
        AnatomicalHeader
            AnatomicalHeader with updated origin.
        """
        import warnings

        if zarr_import_version:
            current = Version(zarr_import_version)
            verified = Version(self.last_verified_buggy_version)

            if current > verified:
                warnings.warn(
                    f"Applying RAS corner anchor correction for "
                    f"zarr_import version {zarr_import_version}, which "
                    f"exceeds last verified buggy version "
                    f"{self.last_verified_buggy_version}. Assuming the "
                    f"atlas alignment bug persists in this version. If "
                    f"the bug has been fixed, this correction may "
                    f"introduce errors.",
                    UserWarning,
                    stacklevel=5,
                )

        # Call parent implementation
        return ForceCornerAnchorOverlay.__call__(self, h, meta, multiscale_no, zarr_import_version)


def _require_cardinal(D: np.ndarray, *, atol: float = 1e-6) -> None:
    """
    Validate that a direction matrix is **cardinal** (signed permutation).

    Parameters
    ----------
    D : numpy.ndarray
        3×3 direction matrix to check.
    atol : float, default 1e-6
        Absolute tolerance for one-hot/orthonormal checks.

    Raises
    ------
    ValueError
        If the matrix mixes world axes (oblique) or fails the permutation
        tests.

    Notes
    -----
    - Each column must be a one-hot (up to sign) along exactly one world axis.
    - Each world axis must be selected by exactly one column.
    """
    D = np.asarray(D, float).reshape(3, 3)
    M = np.abs(D)
    # each column picks exactly one world axis; each world axis claimed by
    # exactly one column
    ok = (
        np.allclose(M.max(axis=0), 1.0, atol=atol)
        and np.allclose(M.sum(axis=0), 1.0, atol=atol)
        and np.allclose(M.sum(axis=1), 1.0, atol=atol)
    )
    if not ok:
        raise ValueError("Direction is not cardinal (signed permutation). Oblique not supported.")


def lps_world_to_index_spacing_cardinal(D: np.ndarray, lps_spacing_mm: Vec3) -> Vec3:
    """
    Convert **LPS world** spacings to **index-order** spacings (cardinal only).

    Parameters
    ----------
    D : numpy.ndarray
        3×3 cardinal direction matrix (columns are index-axis unit vectors in
        LPS). Must pass :func:`_require_cardinal`.
    lps_spacing_mm : tuple of float
        Desired world spacings (``sx, sy, sz``) in LPS (mm).

    Returns
    -------
    tuple of float
        Index-order spacings ``(si, sj, sk)`` in millimeters.

    Raises
    ------
    ValueError
        If ``D`` is not cardinal.
    """
    _require_cardinal(D)
    D = np.asarray(D, float).reshape(3, 3)
    sx, sy, sz = map(float, lps_spacing_mm)
    s_world: Vec3 = (sx, sy, sz)
    # For each index axis (column), find which world axis it aligns to (ignore
    # sign).
    # len=3, values in {0,1,2} for x,y,z
    world_for_idx = np.argmax(np.abs(D), axis=0)
    i, j, k = map(int, world_for_idx.tolist())
    return (s_world[i], s_world[j], s_world[k])  # (si, sj, sk)


@dataclass(frozen=True, slots=True)
class SetLpsWorldSpacingOverlay:
    """
    Set index-order spacing from a single **LPS world** spacing triple.

    This overlay **fails fast** if the direction matrix is oblique
    (non-cardinal).

    Parameters
    ----------
    lps_spacing_mm : tuple of float
        Desired spacings along L, P, and S (i.e., world X, Y, Z) in
        millimeters.
    name : str, default "set_world_spacing_lps"
        Overlay name.
    priority : int, default 55
        Execution priority. Typically run after axis permutations/flips (≤50)
        and before corner anchoring (≈90).
    """

    lps_spacing_mm: Vec3
    name: str = "set_world_spacing_lps"
    priority: int = 55  # after permute/flip, before anchoring

    def __call__(
        self,
        h: AnatomicalHeader,
        meta: dict[str, Any],
        multiscale_no: int,
        zarr_import_version: str | None = None,
    ) -> AnatomicalHeader:
        """
        Convert the desired LPS spacings to index-order spacing and set it.

        Parameters
        ----------
        h : AnatomicalHeader
            Input header whose direction determines the mapping.
        meta : dict
            Unused.
        multiscale_no : int
            Multiscale level index. Spacing is downscaled by
            ``(1 / _PIPELINE_MULTISCALE_FACTOR) ** multiscale_no``.
        zarr_import_version : str or None, optional
            Unused.

        Returns
        -------
        AnatomicalHeader
            AnatomicalHeader with updated spacing.
        """
        si, sj, sk = lps_world_to_index_spacing_cardinal(h.direction, self.lps_spacing_mm)
        scaling = (1 / _PIPELINE_MULTISCALE_FACTOR) ** multiscale_no
        base_spacing: Vec3 = (scaling * si, scaling * sj, scaling * sk)
        return replace(h, spacing=base_spacing)


def estimate_pipeline_multiscale(zarr_metadata: dict[str, Any], pipeline_ccf_reg_version: Version) -> int | None:
    """
    Heuristically estimate the multiscale pyramid level used by the pipeline.

    Parameters
    ----------
    zarr_metadata : dict
        Acquisition/Zarr metadata (reserved for future use).
    pipeline_ccf_reg_version : packaging.version.Version
        Registration pipeline version.

    Returns
    -------
    int or None
        Estimated multiscale level (e.g., ``3``) if known for the version,
        otherwise ``None``.

    Notes
    -----
    This is a placeholder; extend with real logic/rules as needed.
    """
    if pipeline_ccf_reg_version in SpecifierSet(">=0.0.18,<0.0.34"):
        return 3
    return None
