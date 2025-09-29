"""
Module to determine and reproduce the *physical domain* that a registration
pipeline used (including historical/buggy variants), based on its version and
acquisition metadata. The goal is to produce a SimpleITK header (origin,
spacing, direction) that recreates exactly the coordinate system **the
transforms were fit in**, so that voxel indices from a Zarr can be mapped to
the correct LPS world coordinates before applying ANTs/ITK transforms.

This module provides:

- A lightweight immutable :class:`Header` describing an ITK/SimpleITK header.
- A composable **overlay system** where small, pure operations (overlays) tweak
  the base header in deterministic order.
- A :class:`OverlaySelector` which chooses a list of overlays to apply based on
  pipeline version, optional acquisition date, and arbitrary acquisition
  metadata predicates.
- A few concrete overlays (axis permutation/flip, spacing fixes, corner
  anchoring) and helpers for cardinal (axis-aligned) headers.

Notes
-----
- All coordinates are expressed in **ITK LPS** convention and **millimeters**.
- SimpleITK direction matrices are 3×3, flattened row-major by
  Get/SetDirection.
  **Columns** are the LPS-world unit vectors of the image **index axes**
  (``i``, ``j``, ``k``). The mapping used here is::

      physical = origin + direction @ (spacing ⊙ index)

  where ``index = [i, j, k]^T`` and ``spacing`` is in **index order**.
- Oblique (non-cardinal) direction matrices are **not supported** by the
  spacing helpers in this module. They will raise with a clear error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from typing_extensions import Self

from aind_zarr_utils.zarr import compute_origin_for_corner

if TYPE_CHECKING:
    from ants.core import ANTsImage  # type: ignore[import-untyped]

Vec3 = tuple[float, float, float]
H = TypeVar("H", bound="Header")
_PIPELINE_MULTISCALE_FACTOR = 2


@dataclass(frozen=True, slots=True)
class Header:
    """
    Lightweight, immutable wrapper of an ITK/SimpleITK header.

    The physical mapping is:

    ``physical = origin + direction @ (spacing ⊙ index)``

    where **columns** of ``direction`` are the LPS unit vectors of the index
    axes ``(i, j, k)``; ``spacing`` is per-index-axis (mm); and ``origin`` is
    LPS (mm).

    Parameters
    ----------
    origin : tuple of float
        Image origin in LPS (mm). Length-3.
    spacing : tuple of float
        Spacing per **index axis** ``(i, j, k)`` in millimeters. Length-3.
    direction : numpy.ndarray
        3×3 direction cosine matrix. Columns are unit vectors for ``i, j, k``
        expressed in LPS. Row-major when flattened for SimpleITK APIs.
    size_ijk : tuple of int
        Image size (number of voxels) along ``(i, j, k)``. Used by overlays
        like corner anchoring.

    Notes
    -----
    - Use :meth:`as_sitk` to obtain a 1×1×1 SimpleITK image carrying this
      header.
    - Use :meth:`update_sitk` to set the header onto an existing image.
    """

    origin: Vec3  # LPS mm
    spacing: Vec3  # per INDEX axis (i,j,k) in mm
    direction: NDArray  # (3,3) columns are unit vectors for i,j,k (in LPS)
    size_ijk: tuple[int, int, int]  # needed for corner anchoring etc.

    def direction_tuple(self) -> tuple[float, ...]:
        """
        Return the direction matrix flattened row-major as a tuple of floats.

        Returns
        -------
        tuple of float
            Length-9 row-major flattening of the 3×3 direction matrix, suitable
            for :meth:`SimpleITK.Image.SetDirection`.
        """
        return tuple(float(x) for x in self.direction.ravel())

    def update_sitk(self, sitk_image: sitk.Image) -> sitk.Image:
        """
        Set this header (origin, spacing, direction) on a SimpleITK image.

        Parameters
        ----------
        sitk_image : SimpleITK.Image
            The image whose header should be updated.

        Returns
        -------
        SimpleITK.Image
            The same image instance, updated in-place for convenience.
        """
        sitk_image.SetOrigin(tuple(self.origin))
        sitk_image.SetSpacing(tuple(self.spacing))
        sitk_image.SetDirection(self.direction_tuple())
        return sitk_image

    def update_ants(self, ants_image: ANTsImage) -> ANTsImage:
        """
        Set this header (origin, spacing, direction) on an ANTs image.

        Parameters
        ----------
        ants_image : ants.core.ANTsImage
            The image whose header should be updated.

        Returns
        -------
        ants.core.ANTsImage
            The same image instance, updated in-place for convenience.
        """
        # ANTs is backwards for reasons
        origin_rev = tuple(reversed(self.origin))
        spacing_rev = tuple(reversed(self.spacing))
        # ANTs uses a 2D numpy array
        dir_mat = np.array(self.direction).reshape((3, 3))
        ants_image.set_origin(origin_rev)
        ants_image.set_spacing(spacing_rev)
        ants_image.set_direction(dir_mat)
        return ants_image

    def as_sitk(self) -> sitk.Image:
        """
        Create a minimal SimpleITK image (1×1×1) carrying this header.

        Returns
        -------
        SimpleITK.Image
            A new image with this :class:`Header`'s origin, spacing, and
            direction set. Pixel type is ``UInt8`` and size is 1×1×1, which is
            sufficient for coordinate transforms via
            :meth:`TransformContinuousIndexToPhysicalPoint`.
        """
        img = sitk.Image([1, 1, 1], sitk.sitkUInt8)
        self.update_sitk(img)
        return img

    @classmethod
    def from_sitk(
        cls,
        sitk_image: sitk.Image,
        size_ijk: tuple[int, int, int] | None = None,
    ) -> Self:
        """
        Construct a :class:`Header` from a SimpleITK image.

        Parameters
        ----------
        sitk_image : SimpleITK.Image
            Source image.
        size_ijk : tuple of int or None
            Size to record as ``(i, j, k)``. If ``None``, uses
            :meth:`SimpleITK.Image.GetSize`.

        Returns
        -------
        Header
            New header with origin, spacing, direction, and ``size_ijk`` taken
            from ``sitk_image`` (or the provided size).
        """
        origin = sitk_image.GetOrigin()
        spacing = sitk_image.GetSpacing()
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
        if size_ijk is None:
            size_ijk = sitk_image.GetSize()

        return cls(
            origin=origin,
            spacing=spacing,
            direction=direction,
            size_ijk=size_ijk,
        )

    @classmethod
    def from_ants(
        cls,
        ants_image: ANTsImage,
        size_ijk: tuple[int, int, int] | None = None,
    ) -> Self:
        """
        Construct a :class:`Header` from an ANTs image.

        Parameters
        ----------
        ants_image : ants.core.ANTsImage
            Source image.
        size_ijk : tuple of int or None
            Size to record as ``(i, j, k)``. If ``None``, uses
            :meth:`ants.core.ANTsImage.shape`.

        Returns
        -------
        Header
            New header with origin, spacing, direction, and ``size_ijk`` taken
            from ``ants_image`` (or the provided size).
        """
        origin = tuple(reversed(ants_image.get_origin()))
        spacing = tuple(reversed(ants_image.get_spacing()))
        direction = ants_image.get_direction()
        if size_ijk is None:
            size_ijk = tuple(reversed(ants_image.shape))

        return cls(
            origin=origin,
            spacing=spacing,
            direction=direction,
            size_ijk=size_ijk,
        )


# Callable overlay interface (dataclass implementations satisfy this)
class Overlay(Protocol):
    """
    Protocol for callable overlays that transform a :class:`Header`.

    Required interface
    ------------------
    name : str
        Human-friendly identifier used in logs/audits.
    priority : int
        Execution priority (lower numbers run earlier). Independent of
        :attr:`OverlayRule.rule_priority`.
    __call__(h, meta, multiscale_no) -> Header
        Apply the overlay, returning a new :class:`Header`.

    Notes
    -----
    - ``meta`` is an acquisition-metadata dictionary used by
       predicates/factories.
    - ``multiscale_no`` is the multiscale level index for pipelines that
      downsample by fixed ratios.
    """

    @property
    def name(self) -> str: ...
    @property
    def priority(self) -> int: ...

    def __call__(
        self,
        h: Header,
        meta: dict[str, Any],
        multiscale_no: int,
    ) -> Header: ...


T = TypeVar("T", bound=Overlay)


# -------- Overlay Rule (version/date/meta -> create ONE overlay instance) ----
@dataclass(frozen=True, slots=True)
class OverlayRule:
    """
    Rule describing *when* to instantiate an overlay and *which* overlay to
    use.

    Parameters
    ----------
    name : str
        Identifier for diagnostics.
    spec : packaging.specifiers.SpecifierSet
        Version constraint (PEP 440), e.g. ``">=0.8.0,<0.9.0"``.
    factory : Callable[[dict], Overlay]
        Factory that builds an overlay instance from ``meta`` on selection.
    start, end : datetime.date, optional
        Inclusive (``start``) / exclusive (``end``) acquisition-date bounds.
        If unset, no date filtering is applied.
    predicate : Callable[[dict], bool], optional
        Additional guard; the rule fires only if it returns ``True``.
    rule_priority : int, default 100
        Ordering among rules during **selection** (not execution). Useful for
        group exclusivity and short-circuiting.
    group : str, optional
        Name of an exclusivity bucket. Only the first matching rule in a group
        is applied.
    stop_after : bool, default False
        If ``True``, selection stops after this rule fires.
    """

    name: str
    spec: SpecifierSet  # e.g. ">=0.8.0,<0.9.0"
    factory: Callable[[dict[str, Any]], Overlay]
    start: date | None = None  # inclusive if set
    end: date | None = None  # exclusive if set
    predicate: Callable[[dict[str, Any]], bool] | None = None
    rule_priority: int = 100  # ordering among rules (not overlay.priority)
    group: str | None = None  # optional: exclusivity bucket
    stop_after: bool = False  # optional: short-circuit after this rule fires


# -------- Selector that returns a LIST of overlays to run --------------------
@dataclass(frozen=True, slots=True)
class OverlaySelector:
    r"""
    Immutable selector that collects **all matching** :class:`OverlayRule`\ s.

    Parameters
    ----------
    rules : tuple of OverlayRule, optional
        The rule set. The selector is immutable; use :meth:`with_rule` or
        :meth:`with_rules` to create modified copies.

    Notes
    -----
    - Selection order is by ``rule_priority`` then name (deterministic).
    - Execution order of overlays is by overlay ``priority`` (independent of
      ``rule_priority``).
    - Use ``group`` to enforce mutual exclusivity within subsets of rules.
    - Use ``stop_after=True`` to short-circuit once a rule fires.
    """

    rules: tuple[OverlayRule, ...] = ()

    def select(
        self,
        *,
        version: str,
        meta: dict[str, Any],
    ) -> list[Overlay]:
        """
        Select and instantiate **all** overlays whose rules match ``version``
        and ``meta``.

        Parameters
        ----------
        version : str
            Pipeline version to evaluate against PEP 440 specifiers.
        meta : dict
            Acquisition metadata dictionary available to predicates and
            factories. ``meta['acq_date']`` (date/str) is used for date-range
            filtering if present.

        Returns
        -------
        list[Overlay]
            Instantiated overlays sorted by overlay ``priority`` (ascending).

        Notes
        -----
        - If multiple rules in the same ``group`` match, only the first
          (by ``rule_priority`` then name) is included.
        - If a rule has ``stop_after=True`` and matches, selection stops after
          adding its overlay.
        """
        v = Version(version)
        acq_date = _as_date(meta.get("acq_date"))
        overlays: list[Overlay] = []
        seen_groups: set[str] = set()

        for r in sorted(self.rules, key=lambda x: (x.rule_priority, x.name)):
            if not r.spec.contains(str(v), prereleases=True):
                continue
            if r.start and (not acq_date or acq_date < r.start):
                continue
            if r.end and (not acq_date or acq_date >= r.end):
                continue
            if r.predicate and not r.predicate(meta):
                continue
            if r.group and r.group in seen_groups:
                continue

            overlays.append(r.factory(meta))
            if r.group:
                seen_groups.add(r.group)
            if r.stop_after:
                break

        # Execution order is overlay.priority (not rule_priority)
        overlays.sort(key=lambda ov: (ov.priority, ov.name))
        return overlays

    # ergonomic immutable “adders”
    def with_rule(self, rule: OverlayRule) -> OverlaySelector:
        """
        Return a new selector with ``rule`` appended.

        Parameters
        ----------
        rule : OverlayRule
            The rule to add.

        Returns
        -------
        OverlaySelector
            A new immutable selector containing the extra rule.
        """
        return replace(self, rules=self.rules + (rule,))

    def with_rules(
        self, rules: tuple[OverlayRule, ...] | list[OverlayRule]
    ) -> OverlaySelector:
        """
        Return a new selector with rules from ``rules`` appended.

        Parameters
        ----------
        rules : sequence of OverlayRule
            Rules to add.

        Returns
        -------
        OverlaySelector
            A new immutable selector containing the additional rules.
        """
        return replace(self, rules=self.rules + tuple(rules))


def _as_date(d: Any) -> date | None:
    """
    Normalize an input into a :class:`datetime.date` if possible.

    Parameters
    ----------
    d : Any
        One of ``None``, ``date``, ``datetime``, or ISO-8601 string.

    Returns
    -------
    date or None
        The normalized date, or ``None`` if ``d`` is ``None``.

    Raises
    ------
    ValueError
        If a string value cannot be parsed by ``datetime.fromisoformat``.
    """
    if d is None:
        return None
    if isinstance(d, datetime):  # Check datetime FIRST (before date)
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)).date()


def apply_overlays(
    base: Header,
    overlays: list[Overlay],
    meta: dict[str, Any],
    registration_multiscale_no: int = 3,
) -> tuple[Header, list[str]]:
    """
    Apply a sequence of overlays to a base header in deterministic order.

    Parameters
    ----------
    base : Header
        Starting header (often constructed from acquisition metadata).
    overlays : list of Overlay
        Overlays to apply. Should already be sorted by overlay ``priority``;
        :meth:`OverlaySelector.select` returns them in the correct order.
    meta : dict
        Acquisition metadata provided to each overlay call.
    registration_multiscale_no : int, default 3
        Multiscale pyramid level used by registration pipeline for overlays
        that depend on scale.

    Returns
    -------
    header : Header
        The final header after all overlays are applied.
    applied : list of str
        ``name`` of each overlay that resulted in a change.

    Notes
    -----
    An overlay is considered to have changed the header if any of
    ``origin``, ``spacing``, or ``direction`` differs from the previous value.
    """
    h = base
    applied: list[str] = []
    for ov in overlays:  # already sorted by overlay.priority
        h2 = ov(h, meta, registration_multiscale_no)
        if (
            (h2.origin != h.origin)
            or (h2.spacing != h.spacing)
            or (h2.direction.tobytes() != h.direction.tobytes())
        ):
            applied.append(ov.name)
        h = h2
    return h, applied


# ---- Build your default rules ---------------------------------------------
def _base_rules() -> tuple[OverlayRule, ...]:
    """
    Internal: construct the default built-in rules for the selector.

    Returns
    -------
    tuple of OverlayRule
        The default rule set shipped with the package.

    Notes
    -----
    Replace/extend these with your project’s real factories/overlays.
    """
    rules: list[OverlayRule] = []

    # Examples (replace with your real factories/overlays):

    rules.append(
        OverlayRule(
            name="Fixed world image spacing (0.0144,0.0144,0.016)",
            spec=SpecifierSet(">=0.0.18,<0.0.32"),
            factory=lambda meta: SetLpsWorldSpacingOverlay(
                lps_spacing_mm=(0.0144, 0.0144, 0.016)
            ),
            rule_priority=55,
        )
    )

    rules.append(
        OverlayRule(
            name="anchor RAS corner to recorded bug point",
            spec=SpecifierSet(">=0.0.18,<0.0.34"),
            factory=lambda meta: ForceCornerAnchorOverlay(
                corner_code="RAS",
                target_point_labeled=(-1.5114, -1.5, 1.5),
            ),
            rule_priority=90,
        )
    )

    return tuple(rules)


@lru_cache(maxsize=1)
def get_selector() -> OverlaySelector:
    """
    Return the shared default, frozen :class:`OverlaySelector`.

    Returns
    -------
    OverlaySelector
        Cached selector constructed from :func:`_base_rules` (safe singleton).
    """
    return OverlaySelector(rules=_base_rules())


def extend_selector(*extra: OverlayRule) -> OverlaySelector:
    """
    Build a new frozen selector consisting of the defaults **plus** ``extra``.

    Parameters
    ----------
    *extra : OverlayRule
        Additional rules to append.

    Returns
    -------
    OverlaySelector
        A new immutable selector; the global default is not mutated.
    """
    return replace(get_selector(), rules=get_selector().rules + tuple(extra))


def make_selector(
    rules: tuple[OverlayRule, ...] | list[OverlayRule],
) -> OverlaySelector:
    """
    Construct a selector from a provided rule list/tuple.

    Parameters
    ----------
    rules : list or tuple of OverlayRule
        Rules to include in the selector.

    Returns
    -------
    OverlaySelector
        Frozen selector containing the provided rules.
    """
    return OverlaySelector(rules=tuple(rules))


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
        self, h: Header, meta: dict[str, Any], multiscale_no: int
    ) -> Header:
        """
        Apply the scaling to the spacing.

        Parameters
        ----------
        h : Header
            Input header.
        meta : dict
            Unused.
        multiscale_no : int
            Unused.

        Returns
        -------
        Header
            New header with spacing multiplied by ``scale``.
        """
        i, j, k = h.spacing
        return replace(
            h, spacing=(i * self.scale, j * self.scale, k * self.scale)
        )


@dataclass(frozen=True, slots=True)
class FlipIndexAxesOverlay:
    """
    Flip one or more **index axes** by negating the corresponding columns of
    the direction matrix.

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
        self, h: Header, meta: dict[str, Any], multiscale_no: int
    ) -> Header:
        """
        Negate selected columns of the direction matrix.

        Returns
        -------
        Header
            Header with updated direction matrix.
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
    """
    Permute the **index axes** (i, j, k) and carry spacing/size/direction
    along.

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
        self, h: Header, meta: dict[str, Any], multiscale_no: int
    ) -> Header:
        """
        Reorder columns of ``direction``, elements of ``spacing``, and
        entries of ``size_ijk`` according to ``order``.

        Returns
        -------
        Header
            Header with permuted index axes.
        """
        i0, i1, i2 = self.order
        D = h.direction[:, [i0, i1, i2]]
        S: Vec3 = (h.spacing[i0], h.spacing[i1], h.spacing[i2])
        N: tuple[int, int, int] = (
            h.size_ijk[i0],
            h.size_ijk[i1],
            h.size_ijk[i2],
        )
        return Header(origin=h.origin, spacing=S, direction=D, size_ijk=N)


@dataclass(frozen=True, slots=True)
class ForceCornerAnchorOverlay:
    """
    Set the origin so a particular anatomical corner lands at a target point.

    Uses :func:`aind_zarr_utils.zarr.compute_origin_for_corner` to compute the
    required origin from the current header, a corner code (e.g., ``"RAS"``),
    and a target point expressed in a labeled frame.

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
        self, h: Header, meta: dict[str, Any], multiscale_no: int
    ) -> Header:
        """
        Compute and set the origin such that the specified corner aligns with
        the target point.

        Returns
        -------
        Header
            Header with updated origin.
        """
        origin_lps, _, _ = compute_origin_for_corner(
            size=h.size_ijk,
            spacing=h.spacing,
            direction=h.direction,
            target_point=self.target_point_labeled,
            corner_code=self.corner_code,
            target_frame=self.target_frame,
            use_outer_box=self.use_outer_box,
        )
        return replace(h, origin=origin_lps)


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
        raise ValueError(
            "Direction is not cardinal (signed permutation). "
            "Oblique not supported."
        )


def lps_world_to_index_spacing_cardinal(
    D: np.ndarray, lps_spacing_mm: Vec3
) -> Vec3:
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
        self, h: Header, meta: dict[str, Any], multiscale_no: int
    ) -> Header:
        """
        Convert the desired LPS spacings to index-order spacing and set it.

        Parameters
        ----------
        h : Header
            Input header whose direction determines the mapping.
        meta : dict
            Unused.
        multiscale_no : int
            Multiscale level index. Spacing is downscaled by
            ``(1 / _PIPELINE_MULTISCALE_FACTOR) ** multiscale_no``.

        Returns
        -------
        Header
            Header with updated spacing.
        """
        si, sj, sk = lps_world_to_index_spacing_cardinal(
            h.direction, self.lps_spacing_mm
        )
        scaling = (1 / _PIPELINE_MULTISCALE_FACTOR) ** multiscale_no
        base_spacing: Vec3 = (scaling * si, scaling * sj, scaling * sk)
        return replace(h, spacing=base_spacing)


def estimate_pipeline_multiscale(
    zarr_metadata: dict[str, Any], pipeline_ccf_reg_version: Version
) -> int | None:
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
