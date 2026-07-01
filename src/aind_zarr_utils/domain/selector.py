"""Overlay protocol, rules, and selector for SmartSPIM pipeline corrections.

This module defines *what* an overlay is (the :class:`Overlay` protocol),
*when* a particular overlay is selected (:class:`OverlayRule` /
:class:`OverlaySelector`), and *how* a list of selected overlays is
applied to an :class:`~aind_anatomical_utils.anatomical_volume.AnatomicalHeader`
in deterministic order.

The default rule set lives in :func:`_base_rules`; the cached singleton
returned by :func:`get_selector` carries those rules.

Version-Aware Corrections
-------------------------
Some overlays apply corrections for known bugs in specific zarr import
pipeline versions. When applying these corrections to versions beyond
the last verified buggy version, warnings are emitted to alert users
that the correction is being applied optimistically (assuming the bug
persists in newer code).

Notes
-----
- All coordinates are expressed in **ITK LPS** convention and **millimeters**.
- SimpleITK direction matrices are 3×3, flattened row-major by
  Get/SetDirection. **Columns** are the LPS-world unit vectors of the image
  **index axes** (``i``, ``j``, ``k``). The mapping used here is::

      physical = origin + direction @ (spacing ⊙ index)

  where ``index = [i, j, k]^T`` and ``spacing`` is in **index order**.
- Oblique (non-cardinal) direction matrices are **not supported** by the
  spacing helpers in :mod:`aind_zarr_utils.domain.overlays`. They will
  raise with a clear error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Protocol, TypeVar

from aind_anatomical_utils.anatomical_volume import AnatomicalHeader
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from aind_zarr_utils.domain.overlays import (
    ForceCornerAnchorOverlayWithVersionWarning,
    SetLpsWorldSpacingOverlay,
)


# Callable overlay interface (dataclass implementations satisfy this)
class Overlay(Protocol):
    """
    Protocol for callable overlays that transform a :class:`AnatomicalHeader`.

    Required interface
    ------------------
    name : str
        Human-friendly identifier used in logs/audits.
    priority : int
        Execution priority (lower numbers run earlier). Independent of
        :attr:`OverlayRule.rule_priority`.
    __call__(h, meta, multiscale_no) -> AnatomicalHeader
        Apply the overlay, returning a new :class:`AnatomicalHeader`.

    Notes
    -----
    - ``meta`` is an acquisition-metadata dictionary used by
       predicates/factories.
    - ``multiscale_no`` is the multiscale level index for pipelines that
      downsample by fixed ratios.
    - ``zarr_import_version`` is an optional parameter for version-aware
      overlays that emit warnings or change behavior based on the zarr
      import process version.
    """

    @property
    def name(self) -> str:
        """Human-friendly identifier for the overlay (used in logs/audits)."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority; lower numbers run earlier."""
        ...

    def __call__(
        self,
        h: AnatomicalHeader,
        meta: dict[str, Any],
        multiscale_no: int,
        zarr_import_version: str | None = None,
    ) -> AnatomicalHeader:
        """Apply the overlay and return a new ``AnatomicalHeader``."""
        ...


T = TypeVar("T", bound=Overlay)


# -------- Overlay Rule (version/date/meta -> create ONE overlay instance) ----
@dataclass(frozen=True, slots=True)
class OverlayRule:
    """Rule describing *when* to instantiate an overlay and *which* overlay to use.

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
        """Select and instantiate **all** overlays whose rules match ``version`` and ``meta``.

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

    def with_rules(self, rules: tuple[OverlayRule, ...] | list[OverlayRule]) -> OverlaySelector:
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
    base: AnatomicalHeader,
    overlays: list[Overlay],
    meta: dict[str, Any],
    registration_multiscale_no: int = 3,
    zarr_import_version: str | None = None,
) -> tuple[AnatomicalHeader, list[str]]:
    """
    Apply a sequence of overlays to a base header in deterministic order.

    Parameters
    ----------
    base : AnatomicalHeader
        Starting header (often constructed from acquisition metadata).
    overlays : list of Overlay
        Overlays to apply. Should already be sorted by overlay ``priority``;
        :meth:`OverlaySelector.select` returns them in the correct order.
    meta : dict
        Acquisition metadata provided to each overlay call.
    registration_multiscale_no : int, default 3
        Multiscale pyramid level used by registration pipeline for overlays
        that depend on scale.
    zarr_import_version : str or None, optional
        Zarr import process version, provided to overlays for version-aware
        behavior (e.g., warnings for untested versions).

    Returns
    -------
    header : AnatomicalHeader
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
        h2 = ov(
            h,
            meta,
            registration_multiscale_no,
            zarr_import_version=zarr_import_version,
        )
        if (h2.origin != h.origin) or (h2.spacing != h.spacing) or (h2.direction.tobytes() != h.direction.tobytes()):
            applied.append(ov.name)
        h = h2
    return h, applied


# ---- Build your default rules ---------------------------------------------
def _base_rules() -> tuple[OverlayRule, ...]:
    """
    Construct the default built-in rules for the selector (internal helper).

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
            factory=lambda meta: SetLpsWorldSpacingOverlay(lps_spacing_mm=(0.0144, 0.0144, 0.016)),
            rule_priority=55,
        )
    )

    rules.append(
        OverlayRule(
            name="anchor RAS corner to recorded bug point",
            spec=SpecifierSet(">=0.0.18"),
            factory=lambda meta: ForceCornerAnchorOverlayWithVersionWarning(
                corner_code="RAS",
                target_point_labeled=(-1.5114, -1.5, 1.5),
                last_verified_buggy_version="0.0.34",
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
