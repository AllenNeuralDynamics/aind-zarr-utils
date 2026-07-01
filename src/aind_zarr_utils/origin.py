"""LPS origin specification for image construction.

This module exposes a single small value type, :class:`Origin`, which
replaces the legacy three-kwarg dance (``set_origin``, ``set_corner``,
``set_corner_lps``) on the various ``zarr_to_*`` builders. Instances are
constructed via classmethods that name what they mean, and translate to
the legacy kwargs internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Origin:
    """LPS-origin specification for an anatomical image.

    Construct via the classmethods:

    * :meth:`default` — origin at LPS ``(0, 0, 0)``.
    * :meth:`at` — origin set directly to the given LPS point.
    * :meth:`at_corner` — origin chosen so that the named anatomical
      corner of the image lands at the given LPS point.

    Internally this is a tagged union: at most one of ``point`` or the
    pair ``(corner_code, corner_lps)`` is set; both ``None`` means the
    default origin.
    """

    point: tuple[float, float, float] | None = None
    corner_code: str | None = None
    corner_lps: tuple[float, float, float] | None = None

    @classmethod
    def default(cls) -> Origin:
        """Return an :class:`Origin` sentinel meaning ``(0, 0, 0)`` in LPS."""
        return cls()

    @classmethod
    def at(cls, point: tuple[float, float, float]) -> Origin:
        """Return an :class:`Origin` whose value is ``point`` in LPS millimeters."""
        return cls(point=point)

    @classmethod
    def at_corner(
        cls,
        corner_code: str,
        lps_point: tuple[float, float, float],
    ) -> Origin:
        """Return an :class:`Origin` that anchors a labelled corner to ``lps_point``.

        Parameters
        ----------
        corner_code : str
            Three-letter anatomical corner code (e.g. ``"RAI"`` for the
            Right-Anterior-Inferior corner of the volume).
        lps_point : tuple of three floats
            Target LPS coordinates of that corner, in millimeters.
        """
        return cls(corner_code=corner_code, corner_lps=lps_point)

    def _legacy_kwargs(self) -> dict[str, Any]:
        """Translate to ``(set_origin, set_corner, set_corner_lps)`` kwargs.

        Used by :class:`~aind_zarr_utils.asset.Asset` to delegate to the
        existing ``zarr_to_sitk`` / ``zarr_to_ants`` / ``zarr_to_sitk_stub``
        free functions without introducing a public-API change.
        """
        return {
            "set_origin": self.point,
            "set_corner": self.corner_code,
            "set_corner_lps": self.corner_lps,
        }
