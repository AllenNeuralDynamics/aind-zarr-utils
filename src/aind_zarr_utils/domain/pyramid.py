"""Version-keyed corrections for SmartSPIM fusion pyramid translations.

No SmartSPIM fusion release writes correct level > 0 translations. Levels are
aligned block means, so level ``k`` should sit ``(2**k - 1) / 2`` level-0 voxels
past level 0, in physical units. Which defect an output carries depends on which
writer produced it, so rules match the fusion process's ``code_version`` together
with the writer that actually ran:

* the capsule's own Python writer records no translations;
* BigStitcher-Spark records the offset in level-0 voxels under micrometer axes.

The geometry itself is computed by
:func:`aind_metadata_reader.smartspim.evaluate_pyramid_translation_correction`;
a matching rule supplies the evidence that rule requires. Registration-frame
geometry is unaffected: pipeline overlays place that frame independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from typing import Any, Literal
from warnings import warn

from aind_metadata_reader import (
    Applicability,
    Document,
    Interpretation,
    Process,
    apply_changes,
    read_processing,
)
from aind_metadata_reader.smartspim import evaluate_pyramid_translation_correction
from ome_zarr.reader import Node  # type: ignore[import-untyped]
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from aind_zarr_utils.io.ngff import attributes_document, interpreted_zarr
from aind_zarr_utils.io.paths import _zarr_base_name_any

_FUSION_PROCESS = "Image tile fusing"
_BIGSTITCHER_ATTRIBUTE = "Bigstitcher-Spark"
_PYTHON_WRITER = "aind-smartspim-fuse"
_BIGSTITCHER_WRITER = "BigStitcher-Spark"

Signature = Literal["missing", "voxel_units"]


@dataclass(frozen=True, slots=True)
class FusionPyramidRule:
    """Which fusion versions and writer carry one pyramid-translation defect.

    Parameters
    ----------
    name : str
        Identifier used in warnings.
    spec : packaging.specifiers.SpecifierSet
        Fusion ``code_version`` range the rule covers.
    writer : str
        Writer that must have produced the Zarr (see :func:`fusion_writer`).
    signature : {"missing", "voxel_units"}
        Level > 0 translations the defect leaves: none, or present in voxels.
        Outputs that disagree are left uncorrected, since the table is then wrong.
    evidence : str
        Why levels are aligned block means with this defect; recorded in the
        interpretation's audit trail.
    last_verified_version : str
        Newest version checked against the writer's code. Newer versions are
        still corrected, with a warning.
    """

    name: str
    spec: SpecifierSet
    writer: str
    signature: Signature
    evidence: str
    last_verified_version: str


DEFAULT_FUSION_PYRAMID_RULES: tuple[FusionPyramidRule, ...] = (
    FusionPyramidRule(
        name="aind-smartspim-fuse Python writer omits pyramid translations",
        spec=SpecifierSet("<0.0.4"),
        writer=_PYTHON_WRITER,
        signature="missing",
        evidence=(
            "aind-smartspim-fuse zarr_writer (create_multiscales.py, smartspim_zarr_writer.py) reduces levels "
            "with xarray_multiscale windowed_mean and passes translation=None to _compute_scales"
        ),
        last_verified_version="0.0.3",
    ),
    FusionPyramidRule(
        name="BigStitcher-Spark writes pyramid translations in voxels",
        spec=SpecifierSet(">=0.0.3"),
        writer=_BIGSTITCHER_WRITER,
        signature="voxel_units",
        evidence=(
            "aind-smartspim-fuse pins BigStitcher-Spark e112363 (multiview-reconstruction 5.2.5): levels are "
            "Downsample.downsample(factors) block means and OMEZarrAttibutes.createOMEZarrMetadata writes the "
            "mipmap translation in level-0 voxels without the level-0 resolution"
        ),
        last_verified_version="0.0.6",
    ),
)


def fusion_producer(processing: Mapping[str, Any], zarr_uri: str) -> Process | None:
    """Return the one fusion process whose output is this Zarr's channel, if exactly one exists."""
    channel = _zarr_base_name_any(zarr_uri)
    if channel is None:
        return None
    candidates = [
        p
        for p in read_processing(Document(dict(processing))).find(_FUSION_PROCESS)
        if p.output_location and _zarr_base_name_any(p.output_location) == channel
    ]
    return candidates[0] if len(candidates) == 1 else None


def fusion_writer(producer: Process, attributes: Document) -> str | None:
    """Identify the writer from the tool's own attribute block and the process notes.

    ``None`` when the two disagree. ``code_version`` cannot decide this: one
    capsule version can run either writer.
    """
    tool_attributes = isinstance(attributes.raw, dict) and _BIGSTITCHER_ATTRIBUTE in attributes.raw
    notes = producer.document.field("/notes").value
    if isinstance(notes, str) and ("BigStitcher" in notes) != tool_attributes:
        return None
    return _BIGSTITCHER_WRITER if tool_attributes else _PYTHON_WRITER


def _translations_present(attributes: Document) -> bool | None:
    """Whether every level > 0 records a translation; ``None`` when levels disagree or cannot be read.

    Presence alone cannot separate voxel from physical units; the reader's rule does that.
    """
    try:
        datasets = attributes.raw["multiscales"][0]["datasets"][1:]
        present = [
            any(isinstance(t, dict) and t.get("type") == "translation" for t in d["coordinateTransformations"])
            for d in datasets
        ]
    except (KeyError, IndexError, TypeError):
        return None
    if not present or (any(present) and not all(present)):
        return None
    return all(present)


def _select_rule(
    producer: Process, writer: str, rules: tuple[FusionPyramidRule, ...]
) -> tuple[FusionPyramidRule | None, Version | None]:
    try:
        code_version = Version(producer.code_version or "")
    except InvalidVersion:
        return None, None
    matching = [r for r in rules if r.writer == writer and r.spec.contains(code_version, prereleases=True)]
    return (matching[0] if matching else None), code_version


def fusion_pyramid_interpretation(
    zarr_uri: str,
    opened_zarr: tuple[Node, dict],
    processing: Mapping[str, Any],
    *,
    producer: Process | None = None,
    rules: tuple[FusionPyramidRule, ...] | None = None,
) -> Interpretation | None:
    """Correct this Zarr's pyramid translations when its fusion producer matches a rule.

    Parameters
    ----------
    zarr_uri : str
        URI of the fused channel Zarr.
    opened_zarr : tuple
        ``(Node, metadata)`` from :func:`aind_zarr_utils.io.zarr._open_zarr`, uninterpreted.
    processing : Mapping
        Parsed ``processing.json``; used to find the producer when ``producer`` is not given.
    producer : Process, optional
        The fusion process that wrote this Zarr, when the channel match is ambiguous.
    rules : tuple of FusionPyramidRule, optional
        Defaults to :data:`DEFAULT_FUSION_PYRAMID_RULES`. ``()`` disables correction.

    Returns
    -------
    Interpretation or None
        The evaluated interpretation (possibly with no changes, for already-correct
        translations), or ``None`` when no rule applies; recorded geometry is then used.
        Every case that leaves a recognized defect in place warns.
    """
    rules = DEFAULT_FUSION_PYRAMID_RULES if rules is None else rules
    if not rules:
        return None
    producer = producer or fusion_producer(processing, zarr_uri)
    if producer is None:
        warn(
            f"No unique '{_FUSION_PROCESS}' process names {zarr_uri}; pyramid levels use recorded NGFF translations",
            stacklevel=3,
        )
        return None
    try:
        attributes = attributes_document(zarr_uri, opened_zarr)
    except ValueError as error:
        warn(f"{error}; pyramid levels of {zarr_uri} use recorded NGFF translations", stacklevel=3)
        return None
    writer = fusion_writer(producer, attributes)
    if writer is None:
        warn(
            f"Fusion notes and Zarr attributes disagree about the writer of {zarr_uri}; "
            "pyramid levels use recorded NGFF translations",
            stacklevel=3,
        )
        return None
    rule, code_version = _select_rule(producer, writer, rules)
    if rule is None:
        warn(
            f"No pyramid-translation rule covers {writer} at fusion version {producer.code_version!r}; "
            "pyramid levels use recorded NGFF translations",
            stacklevel=3,
        )
        return None
    present = _translations_present(attributes)
    if present is None or present != (rule.signature == "voxel_units"):
        observed = "irregular" if present is None else ("present" if present else "absent")
        warn(
            f"{zarr_uri} level translations are {observed}, but rule {rule.name!r} expects {rule.signature} "
            "translations; pyramid levels use recorded NGFF translations",
            stacklevel=3,
        )
        return None
    if code_version is not None and code_version > Version(rule.last_verified_version):
        warn(
            f"Fusion version {code_version} is newer than {rule.last_verified_version}, the last verified for "
            f"rule {rule.name!r}; applying it on the assumption the defect persists",
            stacklevel=3,
        )
    evaluation = evaluate_pyramid_translation_correction(
        attributes,
        producer,
        author=f"aind-zarr-utils {_package_version('aind-zarr-utils')}",
        association_evidence=(
            (
                f"Fusion process {producer.document.pointer or '/'} is the unique '{_FUSION_PROCESS}' entry for "
                f"this channel; writer {writer} identified from Zarr attributes and process notes"
            ),
        ),
        averaging_evidence=(rule.evidence,),
    )
    if evaluation.outcome == Applicability.UNKNOWN:
        warn(
            f"Pyramid translation correction was not applied to {zarr_uri}: {evaluation.reason}; "
            "using recorded NGFF translations",
            stacklevel=3,
        )
    return apply_changes(
        attributes,
        evaluation.changes,
        applied_at=datetime.now(timezone.utc).isoformat(),
        evaluations=(evaluation,),
    )


def fusion_corrected_zarr(
    zarr_uri: str,
    opened_zarr: tuple[Node, dict],
    processing: Mapping[str, Any],
    *,
    rules: tuple[FusionPyramidRule, ...] | None = None,
) -> tuple[Node, dict]:
    """Return ``opened_zarr`` with rule-matched pyramid translations corrected; pixels are untouched."""
    interpretation = fusion_pyramid_interpretation(zarr_uri, opened_zarr, processing, rules=rules)
    return opened_zarr if interpretation is None else interpreted_zarr(zarr_uri, opened_zarr, interpretation)
