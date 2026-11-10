"""Parsing helpers for AIND ``processing.json`` pipeline metadata.

The SmartSPIM pipeline records its provenance in ``processing.json``: a
sequence of named ``data_processes`` plus a top-level ``processing_pipeline``
block recording the pipeline version. The helpers in this module locate the
specific processes the rest of the package relies on (``Image importing``,
``Image atlas alignment``) and derive paths that depend on them (e.g. the
relative location of the atlas-alignment outputs).
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any

from aind_metadata_reader import Document, Process, read_processing
from aind_metadata_reader.smartspim import registration_input

from aind_zarr_utils.io.paths import _zarr_base_name_pathlike

logger = logging.getLogger(__name__)
_KNOWN_GOOD_PIPELINE_VERSIONS = {3, 4, 5}


def _get_zarr_import_view(processing_data: dict[str, Any] | Process, zarr_uri: str | None = None) -> Process | None:
    """Select import provenance explicitly, resolving repeated steps by recorded channel."""
    if isinstance(processing_data, Process):
        if processing_data.process_type != "Image importing":
            raise ValueError("Selected registration must be an Image importing process")
        return processing_data
    view = read_processing(Document(processing_data))
    if view.reader is None:
        raise ValueError(f"Unsupported processing representation: {view.diagnostics}")
    candidates = view.find("Image importing")
    if zarr_uri is not None and len(candidates) > 1:
        channel = _zarr_base_name_pathlike(PurePosixPath(zarr_uri))
        matches = tuple(
            p
            for p in candidates
            if any(
                _zarr_base_name_pathlike(PurePosixPath(r.location.value)) == channel
                for r in registration_input(p).value or ()
                if r.location.value is not None
            )
        )
        if matches:
            candidates = matches
        elif any(registration_input(p).value for p in candidates):
            raise ValueError(f"No registration import matches channel {channel!r}")
    if len(candidates) > 1:
        raise ValueError("Ambiguous registration import; supply a selected Process or an unambiguous channel")
    return candidates[0] if candidates else None


def _registration_input_path(processing_data: dict[str, Any] | Process, zarr_uri: str | None = None) -> str | None:
    """Resolve a registration's file input without treating asset URLs as files."""
    imported = _get_zarr_import_view(processing_data, zarr_uri)
    if imported is not None:
        paths = registration_input(imported)
        if paths.value is not None:
            if len(paths.value) != 1:
                raise ValueError("Registration import records multiple inputs; an explicit selection is required")
            return paths.value[0].location.value
    if isinstance(processing_data, Process):
        return None
    # Historical records can describe atlas alignment without a separate import step.
    aligned = _get_image_atlas_alignment_process(processing_data)
    if aligned is None:
        return None
    view = read_processing(Document(processing_data))
    matching = [p for p in view.find("Image atlas alignment") if p.document.raw == aligned]
    aligned_inputs = matching[0].inputs.value
    if aligned_inputs is not None and len(aligned_inputs) == 1 and aligned_inputs[0].kind == "file":
        return aligned_inputs[0].location.value
    return None


def _registration_pipeline_major(processing_data: dict[str, Any] | Process, zarr_uri: str | None = None) -> int:
    """Read the selected registration's pipeline version independently of schema version."""
    imported = _get_zarr_import_view(processing_data, zarr_uri)
    version = imported.pipeline_version if imported is not None else None
    if version is None and not isinstance(processing_data, Process):
        view = read_processing(Document(processing_data))
        if view.reader is not None and view.reader.format.layout == "legacy":
            version = view.document.child("/processing_pipeline").optional_string("/pipeline_version")
    if not version:
        raise ValueError("Missing pipeline version")
    major = int(version.split(".")[0])
    if major < min(_KNOWN_GOOD_PIPELINE_VERSIONS):
        raise ValueError(f"Unsupported pipeline version: {major}")
    if major > max(_KNOWN_GOOD_PIPELINE_VERSIONS):
        logger.warning(
            "Pipeline version %s is greater than max verified version %s", major, max(_KNOWN_GOOD_PIPELINE_VERSIONS)
        )
    return major


def _get_processing_pipeline_data(
    processing_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Return validated processing pipeline metadata.

    Parameters
    ----------
    processing_data : dict
        Top-level metadata (e.g. contents of ``processing.json``) expected
        to contain a ``processing_pipeline`` key with a semantic version.

    Returns
    -------
    dict
        The nested ``processing_pipeline`` dictionary.

    Raises
    ------
    ValueError
        If the pipeline version is missing or the major version is not 3.
    """
    ver_str = processing_data.get("processing_pipeline", {}).get("pipeline_version", None)
    if not ver_str:
        raise ValueError("Missing pipeline version")
    pipeline_ver = int(ver_str.split(".")[0])
    if pipeline_ver not in _KNOWN_GOOD_PIPELINE_VERSIONS:
        maxver = max(_KNOWN_GOOD_PIPELINE_VERSIONS)
        if pipeline_ver > maxver:
            logger.warning(
                f"Pipeline version {pipeline_ver} is greater than max "
                f"verified version {maxver}, results may not be accurate. "
                "File an issue at "
                "https://github.com/AllenNeuralDynamics/aind-zarr-utils/issues."
            )
        else:
            raise ValueError(f"Unsupported pipeline version: {pipeline_ver}")
    pipeline: dict[str, Any] = processing_data.get("processing_pipeline", {})
    return pipeline


def _get_zarr_import_process(
    processing_data: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Locate the *Image importing* process block.

    Parameters
    ----------
    processing_data : dict
        Processing metadata supplying ``data_processes`` list.

    Returns
    -------
    dict or None
        Matching process dict or ``None`` if not present.
    """
    proc = _get_zarr_import_view(processing_data)
    return proc.document.raw if proc is not None else None


def _get_image_atlas_alignment_process(
    processing_data: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Locate the *Image atlas alignment* process for SmartSPIM → CCF.

    The process is uniquely identified by name plus a notes string describing
    the LS → template → CCF chain.

    Parameters
    ----------
    processing_data : dict
        Processing metadata.

    Returns
    -------
    dict or None
        Matching process dict or ``None`` if not found.
    """
    want_name = "Image atlas alignment"
    want_notes = "Template based registration: LS -> template -> Allen CCFv3 Atlas"

    view = read_processing(Document(processing_data))
    candidates = [p for p in view.find(want_name) if p.document.field("/notes").value == want_notes]
    if len(candidates) > 1:
        raise ValueError("Ambiguous atlas alignment process")
    return candidates[0].document.raw if candidates else None


def image_atlas_alignment_path_relative_from_processing(
    processing_data: dict[str, Any] | Process,
    *,
    zarr_uri: str | None = None,
) -> str | None:
    """
    Return relative path to atlas alignment outputs for a processing run.

    The relative path (if determinable) has the form::

        image_atlas_alignment/<channel>/

    where ``<channel>`` is derived from the base name of the input LS Zarr.

    Parameters
    ----------
    processing_data : dict or Process
        Processing metadata or an explicitly selected registration import,
        retaining its source and pipeline association.
    zarr_uri : str, optional
        Select between multiple registration inputs by recorded channel.

    Returns
    -------
    str or None
        Relative path or ``None`` if the required process / channel can't
        be resolved.
    """
    input_zarr = _registration_input_path(processing_data, zarr_uri)
    channel = _zarr_base_name_pathlike(PurePosixPath(input_zarr)) if input_zarr else None
    rel_path = f"image_atlas_alignment/{channel}/" if channel else None

    return rel_path
