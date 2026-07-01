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

from aind_zarr_utils.io.paths import _zarr_base_name_pathlike

logger = logging.getLogger(__name__)
_KNOWN_GOOD_PIPELINE_VERSIONS = {3, 4, 5}


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
    pipeline = _get_processing_pipeline_data(processing_data)
    want_name = "Image importing"
    proc = next(
        (p for p in pipeline["data_processes"] if p.get("name") == want_name),
        None,
    )
    return proc


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
    pipeline = _get_processing_pipeline_data(processing_data)
    want_name = "Image atlas alignment"
    want_notes = "Template based registration: LS -> template -> Allen CCFv3 Atlas"

    proc = next(
        (p for p in pipeline["data_processes"] if p.get("name") == want_name and p.get("notes") == want_notes),
        None,
    )
    return proc


def image_atlas_alignment_path_relative_from_processing(
    processing_data: dict[str, Any],
) -> str | None:
    """
    Return relative path to atlas alignment outputs for a processing run.

    The relative path (if determinable) has the form::

        image_atlas_alignment/<channel>/

    where ``<channel>`` is derived from the base name of the input LS Zarr.

    Parameters
    ----------
    processing_data : dict
        Processing metadata.

    Returns
    -------
    str or None
        Relative path or ``None`` if the required process / channel can't
        be resolved.
    """
    proc = _get_image_atlas_alignment_process(processing_data)
    input_zarr = proc.get("input_location") if proc else None
    channel = _zarr_base_name_pathlike(PurePosixPath(input_zarr)) if input_zarr else None
    rel_path = f"image_atlas_alignment/{channel}/" if channel else None

    return rel_path
