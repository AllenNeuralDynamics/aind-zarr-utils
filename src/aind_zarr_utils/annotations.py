"""
Module for working with points in ZARR files
"""

import re

from aind_anatomical_utils.sitk_volume import (
    transform_sitk_indices_to_physical_points,
)


def annotation_indices_to_anatomical(img, annotations):
    """
    Transforms annotation indices from image space to anatomical space.

    Parameters
    ----------
    img : SimpleITK.Image
        The reference image.
    annotations : dict
        Dictionary where keys are annotation names and values are numpy arrays
        of indices.

    Returns
    -------
    physical_points : dict
        Dictionary where keys are annotation names and values are physical
        points.
    """
    physical_points = {}
    for annotation, indices in annotations.items():
        indices_sitk = indices[:, ::-1]  # numpy to sitk indexing
        physical_points[annotation] = (
            transform_sitk_indices_to_physical_points(
                img,
                indices_sitk,
            )
        )
    return physical_points


def annotations_and_descriptions_to_dict(annotation_points, descriptions):
    """
    Converts annotation points and descriptions to a description to point
    dictionary.

    Parameters
    ----------
    annotation_points : dict
        Dictionary where keys are annotation names and values are lists of
        points.
    descriptions : dict
        Dictionary where keys are annotation names and values are lists of
        descriptions.

    Returns
    -------
    dict
        Dictionary where keys are annotation names and values are point
        dictionaries.
    """
    pt_dicts = {}
    for annotation_name, points in annotation_points.items():
        description_list = descriptions[annotation_name]
        pt_dict = _pts_and_descriptions_to_pt_dict(points, description_list)
        pt_dicts[annotation_name] = pt_dict
    return pt_dicts


def _pts_and_descriptions_to_pt_dict(points, description_list):
    """
    Converts points and their descriptions into a dictionary.

    Parameters
    ----------
    points : list of list
        List of points, where each point is a list of coordinates.
    description_list : list of str or None
        List of descriptions corresponding to the points. If None, numeric
        labels are assigned.

    Returns
    -------
    dict
        Dictionary where keys are descriptions and values are points.
    """
    pt_dict = {}
    j = 1
    for i, point in enumerate(points):
        pt_description = description_list[i]
        if pt_description is None:
            pt_description_sanitized = f"{j}"
            j += 1
        else:
            pt_description_sanitized = re.sub(
                r"[\r\n,]+", "", pt_description.strip()
            )
        pt_dict[pt_description_sanitized] = point
    return pt_dict
