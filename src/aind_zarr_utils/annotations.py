"""
Module for working with points in ZARR files
"""

import re

from aind_mri_utils.file_io.neuroglancer import (
    get_neuroglancer_annotation_points,
)
from aind_anatomical_utils.sitk_volume import (
    transform_sitk_indices_to_physical_points,
)

from aind_zarr_utils.utils import read_metadata_json
from aind_zarr_utils.zarr import zarr_to_stub_image


def transform_annotation_indices(img, annotations):
    """
    Transforms annotation indices from image space to physical space.

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


def _convert_annotation_pts_to_pt_dicts(annotation_points, descriptions):
    """
    Converts annotation points and descriptions into dictionaries.

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


def transform_neuroglancer_annotations_to_physical_points(
    zarr_s3_path,
    acq_metadata_s3,
    ng_annotation_path,
):
    """
    Transforms Neuroglancer annotations to physical points in the image space.

    Parameters
    ----------
    zarr_s3_path : str
        S3 path to the Zarr file.
    acq_metadata_s3 : str
        S3 path to the acquisition metadata JSON file.
    ng_annotation_path : str
        Path to the Neuroglancer annotation file.

    Returns
    -------
    physical_points : dict
        Dictionary where keys are annotation names and values are physical
        points.
    descriptions : dict
        Dictionary where keys are annotation names and values are lists of
        descriptions.
    """
    acq_metadata = read_metadata_json(acq_metadata_s3)
    stub_img = zarr_to_stub_image(zarr_s3_path, acq_metadata)
    annotations, descriptions = get_neuroglancer_annotation_points(
        ng_annotation_path,
    )
    annotation_points = transform_annotation_indices(
        stub_img,
        annotations,
    )
    return annotation_points, descriptions
