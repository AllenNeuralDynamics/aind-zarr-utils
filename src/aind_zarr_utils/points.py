import numpy as np


def index_to_physical_space(sitk_image, points):
    """
    Convert points from index space to physical space using a SimpleITK image.
    Parameters
    ----------
    sitk_image : SimpleITK.Image
        The SimpleITK image used to transform the points.
        Note that the image can be a "stub" image, i.e. it does not need to
        contain any data, but it must have the correct metadata. See
        zarr.zarr_to_stub_image for details.

    points : np.ndarray
        An array of points in index space, shape (N, 3) where N is the number of
        points.

    Returns
    -------
    pts_physical : np.ndarray
        An array of points in physical space, shape (N, 3).

    """
    pts_physical = np.empty(points.shape)
    for ii in range(points.shape[0]):
        pts_physical[ii, :] = (
            sitk_image.TransformContinuousIndexToPhysicalPoint(points[ii, :])
        )
    return pts_physical


def physical_to_index_space(sitk_image, points):
    """
    Convert points from physical space to index space using a SimpleITK image.
    Parameters
    ----------
    sitk_image : SimpleITK.Image
        The SimpleITK image used to transform the points.
        Note that the image can be a "stub" image, i.e. it does not need to
        contain any data, but it must have the correct metadata. See
        zarr.zarr_to_stub_image for details.

    points : np.ndarray
        An array of points in physical space, shape (N, 3) where N is the number
        of points.

    Returns
    -------
    pts_index : np.ndarray
        An array of points in index space, shape (N, 3).

    """
    pts_index = np.empty(points.shape)
    for ii in range(points.shape[0]):
        pts_index[ii, :] = sitk_image.TransformPhysicalPointToContinuousIndex(
            points[ii, :]
        )
    return pts_index
