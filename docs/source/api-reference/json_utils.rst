json_utils module
=================

.. automodule:: aind_zarr_utils.json_utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``json_utils`` module provides unified JSON loading from multiple sources: local files, HTTP/HTTPS URLs, and S3 URIs. It handles authentication and provides both anonymous and authenticated S3 access.

Functions
---------

.. autosummary::
   :toctree: generated/
   
   get_json
   get_json_s3
   get_json_s3_uri
   get_json_url

Examples
--------

Load JSON from different sources::

    from aind_zarr_utils.json_utils import get_json

    # Local file
    data = get_json("/path/to/file.json")
    
    # HTTP URL
    data = get_json("https://example.com/data.json")
    
    # S3 URI (anonymous access)
    data = get_json("s3://aind-open-data/path/to/file.json", anonymous=True)