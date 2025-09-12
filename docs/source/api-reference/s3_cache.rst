s3_cache module
===============

.. automodule:: aind_zarr_utils.s3_cache
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``s3_cache`` module provides efficient caching mechanisms for S3 resources. It implements ETag-based validation to avoid unnecessary downloads and supports both persistent and temporary cache directories.

Classes
-------

.. autosummary::
   :toctree: generated/
   
   CacheManager
   CacheResult

Functions
---------

.. autosummary::
   :toctree: generated/
   
   get_local_path_for_resource

Features
--------

- **ETag validation**: Avoids re-downloading unchanged files
- **Flexible caching**: Persistent, temporary, or custom cache directories
- **Context management**: Automatic cleanup of temporary caches
- **S3 optimization**: Efficient handling of large files and metadata

Examples
--------

Basic caching usage::

    from aind_zarr_utils.s3_cache import get_local_path_for_resource, CacheManager

    # Simple caching
    result = get_local_path_for_resource("s3://bucket/large-file.json")
    if result.from_cache:
        print("File was cached!")
    print(f"Local path: {result.path}")

Context manager for temporary cache::

    with CacheManager(persistent=False) as cm:
        result = get_local_path_for_resource(
            "s3://bucket/file.json", 
            cache_dir=cm.dir
        )
        # Use result.path...
    # Cache automatically cleaned up

Persistent cache directory::

    with CacheManager(cache_dir="~/.my_app_cache") as cm:
        result = get_local_path_for_resource(
            "s3://bucket/file.json",
            cache_dir=cm.dir
        )
        # Cache persists after context exit