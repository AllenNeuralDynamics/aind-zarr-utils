# Changelog

All notable changes to aind-zarr-utils will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation website with Sphinx
- User guides for ZARR conversion, S3 integration, coordinate systems, and pipeline corrections
- Complete API reference documentation with autosummary
- Installation, quickstart, and examples guides
- Development setup and testing procedures documentation
- Bug reporting guidelines

### Changed
- Enhanced docstrings across all modules for better API documentation
- Improved error messages with more specific guidance

### Fixed
- Documentation build warnings and missing cross-references

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of aind-zarr-utils
- Core ZARR to SimpleITK/ANTs image conversion functionality
- Neuroglancer annotation processing with coordinate transformations
- Multi-source JSON loading (S3, HTTP, local files) with caching
- Pipeline-specific domain corrections for SmartSPIM compatibility
- S3 integration with anonymous access and ETag-based caching
- URI manipulation utilities for cross-platform path handling
- Point coordinate transformations from image to anatomical space
- Pipeline transformation chains for CCF registration

### Core Modules
- `zarr`: ZARR file conversion to medical imaging formats
- `neuroglancer`: Neuroglancer state processing and annotation extraction
- `annotations`: Point transformation utilities
- `json_utils`: Unified JSON loading with S3 support
- `pipeline_transformed`: Pipeline-specific coordinate transformations
- `pipeline_domain_selector`: Version-aware spatial domain corrections
- `s3_cache`: Efficient S3 resource caching with ETag validation
- `uri_utils`: Cross-platform URI and path manipulation

### Features
- **Multi-resolution support**: Process ZARR files at different resolution levels
- **Coordinate system standardization**: All outputs in LPS (Left-Posterior-Superior) coordinates
- **Unit conversion**: Automatic scaling between micrometers, millimeters, etc.
- **Pipeline compatibility**: Reproduce spatial domains from any SmartSPIM pipeline version
- **Efficient caching**: Reduce S3 transfer costs with intelligent local caching
- **Anonymous S3 access**: No credentials needed for public AIND datasets
- **Memory optimization**: Stub images for coordinate-only operations

## Version Format

This project uses [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Change Categories

### Added
New features or functionality

### Changed
Changes to existing functionality (non-breaking)

### Deprecated
Features that will be removed in future versions

### Removed
Features removed in this version

### Fixed
Bug fixes

### Security
Security vulnerability fixes

## Release Process

1. **Version Bump**: Update version in `src/aind_zarr_utils/__init__.py`
2. **Update Changelog**: Document all changes since last release
3. **Tag Release**: Create git tag with version number
4. **GitHub Release**: Create GitHub release with changelog notes
5. **PyPI Upload**: Automated upload to PyPI via GitHub Actions

## Migration Guides

### From 0.x.x to 1.0.0 (Future)

When version 1.0.0 is released, this section will contain:
- Breaking changes and how to adapt code
- Deprecated feature replacements
- New recommended usage patterns

## Notable Dependencies

### Core Dependencies
- **numpy**: Numerical computing foundation
- **ome-zarr**: ZARR file format support
- **SimpleITK**: Medical image processing
- **antspyx**: Advanced normalization tools
- **boto3**: AWS S3 integration
- **requests**: HTTP client for URL fetching

### Version Compatibility

| aind-zarr-utils | Python | SimpleITK | ANTs | ome-zarr |
|----------------|--------|-----------|------|----------|
| 0.1.x          | 3.10+  | 2.2+      | 0.3+ | 0.7+     |
| 1.x.x (future) | 3.11+  | 2.3+      | 0.4+ | 0.8+     |

## Contributing

See [Development Setup](../contributing/development.md) for information about:
- Setting up development environment
- Running tests and quality checks
- Submitting pull requests
- Documentation contributions

## Support

- **Documentation**: [Full documentation](https://aind-zarr-utils.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/AllenNeuralDynamics/aind-zarr-utils/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AllenNeuralDynamics/aind-zarr-utils/discussions)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This changelog is automatically updated with each release. For detailed commit history, see the [GitHub commit log](https://github.com/AllenNeuralDynamics/aind-zarr-utils/commits/main).*