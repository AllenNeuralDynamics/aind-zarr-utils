# Versioned Documentation

Guide for building and maintaining versioned documentation using sphinx-multiversion.

## Overview

The aind-zarr-utils documentation supports multiple versions using [sphinx-multiversion](https://holzhaus.github.io/sphinx-multiversion/), allowing users to access documentation for different package versions.

## Features

### Version Selector
- **Dropdown menu** in the documentation sidebar
- **Latest/stable** version indicators  
- **Tag-based versioning** for releases
- **Branch-based** for development versions

### Automatic Building
- **All git tags** matching `v*.*.*` pattern
- **Main branch** for latest development
- **Stable branch** if it exists

## Building Versioned Docs

### Single Version (Development)

```bash
# Standard single-version build
uv run --group docs sphinx-build -b html docs/source docs/build/html
```

### All Versions (Production)

```bash
# Build all versions with sphinx-multiversion
uv run --group docs sphinx-multiversion docs/source docs/build/html
```

**Output Structure:**
```
docs/build/html/
├── main/                    # Latest development (main branch)
├── v0.1.4/                  # Current release
├── v0.1.3/                  # Previous release
├── v0.1.2/                  # Older release
├── ...                      # All tagged versions
└── index.html               # Version selector page
```

## Configuration

The multiversion behavior is configured in `docs/source/conf.py`:

```python
# -- Sphinx Multiversion configuration --------------------------------------
# Whitelist pattern for branches (regex)
smv_branch_whitelist = r"^(main|stable).*$"

# Whitelist pattern for tags (regex) 
smv_tag_whitelist = r"^v\d+\.\d+.*$"

# Whitelist pattern for remotes (regex)
smv_remote_whitelist = r"^(origin|upstream).*$"

# Pattern for released versions (appears in sidebar)
smv_released_pattern = r"^tags/.*$"

# Pattern for in-development versions
smv_outputdir_format = "{ref.name}"

# Latest version shown first in version selector
smv_prefer_remote_refs = False

# Default branch to show when visiting docs without version
smv_latest_version = "main"
```

### Configuration Options

| Setting | Purpose | Example |
|---------|---------|---------|
| `smv_branch_whitelist` | Which branches to build | `main`, `stable`, `develop` |
| `smv_tag_whitelist` | Which tags to build | `v1.0.0`, `v2.1.3` |
| `smv_remote_whitelist` | Which remotes to check | `origin`, `upstream` |
| `smv_released_pattern` | Released vs dev versions | Tags = released, branches = dev |
| `smv_latest_version` | Default version to show | `main`, `stable` |

## Version Management

### Creating a New Release

1. **Update version** in code:
   ```bash
   # Update version in pyproject.toml and __init__.py
   ```

2. **Create and push tag**:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

3. **Rebuild documentation**:
   ```bash
   uv run --group docs sphinx-multiversion docs/source docs/build/html
   ```

4. **Deploy** (if using automated deployment)

### Version Patterns

The configuration accepts these version patterns:

✅ **Accepted Tags:**
- `v1.0.0` - Standard semantic version
- `v0.1.4` - Pre-1.0 versions
- `v2.1.0-beta` - Pre-release versions
- `v1.0.0-rc1` - Release candidates

❌ **Ignored Tags:**
- `release-1.0` - Wrong format
- `1.0.0` - Missing 'v' prefix
- `dev-tag` - Not version format

## Deployment Strategies

### Local Development

```bash
# Build single version for development
uv run --group docs sphinx-build -b html docs/source docs/build/html

# Build all versions for testing
uv run --group docs sphinx-multiversion docs/source docs/build/html

# Serve locally
python -m http.server 8000 -d docs/build/html
```

### Read the Docs

If using Read the Docs, add to `.readthedocs.yml`:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

sphinx:
  builder: html
  configuration: docs/source/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs

# Enable multiversion builds
formats: []
```

### GitHub Actions

Example workflow for automated versioned builds:

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for all versions
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Install dependencies
      run: uv sync --group docs
    
    - name: Build documentation
      run: uv run --group docs sphinx-multiversion docs/source docs/build/html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: docs/build/html
```

## Customization

### Version Display Names

You can customize how versions appear in the selector:

```python
# In conf.py - custom version formatting
def format_version(version):
    if version.startswith('v'):
        return version[1:]  # Remove 'v' prefix
    return version

smv_format_ref = format_version
```

### Version-Specific Content

Use the `version` variable in templates:

```html
<!-- In _templates/layout.html -->
{% if version == "main" %}
  <div class="dev-warning">
    ⚠️ This is development documentation
  </div>
{% endif %}
```

### Branch-Specific Configuration

Different settings for different branches:

```python
# In conf.py
import os

# Get current ref from environment
current_ref = os.environ.get('SPHINX_MULTIVERSION_NAME', 'main')

if current_ref == 'main':
    # Development version settings
    html_theme_options['announcement'] = "Development version - may be unstable"
elif current_ref.startswith('v'):
    # Release version settings
    html_theme_options['announcement'] = None
```

## Troubleshooting

### Common Issues

#### Missing Versions

**Problem**: Some tags/branches don't appear in build.

**Solution**: Check whitelist patterns:
```bash
# Test regex patterns
python -c "
import re
pattern = r'^v\d+\.\d+.*$'
test_tags = ['v1.0.0', 'v0.1.4', 'release-1.0']
for tag in test_tags:
    if re.match(pattern, tag):
        print(f'✓ {tag} matches')
    else:
        print(f'✗ {tag} ignored')
"
```

#### Build Failures

**Problem**: Some versions fail to build.

**Solutions**:
```bash
# Check git history
git log --oneline --all

# Verify dependencies across versions
git checkout v0.1.0
uv sync --group docs  # Check if dependencies work

# Test single version build
uv run --group docs sphinx-build -b html docs/source docs/build/html
```

#### Large Build Times

**Problem**: Building all versions is slow.

**Solutions**:
```python
# Limit to recent versions only
smv_tag_whitelist = r"^v0\.[5-9]\.\d+.*$|^v[1-9]\.\d+\.\d+.*$"

# Or limit branches
smv_branch_whitelist = r"^main$"
```

### Performance Optimization

```bash
# Build only changed versions (development)
uv run --group docs sphinx-multiversion docs/source docs/build/html --no-cleanup

# Parallel builds (if supported)
export SPHINX_MULTIVERSION_JOBS=4
uv run --group docs sphinx-multiversion docs/source docs/build/html
```

## Best Practices

### 1. Version Strategy

- **Tag consistently**: Use semantic versioning (`v1.2.3`)
- **Main branch**: Always buildable development docs
- **Stable branch**: Optional for LTS versions
- **Clean history**: Avoid force-pushing to tagged versions

### 2. Documentation Quality

- **Version compatibility**: Note breaking changes in docs
- **Migration guides**: Help users upgrade between versions
- **Deprecation warnings**: Document deprecated features
- **Version-specific examples**: Update examples for each version

### 3. Deployment

- **Automated builds**: Use CI/CD for consistent builds
- **Fast deploys**: Deploy only on tags/main branch changes
- **Rollback capability**: Keep previous builds available

### 4. User Experience

- **Clear version indicators**: Show which version user is viewing
- **Migration paths**: Link to upgrade guides
- **Version warnings**: Alert users about old/dev versions
- **Search scope**: Search within current version by default

## Integration with CI/CD

### Conditional Building

```yaml
# Only build multiversion on specific events
- name: Build single version docs
  if: github.event_name == 'pull_request'
  run: uv run --group docs sphinx-build -b html docs/source docs/build/html

- name: Build all versions
  if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/')
  run: uv run --group docs sphinx-multiversion docs/source docs/build/html
```

### Cache Optimization

```yaml
- name: Cache documentation builds
  uses: actions/cache@v3
  with:
    path: docs/build
    key: docs-${{ hashFiles('docs/**', 'src/**/*.py') }}
```

The versioned documentation system ensures users can always access relevant documentation for their installed version while providing a professional, maintainable documentation experience.