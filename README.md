# aind-zarr-utils

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

A collection of utilities for working with ZARR files and AIND metadata. This
can be used to generate SimpleITK or ANTS images from a ZARR dataset, or
generate a stub image to do index conversion.

## Usage
aind_zarr_utils provides an anatomically aware platform for interacting with AIND Zarr images and their corresponding metadata.

To load an image, first read the metadata:
```
from aind_zarr_utils.json_utils import get_json
metadata_path = 'path_to_metadata'
metadata = {'acquisition':get_json(metadata_path)}
```
you can then use the "zarr_to_ants" or "zarr_to_sitk" to load the zarr as an ANTs image or SITK image, respectively. i.e.:
```
from aind_zarr_utils.zarr import zarr_to_ants,zarr_to_sitk
channel_path = 'path_to_image'
level = 3 # Zarr level to read
ants_image = zarr_to_ants(channel_path, metadata, level=level)
sitk_image = zarr_to_sitk(channel_path, metadata, level=level)
```

In some cases, it is useful to interact with point data in image space, without explicitly loading the underlying image. In this case, aind_zarr_utils allows for the use of a "stub" image. Here, an empty sitk image will be loaded. This image will have origin, direction, and spacing that match the full image, but takes up minimal space in memory. Once a stub image is loaded, regular SITK functions for moving from, for example, indices to physical space can be used.

```
from aind_zarr_utils.zarr import zarr_to_sitk_stub
sitk_stub = zarr_to_sitk_stub(channel_path,metadata,level
```

Stub images are particularly useful when dealing with points annotated in image space. aind_zarr_utils includes functionality for reading neuroglancer annotations. Since neuroglancer itself is not anatomically aware, this operation also depends on knowledge of the metadata. i.e.:
```
neuroglancer_json = get_json('path_to_neuroglancer.json')
points_neuroglancer_indicies,_ = neuroglancer_annotations_to_indices(neuroglancer_json)
points_image_anatomical = annotation_indices_to_anatomical(stub,points_neuroglancer_indicies)
```




## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
./scripts/run_linters_and_checks.sh -c
```

which is equivalent to:
```bash
uv run --frozen ruff format
uv run --frozen ruff check
uv run --frozen interrogate -v
uv run --frozen codespell --check-filenames
uv run --frozen pytest --cov aind_zarr_utils
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o docs/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html docs/source/ docs/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

### Read the Docs Deployment
Note: Private repositories require **Read the Docs for Business** account. The following instructions are for a public repo.

The following are required to import and build documentations on *Read the Docs*:
- A *Read the Docs* user account connected to Github. See [here](https://docs.readthedocs.com/platform/stable/guides/connecting-git-account.html) for more details.
- *Read the Docs* needs elevated permissions to perform certain operations that ensure that the workflow is as smooth as possible, like installing webhooks. If you are not the owner of the repo, you may have to request elevated permissions from the owner/admin. 
- A **.readthedocs.yaml** file in the root directory of the repo. Here is a basic template:
```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

# Required
version: 2

# Set the OS, Python version, and other tools you might need
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"

# Path to a Sphinx configuration file.
sphinx:
  configuration: docs/source/conf.py

# Declare the Python requirements required to build your documentation
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
```

Here are the steps for building docs in *Read the Docs*. See [here](https://docs.readthedocs.com/platform/stable/intro/add-project.html) for detailed instructions:
- From *Read the Docs* dashboard, click on **Add project**.
- For automatic configuration, select **Configure automatically** and type the name of the repo. A repo with public visibility should appear as you type. 
- Follow the subsequent steps.
- For manual configuration, select **Configure manually** and follow the subsequent steps

Once a project is created successfully, you will be able to configure/modify the project's settings; such as **Default version**, **Default branch** etc.
