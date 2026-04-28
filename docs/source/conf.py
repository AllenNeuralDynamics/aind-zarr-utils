"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path Setup --------------------------------------------------------------
import sys
from datetime import date
from os.path import abspath, dirname
from pathlib import Path

# Add the project src directory to sys.path for imports
project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, str(Path(project_root) / "src"))

from aind_zarr_utils import __version__ as package_version  # noqa:E402

INSTITUTE_NAME = "Allen Institute for Neural Dynamics"

current_year = date.today().year

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AIND Zarr Utils"
copyright = f"{current_year}, {INSTITUTE_NAME}"
author = INSTITUTE_NAME
release = package_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- MyST configuration -----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",  # ::: code blocks
    "deflist",  # Definition lists
    "html_admonition",  # Callout boxes
    "substitution",  # Variable substitution
    "tasklist",  # Task lists
]

# -- Autodoc configuration --------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Autosummary configuration ----------------------------------------------
autosummary_generate = True

# -- Napoleon configuration -------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "light_logo": "light-logo.svg",
    "dark_logo": "dark-logo.svg",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/AllenNeuralDynamics/aind-zarr-utils",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False
