"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

# -- Path Setup --------------------------------------------------------------
import sys
from os.path import abspath, dirname
from pathlib import Path

# Add the project root to sys.path for imports
sys.path.insert(0, abspath(dirname(dirname(dirname(__file__)))))

from aind_zarr_utils import __version__ as package_version

INSTITUTE_NAME = "Allen Institute for Neural Dynamics"

current_year = date.today().year

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = Path(dirname(dirname(dirname(abspath(__file__))))).name
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
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx_multiversion",
]
templates_path = ["_templates"]
exclude_patterns = []

# -- MyST configuration -----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",      # ::: code blocks
    "deflist",          # Definition lists
    "html_admonition",  # Callout boxes
    "substitution",     # Variable substitution
    "tasklist",         # Task lists
]

# -- Intersphinx configuration ----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "SimpleITK": ("https://simpleitk.readthedocs.io/en/master/", None),
    "ants": ("https://antspy.readthedocs.io/en/latest/", None),
}

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
    "source_repository": "https://github.com/AllenNeuralDynamics/aind-zarr-utils/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

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
