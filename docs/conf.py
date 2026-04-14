# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'linref'
copyright = '2025, Tariq Shihadah'
author = 'Tariq Shihadah'

# Read version from pyproject.toml (single source of truth)
with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), 'rb') as f:
    _pyproject = tomllib.load(f)
release = _pyproject['project']['version']
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'nbsphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'examples/README.md']

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

# -- Napoleon (NumPy / Google style docstrings) ------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx (cross-project links) --------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
}

# -- nbsphinx (notebook rendering) ------------------------------------------

nbsphinx_execute = 'never'  # Render pre-executed outputs; CI validates separately

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'source/Linref_Logo_Simple_AllWhite.svg'
html_theme_options = {
    'logo_only': True,
}
