# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# from sphinx.ext.napoleon.docstring import GoogleDocstring
# def parse_keys_section(self, section):
#     return self._format_fields('Keys', self._consume_fields())
# GoogleDocstring._parse_keys_section = parse_keys_section
#  
# def parse_attributes_section(self, section):
#     return self._format_fields('Attributes', self._consume_fields())
# GoogleDocstring._parse_attributes_section = parse_attributes_section
# 
# def parse_class_attributes_section(self, section):
#     return self._format_fields('Class Attributes', self._consume_fields())
# GoogleDocstring._parse_class_attributes_section = parse_attributes_section
# 
# # we now patch the parse method to guarantee that the the above methods are
# # assigned to the _section dict
# def patched_parse(self):
#     self._sections['keys'] = self._parse_keys_section
#     self._sections['class attributes'] = self._parse_class_attributes_section
#     self._unpatched_parse()
# GoogleDocstring._unpatched_parse = GoogleDocstring._parse
# GoogleDocstring._parse = patched_parse


# -- Project information -----------------------------------------------------

project = 'dataset'
copyright = '2019, Shuo Han'
author = 'Shuo Han'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.intersphinx',
        'sphinx.ext.napoleon',
        'sphinx.ext.autodoc',
        'sphinx.ext.mathjax',
        'sphinx.ext.ifconfig',
        'sphinx.ext.todo'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# import sphinx_bootstrap_theme
# html_theme = 'bootstrap'
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

napoleon_use_rtype = True
napoleon_use_ivar = False
autodoc_mock_imports = ['numpy', 'scipy', 'nibabel', 'image_processing_3d']
autodoc_member_order = 'bysource'
