# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib
import os
import sys

import sphinx.builders.html
import sphinx.builders.latex
import sphinx.builders.texinfo
import sphinx.builders.text
import sphinx.ext.autodoc


sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'MEDimage'
copyright = '2022, MEDimage'
author = 'MEDimage'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.ifconfig',
    'sphinx_rtd_dark_mode',
    'sphinx-jsonschema',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

#Todo settings
todo_include_todos=True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# user starts in light mode
default_dark_mode = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['css/custom.css']

# assing master document
master_doc = 'index'

def setup(app):
    app.add_css_file('custom.css')

# PATCH `sphinx-jsonschema`
#  to render the extra `options`` and ``tags`` schema properties
#
def _patched_sphinx_jsonschema_simpletype(self, schema):
    """Render the *extra* ``required`` and ``options`` schema properties for every object."""
    rows = _original_sphinx_jsonschema_simpletype(self, schema)

    if "required" in schema:
        required = schema["required"]
        if required not in ["true", "false"]:
            raise Exception("The required argument must be one of true, false")
        rows.append(self._line(self._cell("required"), self._cell(required)))
        del schema["required"]

    if "range" in schema:
        range = schema["range"]
        rows.append(self._line(self._cell("range"), self._cell(range)))
        del schema["range"]

    # if "options" in schema:
    #     rows.append(self._line(self._cell("options"), self._cell("")))
    #     for option in schema["options"]:
    #         rows.append(self._line(self._cell(""), self._cell(f"``{option}``"), self._cell("test")))
    #
    #     del schema["options"]

    if "options" in schema:
        key = "options"
        rows.append(self._line(self._cell(key)))

        for prop in schema[key].keys():
            # insert spaces around the regexp OR operator
            # allowing the regexp to be split over multiple lines.
            # proplist = prop.split('|')
            # dispprop = self._escape(' | '.join(proplist))
            dispprop = prop
            bold = '``'
            label = self._cell(bold + dispprop + bold)

            if isinstance(schema[key][prop], dict):
                obj = schema[key][prop]
                rows.extend(self._dispatch(obj, label)[0])
            else:
                rows.append(self._line(label, self._cell(schema[key][prop])))
        del schema[key]

    return rows


sjs_wide_format = importlib.import_module("sphinx-jsonschema.wide_format")
_original_sphinx_jsonschema_simpletype = sjs_wide_format.WideFormat._simpletype  # type: ignore
sjs_wide_format.WideFormat._simpletype = _patched_sphinx_jsonschema_simpletype  # type: ignore
