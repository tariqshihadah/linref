# Example Notebooks — Contributor Guide

This folder contains the example notebooks rendered by Sphinx as part of the
linref documentation.  Each notebook is a standalone tutorial focused on a
single topic.

## Adding a new example

1. Create a new `.ipynb` file in this folder using the next available number
   prefix (e.g., `07-new-topic.ipynb`).
2. Use `_setup.py` for shared imports and data loading:
   ```python
   import sys
   sys.path.insert(0, '.')
   from _setup import lr, roadways, crashes, pavement
   ```
3. Add the notebook stem name to `index.rst` in this folder.
4. Run the notebook end-to-end before committing so that cell outputs are
   embedded.  CI will also validate execution.

## Modifying shared setup

Edit `_setup.py`, then re-execute **all** notebooks so their outputs reflect
the change.

## Naming convention

`NN-short-topic.ipynb` — numbered prefix controls sort order; use kebab-case
for the topic slug.

## Self-contained examples

If a notebook is meant to be fully self-contained (like the HIN workflow),
include all setup inline instead of importing `_setup.py` and note this in
the notebook's introduction.
