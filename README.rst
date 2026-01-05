Overview
========
The ``linref`` library builds on tabular and geospatial libraries ``pandas`` and ``geopandas`` to provide powerful features for linearly referenced data management, manipulation, and analysis. Using a modern pandas accessor pattern (``.lr``), linref seamlessly extends DataFrames with linear referencing capabilities while maintaining full compatibility with existing pandas/geopandas workflows.

At its core, linref uses:

* **LRS (Linear Referencing System)** objects to define the schema of your linearly referenced data
* **EventsData** as the underlying computational engine for efficient linear operations
* **DataFrame accessor pattern** (``.lr``) for intuitive, pandas-like syntax
* Optimized implementations powered by ``numpy``, ``shapely``, and ``scipy``

Some of the main features of this library include:

* **Event data engineering** - Merge consecutive events with ``df.lr.dissolve()``, create uniform segments with ``df.lr.resegment()``, and project point and linear data onto linearly referenced data
* **Data conflation operations** - Create relationships between datasets with ``df.lr.relate()`` and aggregate data by attributes, counts, and more using count and length-weighted methods
* **Geometry operations** - Generate geometries from mile markers, extract mile markers from geometries, and perform spatial linear referencing operations
* **Integration** - Combine multiple event datasets into unified linearly referenced frameworks
* **Prepare and run in-depth analyses** - Built-in support for advanced analyses such as high-injury networks, intersection influence areas, and more through simple, well-documented methods

Getting Started
===============

Installation
------------
Install linref using pip::

    pip install linref

Basic Concepts
--------------
**Linear Referencing System (LRS)**

An LRS defines the schema of your linearly referenced data by specifying which columns represent:

* **Key columns** ``key_col`` - One or more unique route identifier or grouping columns (e.g., 'Route', 'County')
* **Location columns** - For point events use ``loc_col`` (e.g., 'Milepost'), for linear events ``beg_col`` and ``end_col`` (e.g., 'Begin_Milepost', 'End_Milepost')
* **Geometry columns** - For spatial data ``geom_col`` and for spatial data that is m-enabled ``geom_m_col`` (generally m-enabled geometries are prepared and managed by ``linref``)
* **Closure type** ``closed`` - How range endpoints are handled: 'left', 'right', 'both', 'neither', or 'left_mod'/'right_mod'

Documentation
=============

* **Usage Guide**: `USAGE.md <USAGE.md>`_ - Comprehensive examples and patterns
* **API Documentation**: https://linref.readthedocs.io/
* **Examples**: See the ``examples/`` folder for Jupyter notebooks
* **GitHub**: https://github.com/tariqshihadah/linref
* **Release Notes**: `CHANGELOG.md <CHANGELOG.md>`_ - Detailed version history