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

* **Key columns** (``key_col``) - One or more unique route identifier or grouping columns (e.g., 'Route', 'Direction')
* **Location columns** - For point events (``loc_col``) or linear events (``beg_col``, ``end_col``)
* **Geometry columns** - For spatial data (``geom_col``, ``geom_m_col``)
* **Closure type** (``closed``) - How range endpoints are handled: 'left', 'right', 'both', 'neither', or 'left_mod'/'right_mod'

**DataFrame Accessor Pattern**

Linref uses the ``.lr`` accessor to extend pandas DataFrames. Once you set an LRS on a DataFrame, all linear referencing methods become available::

    import linref as lr
    import pandas as pd
    
    # Create a DataFrame
    df = pd.DataFrame({
        'route': ['A', 'A', 'B'],
        'beg': [0.0, 1.0, 0.0],
        'end': [1.0, 2.0, 1.5],
        'speed_limit': [55, 45, 65]
    })
    
    # Set up the LRS
    df = df.lr.set_lrs(
        key_col=['route'],
        beg_col='beg',
        end_col='end',
        closed='right'
    )
    
    # Now you can use linear referencing methods
    dissolved = df.lr.dissolve()

**Setting a Default LRS**

For convenience, you can set a default LRS that will be automatically applied to all DataFrames::

    # Define your LRS once
    default_lrs = lr.LRS(
        key_col=['route'],
        beg_col='begin_mp',
        end_col='end_mp',
        geom_col='geometry',
        closed='left_mod'
    )
    
    # Set as default
    lr.set_default_lrs(default_lrs)
    
    # Now all DataFrames automatically have this LRS unless overridden

For detailed usage examples, common patterns, and comprehensive documentation, see `USAGE.md <USAGE.md>`_.

Documentation
=============

* **Usage Guide**: `USAGE.md <USAGE.md>`_ - Comprehensive examples and patterns
* **API Documentation**: https://linref.readthedocs.io/
* **Examples**: See the ``examples/`` folder for Jupyter notebooks
* **GitHub**: https://github.com/tariqshihadah/linref

Migration from v0.1.x
=====================

The ``redesign`` branch represents a major architectural overhaul of linref. Here are the key changes:

**API Changes**

* **EventsCollection** → **DataFrame with .lr accessor**
  
  * Old: ``ec = EventsCollection(df, keys=['Route'], beg='Begin', end='End')``
  * New: ``df = df.lr.set_lrs(key_col=['Route'], beg_col='Begin', end_col='End')``

* **EventsGroup** → **DataFrame.lr.get_group()**
  
  * Old: ``eg = ec['Route 50', 2018]``
  * New: ``group_df = df.lr.get_group(('Route 50', 2018))``

* **Dissolve operations**
  
  * Old: ``ec.dissolve(attr=['Speed_Limit'], aggs=['County'])``
  * New: ``df.lr.dissolve(retain=['Speed_Limit', 'County'])``

* **Merge and overlay**
  
  * Old: ``em = ec1.merge(ec2); result = em.overlay_average(...)``
  * New: ``result = df1.lr.overlay(df2, ...)``

**New Capabilities**

* Direct integration with pandas operations - no need to extract/wrap DataFrames
* More flexible LRS configuration and modification
* Improved performance through optimized EventsData core
* Better geometry handling with M-enabled geometry support
* Simplified API with pandas-like method chaining

**Core Concepts**

* **LRS objects** explicitly define your linear referencing schema
* **EventsData** provides the computational engine (typically accessed internally)
* **.lr accessor** gives you intuitive, DataFrame-native linear referencing methods
* **Set default LRS** to avoid repetitive configuration

For detailed migration assistance, see the documentation at https://linref.readthedocs.io/

For version history and release notes, see `CHANGELOG.md <CHANGELOG.md>`_.