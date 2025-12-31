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

Code Snippets
=============

Setting up an LRS
-----------------
Create a DataFrame with linearly referenced events and configure its LRS::

    import pandas as pd
    import linref as lr
    
    # Sample data with route ID and mile markers
    df = pd.DataFrame({
        'route': ['Route 50', 'Route 50', 'Route 66', 'Route 66'],
        'year': [2018, 2018, 2018, 2020],
        'beg': [0.0, 1.5, 0.0, 0.5],
        'end': [1.5, 3.0, 2.0, 2.5],
        'speed_limit': [55, 45, 65, 65],
        'volume': [1000, 1200, 800, 850]
    })
    
    # Configure the LRS
    df = df.lr.set_lrs(
        key_col=['route', 'year'],
        beg_col='beg',
        end_col='end',
        closed='right'
    )

Selecting Events by Group
--------------------------
Select events from specific groups using ``get_group()``::

    # Get all events for Route 50 in 2018
    route_50_2018 = df.lr.get_group(('Route 50', 2018))
    
    # Iterate over all groups
    for group, group_df in df.lr.iter_groups():
        print(f"Processing {group}: {len(group_df)} events")

Dissolving Events
-----------------
Merge consecutive events with the same keys and optional other attributes::

    # Dissolve events, keeping speed_limit and volume columns
    dissolved = df.lr.dissolve(retain=['speed_limit'])

Conflating Datasets
-------------------
Create relationships between two linearly referenced datasets::

    # Two datasets with the same LRS
    roads = roads_df.lr.set_lrs(
        key_col=['route'],
        beg_col='beg',
        end_col='end'
    )
    
    crashes = crashes_df.lr.set_lrs(
        key_col=['route'],
        beg_col='crash_beg',
        end_col='crash_end'
    )
    
    # Overlay crashes onto roads
    relation = roads.lr.relate(crashes)

    # Aggregate crashes by road segment
    roads['crash_count'] = relation.count()

Resegmentation
--------------
Create uniform segments from variable-length events::

    # Create 0.1-mile segments
    dissolved = df.lr.dissolve()
    segmented = dissolved.lr.resegment(length=0.1)

Working with Geometries
-----------------------
Generate geometries from mile markers or extract mile markers from geometries::

    # Generate geometries from mile markers
    # First, create a reference GeoDataFrame with route geometries
    import geopandas as gpd
    from shapely.geometry import LineString
    
    ref_routes = gpd.GeoDataFrame({
        'route': ['Route 50', 'Route 66'],
        'geometry': [
            LineString([(0, 0), (5, 0)]),
            LineString([(0, 2), (5, 2)])
        ]
    }).lr.set_lrs(
        key_col=['route'],
        geom_col='geometry'
    )
    
    # Generate M-enabled geometries with begin/end mile markers
    ref_routes = ref_routes.lr.generate_linear_events()
    
    # Cut geometries for your events from the reference routes
    result = df.lr.cut_from(ref_routes)

Point Events
------------
Work with point-based linearly referenced data::

    # Point event data
    signs = pd.DataFrame({
        'route': ['A', 'A', 'B'],
        'milepost': [0.5, 1.2, 0.8],
        'sign_type': ['Speed Limit', 'Stop', 'Yield']
    }).lr.set_lrs(
        key_col=['route'],
        loc_col='milepost'
    )
    
    # Convert to linear events (useful for buffering)
    signs_buffered = signs.lr.extend(
        extend_begs=0.1,
        extend_ends=0.1
    )

Common Patterns
===============

Pattern 1: Dissolving by Attributes
------------------------------------
Merge consecutive events that share the same attribute values::

    # Dissolve on speed_limit, keeping track of original events
    dissolved = df.lr.dissolve(
        retain=['speed_limit', 'pavement_type'],
        inverse_index=True,
        inverse_col='original_ids'
    )
    
    # The result contains merged events with 'original_ids' 
    # showing which events were combined

Pattern 2: Weighted Overlay Analysis
-------------------------------------
Compute linearly-weighted statistics from overlaying datasets::

    # Create relationship between roads and crashes
    relation = roads.lr.relate(crashes)
    
    # Count crashes per road segment
    roads['crash_count'] = relation.count()
    
    # Get crash severity data
    crash_severity = crashes['severity'].values
    
    # Calculate mean severity using first matching crash
    roads['avg_severity'] = relation.first(data=crash_severity)
    
    # Or sum total severity across all matching crashes
    roads['total_severity'] = relation.sum(data=crash_severity)

Pattern 3: Creating Reference Geometries
-----------------------------------------
Generate mile-marker-enabled geometries from existing linework::

    # Start with a GeoDataFrame of routes
    routes_gdf = gpd.read_file('routes.geojson')
    
    # Set up basic LRS
    routes = routes_gdf.lr.set_lrs(
        key_col=['route_id'],
        geom_col='geometry'
    )
    
    # Generate linear events with mile markers and chains
    routes = routes.lr.generate_linear_events(
        beg_col='beg_mp',
        end_col='end_mp',
        chain_col='chain',
        scale=1/5280,  # Convert feet to miles
        decimals=3,
        add_geom_m=True
    )
    
    # Now routes has begin/end mile markers and M-enabled geometries

Pattern 4: Integrating Multiple Event Datasets
-----------------------------------------------
Combine multiple datasets into a unified linear reference framework::

    # Multiple event datasets with the same LRS configuration
    pavement_df = pavement_df.lr.set_lrs(
        key_col=['route'],
        beg_col='beg',
        end_col='end'
    )
    traffic_df = traffic_df.lr.set_lrs(
        key_col=['route'],
        beg_col='beg',
        end_col='end'
    )
    crash_df = crash_df.lr.set_lrs(
        key_col=['route'],
        beg_col='beg',
        end_col='end'
    )
    
    # Integrate all three datasets
    integrated_df = pavement_df.lr.integrate(
        [traffic_df, crash_df],
        fill_gaps=False
    )

Pattern 5: Extracting M-Values from Geometries
-----------------------------------------------
Extract mile markers from M-enabled geometries::

    # DataFrame with M-enabled geometries
    df = df.lr.set_lrs(
        key_col=['route'],
        geom_col='geometry',
        geom_m_col='geometry_m'
    )
    
    # Extract begin and end M-values
    df = df.lr.extract_m_values(
        beg_col='extracted_beg',
        end_col='extracted_end'
    )

Pattern 6: Buffering Point Events
----------------------------------
Convert point events to linear events with buffers::

    # Point events
    signs = signs_df.lr.set_lrs(
        key_col=['route'],
        loc_col='milepost'
    )
    
    # Extend to create buffer zones
    sign_zones = signs.lr.extend(
        extend_begs=0.25,  # 0.25 miles before
        extend_ends=0.25   # 0.25 miles after
    )

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

Version Notes
=============

1.0.0 (redesign branch) - Major Architectural Overhaul
-------------------------------------------------------
**Breaking Changes:**

* Complete API redesign using pandas accessor pattern
* EventsCollection, EventsGroup, EventsMerge, EventsUnion classes replaced with DataFrame.lr accessor methods
* New LRS (Linear Referencing System) configuration objects for consistent schema definition following geopandas CRS pattern
* EventsData serves as internal computational core instead of user-facing API

**New Features:**

* Direct DataFrame integration through .lr accessor
* Flexible LRS configuration and modification
* Default LRS settings for convenience
* Redesign of M-enabled geometry handling
* Enhanced performance through optimized EventsData implementation
* Simplified, pandas-like method chaining API

**Note:** This is a complete redesign. Users of v0.1.x will need to update their code. See "Migration from v0.1.x" section above.

0.1.2 (2025-03-18)
------------------
* Allow for named index on dataframes being analyzed by `lr.generate_linear_events` by using dynamic indices during spatial join.
* Performance improvements
* Various bug fixes, minor features

0.1.1 (2024-08-20)
-------------------
* Addition of array-like, dataframe label-based, and callable ``length`` parameter options to the ``EventsCollection.to_windows()`` method and ``distance`` parameter to the ``EventsCollection.shift()`` method. This allows for the creation of segments based on a variable length. This will be made available to other methods in future versions as well.
* Address ``numpy>=2.x`` compatibility issue in ``EventsCollection.dissolve()`` method which was causing errors in some cases. More fixes related to ``numpy`` updates are expected to come in future versions.
* Address datatype loss issues for LRS columns during ``EventsCollection.to_windows()`` and ``EventsUnion.union()`` methods. This was causing some instances of key columns to be converted to incorrect dtypes.
* Restructure .targets property to be ``keys + [beg, end]`` for more intuitive and consistent performance.
* Addition of ``min`` and ``max`` aggregation methods to the ``EventsMergeAttribute`` class.
* Several example Jupyter Notebooks for common use cases have been added to the ``examples`` folder for user reference. More to come.
* Performance improvements
* Various bug fixes, minor features

0.1.0 (2024-01-16)
-------------------
My heart is in Gaza.

* Initial deployment of synthesis module featuring some tools for generating linear referencing information for chains of linear asset data with geometry but no LRS. These features are currently experimental and outputs should be reviewed for quality and expected outcomes. They will get refined in future versions based on performance in various applications and input from users.
* Addition of ``retain`` parameter to the ``EventsCollection.to_windows()`` method which retains all non-target and non-spatial fields from the original events dataframe when performing the operation. Previously, only newly generated target fields would be present in the output ``EventsCollection``'s events dataframe.
* Fixed implementation of ``EventsCollection.spatials`` property to correctly return a list of spatial column labels (e.g., geometry and route column labels) in the events dataframe. This also corrects ``EventsCollection.others`` which previously incorrectly included these labels.
* Addition of ``EventsCollection.clip()`` method and expansion of the ``EventsCollection.shift()`` method for better parameterization.
* Transition to ``pyproject.toml`` setup with ``setuptools``.
* Performance improvements
* Various bug fixes, minor features
* Why not push to v0.1 finally?

0.0.10 (2023-05-03)
-------------------
Not a lot of updates to share, I guess that's a good thing?

* Minor updates to MLSRoute class to account for deprecation of subscripting MultiLineStrings. Most issues were addressed previously but a few were missed, most notably in the MLSRoute.bearing() method and a couple odd cases in the MLSRoute.cut() method.
* Fix minor issue with EventsCollection.project_parallel() implementation related to unmatched sampling points.
* Addition of EventsFrame.cast_gdf() method to cast events dataframes to geopandas geodataframes in-line.
* Performance improvements
* Various bug fixes, minor features

0.0.9 (2023-03-02)
------------------
First update of 2023. Been a quiet start to the year.

* Add missing .any() aggregation method to EventsMergeAttribute API. Was previously available but missed during a previous update.
* Update documentation
* Performance improvements
* Various bug fixes, minor features

0.0.8.post2 (2022-12-23)
------------------------
Final update of 2022 including small feature updates and bug fixes from 0.0.8. Happy Holidays!

* Add .set_df() method for in-line modification of an EventsFrame's dataframe, inplace or not.
* Addition of suffixes parameter and default setting to EventsUnion.union() method.
* Performance improvements
* Various bug fixes, minor features

0.0.8.post1 (2022-12-16)
------------------------
* Improve performance of .project() method when nearest=False by removing dependence on join_nearby() function and using native gpd features.
* Add .size and .shape properties to EventsFrames and subclasses.
* Various bug fixes, minor features

0.0.8 (2022-12-14)
------------------
* Improve performance of essential .get_group() method, reducing superfluous initialization of empty dataframes and events collections and improving logging of initialized groups.
* Improve performance of .union() method with updated RangeCollection.union() features and optimized iteration and aggregation of unified data. Performance times are significantly improved, especially for large datasets with many events groups.
* Improve distribute method performance which was added in recent versions.
* Drop duplicates in .project() method when using sjoin_nearest with newer versions of geopandas. Improved validation in .project() method, address edge case where projecting geometry column has a non-standard label (e.g., not 'geometry').
* Added .sort() method to events collection. Default sorting methods remain unchanged.
* Added warnings for missing data in target columns when initializing an EventsFrames through standard methods.
* Remove .project_old() method from events collection due to deprecation.
* Performance improvements
* Various bug fixes, minor features

0.0.7 (2022-10-14)
------------------
* Refactoring of EventsMerge system from 2D to 3D vectorized relationships for improved performance and accuracy. API and aggregation methods are largely the same.
* Modified closed parameter use in merge relationships in accordance with rangel v0.0.6, which now performs intersections which honor the closed parameter on the left collection as well as the right collection. This provides more accurate results for events which fall on the edges of intersecting events when using left_mod or right_mod closed parameters.
* Updates to account for rangel 0.0.6 version which is now a minimum version requirement. Added other minimum version requirements for related packages.
* Performance improvements
* Various bug fixes, minor features

0.0.5.post1 (2022-09-06)
------------------------
* Address deprecation of length of and iteration over multi-part geometries in shapely
* Remove code redundancies in linref.events.collection for get_most and get_mode

0.0.5 (2022-09-01)
------------------
* Added sumproduct and count aggregators to EventsMergeAttribute class
* Address deprecation of length of and iteration over multi-part geometries in shapely
* Performance improvements
* Various bug fixes, minor features

0.0.4 (2022-06-24)
------------------
* Minor feature additions
* Performance improvements
* Addition of logos in github repo
* Various bug fixes, minor features

0.0.3 (2022-06-07)
------------------
* Various updates for geopandas 0.10+ dependency including improved performance of project methods
* Automatic sorting of events dataframe prior to performing dissolve
* Performance improvements
* Various bug fixes, minor features

0.0.2 (2022-04-11)
------------------
* Various bug fixes, minor features

0.0.1 (2022-03-31)
------------------
* Original experimental release.

Future improvements
===================
* Check spatial continuity of events groups. This may apply at instantiation of ``EventsCollection``.
* Unify direction of opposing routes which converge/diverge at a point (e.g., W Main St and E Main St).