Migration Guide: v0.1.x → v1.0
================================

Linref v1.0 is a complete redesign of the library. The standalone class-based
API (``EventsCollection``, ``EventsMerge``, ``EventsUnion``) has been replaced
with a pandas accessor pattern (``.lr``). This guide provides side-by-side
comparisons to help you transition existing code.

.. contents:: On this page
   :local:
   :depth: 2

Key Architectural Changes
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - v0.1.x
     - v1.0
   * - Entry point
     - Wrap DataFrames in ``EventsCollection``
     - Use ``.lr`` accessor directly on DataFrames
   * - LRS config
     - Embedded in each ``EventsCollection`` constructor
     - Separate ``LRS`` object, set once or as a default
   * - Return types
     - New ``EventsCollection`` instances
     - Plain DataFrames (with LRS metadata preserved)
   * - Relational ops
     - ``EventsMerge`` with attribute aggregators
     - ``df.lr.relate()`` returning an ``EventsRelation``
   * - Unions
     - ``EventsUnion([ec1, ec2]).union()``
     - ``lr.integrate([df1, df2])``
   * - Geometry
     - ``ec.build_routes()`` + merge-based interpolation
     - ``df.lr.add_geom_m()``, ``df.lr.project()``, relation ``.interpolate()`` / ``.cut()``
   * - Data access
     - ``ec.df`` to get the underlying DataFrame
     - The object *is* a DataFrame — no unwrapping needed

Import Changes
--------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - v0.1.x
     - v1.0
   * - ``import linref as lr``
     - ``import linref as lr`` *(unchanged)*
   * - ``from linref import EventsCollection``
     - Not needed — use ``df.lr`` accessor
   * - ``from linref import EventsUnion``
     - Not needed — use ``lr.integrate()``

Setup & LRS Configuration
--------------------------

**v0.1.x** — LRS columns were passed to the ``EventsCollection`` constructor
for every dataset:

.. code-block:: python

   # v0.1.x
   import linref as lr

   ec = lr.EventsCollection(
       df,
       keys=['route_id'],
       beg='beg',
       end='end',
       geom='geometry'
   )

   # Point events
   point_ec = lr.EventsCollection(
       point_df,
       keys=['route_id'],
       beg='beg'
   )

**v1.0** — Set an LRS once per DataFrame, or define a project-wide default:

.. code-block:: python

   # v1.0 — per-DataFrame
   import linref as lr

   df = df.lr.set_lrs(
       key_col=['route_id'],
       chain_col='chain',
       beg_col='beg',
       end_col='end',
       geom_col='geometry',
       closed='left_mod'
   )

   # Point events
   point_df = point_df.lr.set_lrs(
       key_col=['route_id'],
       loc_col='beg',
       closed='left_mod'
   )

   # Or set a default LRS for all DataFrames at once
   lr.set_default_lrs(lr.LRS(
       key_col=['route_id'],
       chain_col='chain',
       beg_col='beg',
       end_col='end',
       loc_col='loc',
       geom_col='geometry',
       closed='left_mod'
   ))

.. tip::

   With a default LRS set, any DataFrame with matching column names will
   automatically use the correct schema — no per-object setup needed.

Dissolving Events
-----------------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ec = lr.EventsCollection(df, keys=['route_id'], beg='beg', end='end')
   ec_dissolved = ec.dissolve(attr=['speed_limit'], aggs=['county'])

   # Access the result
   result_df = ec_dissolved.df

**v1.0:**

.. code-block:: python

   # v1.0
   dissolved = df.lr.dissolve(retain=['speed_limit'])

   # Result IS a DataFrame — no .df unwrapping
   print(dissolved[['route_id', 'beg', 'end', 'speed_limit']])

   # Geometries are merged automatically (with M-enabled geometry created)
   print(dissolved.iloc[0].geometry_m)

Resegmentation (to_windows → resegment)
-----------------------------------------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ec_dissolved = ec.dissolve()
   ec_segments = ec_dissolved.to_windows(length=5)
   result_df = ec_segments.df

**v1.0:**

.. code-block:: python

   # v1.0
   dissolved = df.lr.dissolve()
   segmented = dissolved.lr.resegment(length=5)

Merging / Relating Events
--------------------------

The ``EventsMerge`` class and its aggregators have been replaced by
``df.lr.relate()`` which returns an ``EventsRelation`` with a similar but
modernized aggregation API.

**v0.1.x — Point-on-linear merge:**

.. code-block:: python

   # v0.1.x
   roads_ec = lr.EventsCollection(roads_df, keys=['route_id'], beg='beg', end='end')
   crash_ec = lr.EventsCollection(crashes_df, keys=['route_id'], beg='loc')

   em = roads_ec.merge(crash_ec)
   crash_count = em.count()       # EventsMergeAttribute aggregator
   avg_speed   = em['speed'].mean()

**v1.0 — Same operation:**

.. code-block:: python

   # v1.0
   relation = roads_df.lr.relate(crashes_df)

   # DataFrame-level aggregation
   roads_df['crash_count'] = relation.count()

   # Column-level aggregation (groupby-like syntax)
   roads_df['avg_severity'] = relation['severity'].mean()
   roads_df['crash_ids']    = relation['crash_id'].list()

   # Multiple columns at once
   roads_df[['ids', 'severities']] = relation[['crash_id', 'severity']].list()

   # Value counts across categories
   categories = ['Fatal', 'Injury', 'Property Damage Only']
   roads_df[categories] = relation['severity'].value_counts()[categories]

**v0.1.x — Linear-on-linear merge:**

.. code-block:: python

   # v0.1.x
   road_ec = lr.EventsCollection(roads_df, keys=['route_id'], beg='beg', end='end')
   pave_ec = lr.EventsCollection(pavement_df, keys=['route_id'], beg='beg', end='end')

   em = road_ec.merge(pave_ec)
   weighted_avg = em['condition'].mean()  # length-weighted

**v1.0:**

.. code-block:: python

   # v1.0
   relation = roads_df.lr.relate(pavement_df)

   roads_df['condition'] = relation['condition_rating'].mean()   # length-weighted
   roads_df['surface']   = relation['surface_type'].mode()       # length-weighted mode

Union / Integration
-------------------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ec1 = lr.EventsCollection(ref_df, keys=['route_id'], beg='beg', end='end')
   ec2 = lr.EventsCollection(input_df, keys=['route_id'], beg='beg', end='end')

   eu = lr.EventsUnion([ec1, ec2])
   union_ec = eu.union()
   union_df = union_ec.df

   # Manual pandas merge to carry over attributes
   result = (union_df
       .merge(ref_df[['val']], left_on='index_0', right_index=True, how='left')
       .merge(input_df[['categ']], left_on='index_1', right_index=True, how='left')
   )

**v1.0:**

.. code-block:: python

   # v1.0
   integrated = lr.integrate([ref_df, input_df])

   # Index columns (integrated_index_0, integrated_index_1) link back to sources
   # Attributes can be joined the same way, or use relate() for aggregation
   print(integrated[['route_id', 'beg', 'end', 'integrated_index_0', 'integrated_index_1']])

Generating Point Geometries from Mileposts
-------------------------------------------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ref_ec = lr.EventsCollection(ref_gdf, keys=['route_id'], beg='beg', end='end', geom='geometry')
   point_ec = lr.EventsCollection(point_df, keys=['route_id'], beg='beg')

   ref_ec.build_routes()

   em = point_ec.merge(ref_ec)
   new_geoms = em.interpolate()

   point_gdf = gpd.GeoDataFrame(point_df, geometry=new_geoms)

**v1.0:**

.. code-block:: python

   # v1.0
   # Ensure reference has M-enabled geometries
   ref_df = ref_df.lr.add_geom_m()

   # Interpolate point geometries from the LRS relationship
   crashes_df['geometry'] = crashes_df.lr.relate(ref_df).interpolate()
   # Or use the convenience function interpolate_from
    crashes_df['geometry'] = crashes_df.lr.interpolate_from(ref_df)

Projecting Points onto a Network (Spatial)
-------------------------------------------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ref_ec = lr.EventsCollection(ref_gdf, keys=['route_id'], beg='beg', end='end', geom='geometry')
   proj_ec = ref_ec.project(point_gdf, nearest=True)

   result_df = proj_ec.df

**v1.0:**

.. code-block:: python

   # v1.0
   ref_df = ref_df.lr.add_geom_m()
   projected = ref_df.lr.project(point_gdf, buffer=2, nearest=True, dropna=True)

   print(projected[['route_id', 'loc', 'project_distance']])

Projecting Linear Geometries onto a Network (Spatial Parallel Projection)
-------------------------------------------------------------------------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ref_ec = lr.EventsCollection(ref_gdf, keys=['route_id'], beg='beg', end='end', geom='geometry')
   proj_ec = ref_ec.project_parallel(line_gdf)

   line_gdf[['route_id', 'beg', 'end']] = proj_ec.df[['route_id', 'beg', 'end']]

**v1.0:**

.. code-block:: python

   # v1.0
   # This utilizes an entirely new algorithm based on parallel Hausdorff distance,
   # which is more robust to complex geometries and can handle cases where the 
   # projected line deviates from the reference.
   from linref.ext.spatial import parallel_project_hausdorff

   ref_df = ref_df.lr.add_geom_m()

   projected = parallel_project_hausdorff(
       target=ref_df,
       projected=line_gdf,
       buffer=2,
       max_distance=1,
       match=1,
       densify=0.1
   )

   print(projected[['route_id', 'beg', 'end']])

Sorting
-------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ec.sort()

**v1.0:**

.. code-block:: python

   # v1.0
   df = df.lr.sort_standard()

Accessing Group Data
--------------------

**v0.1.x:**

.. code-block:: python

   # v0.1.x
   ec = lr.EventsCollection(df, keys=['route_id', 'year'], beg='beg', end='end')

   # Single group
   eg = ec['Route 50', 2018]

   # Partial key
   ec_2018 = ec[:, 2018]

   # Intersection query
   intersecting = eg.intersecting(0.5, 1.5, closed='left_mod')

**v1.0:**

.. code-block:: python

   # v1.0
   # Single group
   group_df = df.lr.get_group('Route 50')

   # Iterate over groups
   for name, group_df in df.lr.iter_groups():
       print(name, len(group_df))

   # Group counts
   print(df.lr.group_counts())

Validation
----------

**v0.1.x** — Warnings at construction time for missing data.

**v1.0:**

.. code-block:: python

   # v1.0
   # Check for invalid events
   print(df.lr.valid_events.sum(), "valid events")
   print(df.lr.invalid_events.sum(), "invalid events")

   # Drop invalid events
   df = df.lr.drop_invalid_events()

Quick Reference Table
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - v0.1.x
     - v1.0 Equivalent
   * - ``lr.EventsCollection(df, keys=..., beg=..., end=...)``
     - ``df.lr.set_lrs(key_col=..., beg_col=..., end_col=...)``
   * - ``ec.df``
     - *(just use the DataFrame directly)*
   * - ``ec.dissolve(attr=..., aggs=...)``
     - ``df.lr.dissolve(retain=...)``
   * - ``ec.to_windows(length=...)``
     - ``df.lr.resegment(length=...)``
   * - ``ec.merge(other_ec)``
     - ``df.lr.relate(other_df)``
   * - ``em.count()``
     - ``relation.count()``
   * - ``em['col'].mean()``
     - ``relation['col'].mean()``
   * - ``em.interpolate()``
     - ``relation.interpolate()``
   * - ``lr.EventsUnion([ec1, ec2]).union()``
     - ``lr.integrate([df1, df2])``
   * - ``ec.build_routes()``
     - ``df.lr.add_geom_m()``
   * - ``ec.project(gdf, nearest=True)``
     - ``df.lr.project(gdf, buffer=..., nearest=True)``
   * - ``ec.project_parallel(gdf)``
     - ``parallel_project_hausdorff(target=df, projected=gdf, ...)``
   * - ``ec.sort()``
     - ``df.lr.sort_standard()``
   * - ``ec['Route 50', 2018]``
     - ``df.lr.get_group('Route 50')``
   * - ``ec.intersecting(0.5, 1.5)``
     - ``df.lr.relate(query_df).intersect()``
