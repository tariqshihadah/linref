Overview
========
The ``linref`` library builds on tabular and geospatial libraries ``pandas`` and ``geopandas`` to implement powerful features for linearly referenced data through ``EventsCollection`` and other object classes. Linear referencing operations powered by the ``numpy``, ``shapely``, and ``rangel`` open-source libraries allow for optimized implementations of common and advanced linearly referenced data management, manipulation, and analysis operations.

Some of the main features of this library include:

* Event dissolves using ``EventsCollection.dissolve()``
* Merging and overlaying multiple tables of events with the ``EventsCollection.merge()`` method and the ``EventsMerge`` class API and its many linearly-weighted overlay aggregators
* Linear aggregations of data such as sliding window analysis with the powerful ``EventsMerge.distribute()`` method
* Resegmentation of linear data with ``EventsCollection.to_windows()`` and related methods
* Creating unions of multiple ``EventsCollection`` instances with the ``EventsUnion`` object class.

Code Snippets
=============
Create an events collection for a sample roadway events dataframe with unique  
route identifier represented by the 'Route' column and data for multiple years, 
represented by the 'Year' column. The begin and end mile points are defined by 
the 'Begin' and 'End' columns::

    ec = EventsCollection(df, keys=['Route','Year'], beg='Begin', end='End')

To select events from a specific route and a specific year, indexing for all 
keys can be used, producing an EventsGroup::

    eg = ec['Route 50', 2018]

To select events on all routes but only those from a specific year, indexing 
for only some keys can be used::

    ec_2018 = ec[:, 2018]

To get all events which intersect with a numeric range, the intersecting() 
method can be used on an EventsGroup instance::

    df_intersecting = eg.intersecting(0.5, 1.5, closed='left_mod')

The intersecting() method can also be used for point locations by ommitting the 
second location attribute::

    df_intersecting = eg.intersecting(0.75, closed='both')

The linearly weighted average of one or more attributes can be obtained using 
the overlay_average() method::

    df_overlay = eg.overlay_average(0.5, 1.5, cols=['Speed_Limit','Volume'])

If the events include information on the roadway speed limit and number of 
lanes, they can be dissolved on these attributes. During the dissolve, other 
attributes can be aggregated, providing a list of associated values or 
performing an aggregation function over these values::

    ec_dissolved = ec.dissolve(attr=['Speed_Limit','Lanes'], aggs=['County'])

Version Notes
=============
0.0.11 (TBD)
-------------------
Feeling the algo-rhythm these days.

* Initial deployment of synthesis module featuring some tools for generating linear referencing information for chains of linear asset data with geometry but no LRS. These features are currently experimental and outputs should be reviewed for quality and expected outcomes. They will get refined in future versions based on performance in various applications and input from users.
* Addition of ``retain`` parameter to the ``EventsCollection.to_windows()`` method which retains all non-target and non-spatial fields from the original events dataframe when performing the operation. Previously, only newly generated target fields would be present in the output ``EventsCollection``'s events dataframe.
* Fixed implementation of ``EventsCollection.spatials`` property to correctly return a list of spatial column labels (e.g., geometry and route column labels) in the events dataframe. This also corrects ``EventsCollection.others`` which previously incorrectly included these labels.
* Performance improvements
* Various bug fixes, minor features

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
