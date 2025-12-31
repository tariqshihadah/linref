# Changelog

All notable changes to this project will be documented in this file.

## 1.0.0 (redesign branch) - Major Architectural Overhaul

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

**Note:** This is a complete redesign. Users of v0.1.x will need to update their code.

## 0.1.2 (2025-03-18)

* Allow for named index on dataframes being analyzed by `lr.generate_linear_events` by using dynamic indices during spatial join.
* Performance improvements
* Various bug fixes, minor features

## 0.1.1 (2024-08-20)

* Addition of array-like, dataframe label-based, and callable `length` parameter options to the `EventsCollection.to_windows()` method and `distance` parameter to the `EventsCollection.shift()` method. This allows for the creation of segments based on a variable length. This will be made available to other methods in future versions as well.
* Address `numpy>=2.x` compatibility issue in `EventsCollection.dissolve()` method which was causing errors in some cases. More fixes related to `numpy` updates are expected to come in future versions.
* Address datatype loss issues for LRS columns during `EventsCollection.to_windows()` and `EventsUnion.union()` methods. This was causing some instances of key columns to be converted to incorrect dtypes.
* Restructure .targets property to be `keys + [beg, end]` for more intuitive and consistent performance.
* Addition of `min` and `max` aggregation methods to the `EventsMergeAttribute` class.
* Several example Jupyter Notebooks for common use cases have been added to the `examples` folder for user reference. More to come.
* Performance improvements
* Various bug fixes, minor features

## 0.1.0 (2024-01-16)

My heart is in Gaza.

* Initial deployment of synthesis module featuring some tools for generating linear referencing information for chains of linear asset data with geometry but no LRS. These features are currently experimental and outputs should be reviewed for quality and expected outcomes. They will get refined in future versions based on performance in various applications and input from users.
* Addition of `retain` parameter to the `EventsCollection.to_windows()` method which retains all non-target and non-spatial fields from the original events dataframe when performing the operation. Previously, only newly generated target fields would be present in the output `EventsCollection`'s events dataframe.
* Fixed implementation of `EventsCollection.spatials` property to correctly return a list of spatial column labels (e.g., geometry and route column labels) in the events dataframe. This also corrects `EventsCollection.others` which previously incorrectly included these labels.
* Addition of `EventsCollection.clip()` method and expansion of the `EventsCollection.shift()` method for better parameterization.
* Transition to `pyproject.toml` setup with `setuptools`.
* Performance improvements
* Various bug fixes, minor features
* Why not push to v0.1 finally?

## 0.0.10 (2023-05-03)

Not a lot of updates to share, I guess that's a good thing?

* Minor updates to MLSRoute class to account for deprecation of subscripting MultiLineStrings. Most issues were addressed previously but a few were missed, most notably in the MLSRoute.bearing() method and a couple odd cases in the MLSRoute.cut() method.
* Fix minor issue with EventsCollection.project_parallel() implementation related to unmatched sampling points.
* Addition of EventsFrame.cast_gdf() method to cast events dataframes to geopandas geodataframes in-line.
* Performance improvements
* Various bug fixes, minor features

## 0.0.9 (2023-03-02)

First update of 2023. Been a quiet start to the year.

* Add missing .any() aggregation method to EventsMergeAttribute API. Was previously available but missed during a previous update.
* Update documentation
* Performance improvements
* Various bug fixes, minor features

## 0.0.8.post2 (2022-12-23)

Final update of 2022 including small feature updates and bug fixes from 0.0.8. Happy Holidays!

* Add .set_df() method for in-line modification of an EventsFrame's dataframe, inplace or not.
* Addition of suffixes parameter and default setting to EventsUnion.union() method.
* Performance improvements
* Various bug fixes, minor features

## 0.0.8.post1 (2022-12-16)

* Improve performance of .project() method when nearest=False by removing dependence on join_nearby() function and using native gpd features.
* Add .size and .shape properties to EventsFrames and subclasses.
* Various bug fixes, minor features

## 0.0.8 (2022-12-14)

* Improve performance of essential .get_group() method, reducing superfluous initialization of empty dataframes and events collections and improving logging of initialized groups.
* Improve performance of .union() method with updated RangeCollection.union() features and optimized iteration and aggregation of unified data. Performance times are significantly improved, especially for large datasets with many events groups.
* Improve distribute method performance which was added in recent versions.
* Drop duplicates in .project() method when using sjoin_nearest with newer versions of geopandas. Improved validation in .project() method, address edge case where projecting geometry column has a non-standard label (e.g., not 'geometry').
* Added .sort() method to events collection. Default sorting methods remain unchanged.
* Added warnings for missing data in target columns when initializing an EventsFrames through standard methods.
* Remove .project_old() method from events collection due to deprecation.
* Performance improvements
* Various bug fixes, minor features

## 0.0.7 (2022-10-14)

* Refactoring of EventsMerge system from 2D to 3D vectorized relationships for improved performance and accuracy. API and aggregation methods are largely the same.
* Modified closed parameter use in merge relationships in accordance with rangel v0.0.6, which now performs intersections which honor the closed parameter on the left collection as well as the right collection. This provides more accurate results for events which fall on the edges of intersecting events when using left_mod or right_mod closed parameters.
* Updates to account for rangel 0.0.6 version which is now a minimum version requirement. Added other minimum version requirements for related packages.
* Performance improvements
* Various bug fixes, minor features

## 0.0.5.post1 (2022-09-06)

* Address deprecation of length of and iteration over multi-part geometries in shapely
* Remove code redundancies in linref.events.collection for get_most and get_mode

## 0.0.5 (2022-09-01)

* Added sumproduct and count aggregators to EventsMergeAttribute class
* Address deprecation of length of and iteration over multi-part geometries in shapely
* Performance improvements
* Various bug fixes, minor features

## 0.0.4 (2022-06-24)

* Minor feature additions
* Performance improvements
* Addition of logos in github repo
* Various bug fixes, minor features

## 0.0.3 (2022-06-07)

* Various updates for geopandas 0.10+ dependency including improved performance of project methods
* Automatic sorting of events dataframe prior to performing dissolve
* Performance improvements
* Various bug fixes, minor features

## 0.0.2 (2022-04-11)

* Various bug fixes, minor features

## 0.0.1 (2022-03-31)

* Original experimental release.

## Future Improvements

* Check spatial continuity of events groups. This may apply at instantiation of `EventsCollection`.
* Unify direction of opposing routes which converge/diverge at a point (e.g., W Main St and E Main St).
