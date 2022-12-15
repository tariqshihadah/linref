# Overview
Module featuring EventsCollection object class for the management of linearly referenced data and optimized performance of various simple and complex events and geospatial operations.

# Version Notes
## 0.0.9 (TBD)
- Improve performance of .project() method when nearest=False by removing dependence on join_nearby() function and using native gpd features.

## 0.0.8 (2022-12-14)
- Improve performance of essential .get_group() method, reducing superfluous initialization of empty dataframes and events collections and improving logging of initialized groups.
- Improve performance of .union() method with updated RangeCollection.union() features and optimized iteration and aggregation of unified data. Performance times are significantly improved, especially for large datasets with many events groups.
- Improve distribute method performance which was added in recent versions.
- Drop duplicates in .project() method when using sjoin_nearest with newer versions of geopandas. Improved validation in .project() method, address edge case where projecting geometry column has a non-standard label (e.g., not 'geometry').
- Added .sort() method to events collection. Default sorting methods remain unchanged.
- Added warnings for missing data in target columns when initializing an EventsFrames through standard methods.
- Remove .project_old() method from events collection due to deprecation.
- Performance improvements
- Various bug fixes, minor features

## 0.0.7 (2022-10-14)
- Refactoring of EventsMerge system from 2D to 3D vectorized relationships for improved performance and accuracy. API and aggregation methods are largely the same.
- Modified closed parameter use in merge relationships in accordance with rangel v0.0.6, which now performs intersections which honor the closed parameter on the left collection as well as the right collection. This provides more accurate results for events which fall on the edges of intersecting events when using left_mod or right_mod closed parameters.
- Updates to account for rangel 0.0.6 version which is now a minimum version requirement. Added other minimum version requirements for related packages.
- Performance improvements
- Various bug fixes, minor features

## 0.0.5.post1 (2022-09-06)
- Address deprecation of length of and iteration over multi-part geometries in shapely
- Remove code redundancies in linref.events.collection for get_most and get_mode

## 0.0.5 (2022-09-01)
- Added sumproduct and count aggregators to EventsMergeAttribute class
- Address deprecation of length of and iteration over multi-part geometries in shapely
- Performance improvements
- Various bug fixes, minor features

## 0.0.4 (2022-06-24)
- Minor feature additions
- Performance improvements
- Addition of logos in github repo
- Various bug fixes, minor features

## 0.0.3 (2022-06-07)
- Various updates for geopandas 0.10+ dependency including improved performance of project methods
- Automatic sorting of events dataframe prior to performing dissolve
- Performance improvements
- Various bug fixes, minor features

## 0.0.2 (2022-04-11)
- Various bug fixes, minor features

## 0.0.1 (2022-03-31)
- Original experimental release.