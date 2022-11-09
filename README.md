# Overview
Module featuring EventsCollection object class for the management of linearly referenced data and optimized performance of various simple and complex events and geospatial operations.

# Version Notes
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