# Linref Usage Guide

This guide covers the essential features and usage patterns for linref v1.0.

## Installation

Linref can easily be installed using `pip`. The package is dependent on a few major libraries (`geopandas`, `numpy`, `scipy`) which may require additional effort on some machines. Please review documentation for those packages as needed if any issues arise.

```bash
pip install linref
```

## Core Functionality

First, import `linref` with the shortened `lr` for ease of access and consistency with the dataframe accessor.

```python
import linref as lr
```

### Load Sample Datasets

```python
# Load sample roadway, crash, and pavement datasets
# By default, LRS is not configured - you must set it up
roadways = lr.datasets.load('roadways')
crashes  = lr.datasets.load('crashes')
pavement = lr.datasets.load('pavement')

# Or load with LRS pre-configured using set_lrs=True
# roadways = lr.datasets.load('roadways', set_lrs=True)

print(roadways.head(3))
```

### Setting Up an Existing LRS

**Method 1: Set LRS per DataFrame**

```python
roadways = roadways.lr.set_lrs(
    key_col=['route'],
    loc_col='loc', # Should be defined even if not present in all datasets
    beg_col='beg',
    end_col='end',
    geom_col='geometry',
    geom_m_col='geometry_m', # Not currently present but will be generated later
    closed='left_mod' # 'left_mod' is a the recommended convention for most linear datasets
)
print(roadways.lr)
```

**Method 2: Set Default LRS**

```python
# Define default LRS once
default_lrs = lr.LRS(
    key_col=['route'],
    loc_col='loc', # Should be defined even if not present in all datasets
    beg_col='beg',
    end_col='end',
    geom_col='geometry',
    geom_m_col='geometry_m', # Not currently present but will be generated later
    closed='left_mod' # 'left_mod' is a the recommended convention for most linear datasets
)

# Set as default for all DataFrames
lr.set_default_lrs(default_lrs)

# Now all DataFrames will use this LRS unless otherwise specified
print(crashes.lr)
```

## Essential Operations

### Standard Sorting

To ensure best performance, be sure to keep your data sorted. This can be achieved with the `sort_standard` method, which sorts data by your key columns (e.g., `key_col`) followed by your event measure columns (e.g., `loc_col`, `beg_col`, and `end_col`).

```python
# Sort events upon loading to ensure best performance and correctness in downstream operations
resorted = roadways.sample(frac=1).lr.sort_standard()

print(resorted.head(3)[['route', 'beg', 'end']])
```

### Remove Invalid Events

To avoid complications with some linref functionality, it is best to remove invalid events that may cause errors. This can be done with the `drop_invalid_events` method or by using the `valid_events` or `invalid_events` properties. These check for records which have missing data in the key columns or in event measure columns.

```python
# Remove events with missing data to avoid downstream errors
has_invalid = roadways.copy()
has_invalid.loc[1, 'beg'] = None
all_valid = has_invalid.lr.drop_invalid_events()
print(all_valid.head(3)[['route', 'beg', 'end']])
```

### Selecting Events by Group

If you simply need to access or analyze the dataframe, grouped by the unique groups represented in the LRS's key column(s), you can do that with the `get_group` or `iter_groups` methods.

```python
# Get events for a specific group
print(crashes.lr.get_group('SR-1')[['crash_id', 'route', 'loc']])

# Iterate over all groups to subset the data by unique groups
for group_name, group_df in crashes.lr.iter_groups():
    print(f"Group {group_name} has {len(group_df)} events")

# You can similarly get counts of events per group with the group_counts method
print(crashes.lr.group_counts())
```

### Dissolving Events

Merge consecutive linear events within the same group based on shared event begin and end points. To dissolve by additional attributes, use the `retain` parameter which accepts a list of one or more dataframe column labels.

```python
# Dissolve events, keeping specific columns
dissolved = roadways.lr.dissolve()

# Note that the dissolved_index column links back to the original events
view_columns = ['route', 'beg', 'end', 'dissolved_index']
print(dissolved[view_columns])

# Use the retain parameter to keep additional columns
print(roadways.lr.dissolve(retain=['speed_limit'])[view_columns + ['speed_limit']])
```

The dissolve function will also merge geometries by default, retaining event bound information by upgrading geometries to be m-enabled.

```python
# View the geometries created by the dissolve operation
print(dissolved.iloc[0].beg, dissolved.iloc[0].end)
print(dissolved.iloc[0].geometry)
print(dissolved.iloc[0].geometry_m)
```

**NOTE:** The LINESTRING M shown here is using linref's `LineStringM` class located in the events.geometry module. This is an implementation of m-enabled linear geometries that provides an extension of the existing `shapely.LineString` with the features needed for linref's core functionality. Future versions of shapely are expected to provide native support for m-enabled linear geometries at which point linref will transition to their implementation. For now, please be cautious and review appropriate documentation if you are intending to use exports of these geometries in other programs. Use the WKT features of the `LineStringM` class and the linref `LRS_Accessor` class to support interoperability for now.

### Resegmentation

Continuing with data engineering, we can take the dissolved roadways layer and resegment it to a fixed length using the `resegment` method:

```python
# Create 5-mile segments
segmented = dissolved.lr.resegment(length=5)
print(segmented.lr.get_group('I-5')[['route', 'beg', 'end']])
```

Note that by default, the final segment is allowed to be shorter than others if the length of the parent route isn't a multiple of the desired length. This behavior can be modified using the `fill` parameter:

```python
# Various fill parameter options
resegmented = dissolved.lr.resegment(length=5, fill='cut') # cut (default)
print(resegmented.lr.get_group('I-5')[['route', 'beg', 'end']])

resegmented = dissolved.lr.resegment(length=5, fill='extend') # extend
print(resegmented.lr.get_group('I-5')[['route', 'beg', 'end']])

resegmented = dissolved.lr.resegment(length=5, fill='balance') # balance (cut when long, extend when short)
print(resegmented.lr.get_group('I-5')[['route', 'beg', 'end']])
# Not shown: right, left
```

### Relating Datasets

An important functionality of linearly referenced data is events-based conflation. Linref makes this easy with the `relate` method, which builds dynamic relationships between two linearly referenced dataframes that can then be aggregated with a variety of methods.

#### Conflating *point* data onto *linear* data

```python
# Conflating point data onto linear data
#          left_df               right_df   <-- relations use terminology of left and right
relation = resegmented.lr.relate(crashes)

# Dataframe-level aggregations like count are easy
print(relation.count()) # Count the number of crashes on each roadway segment

# Aggregator results are designed to be assigned as new columns to the left dataframe
resegmented['crash_counts'] = relation.count()
print(resegmented.head(3)[['route', 'beg', 'end', 'crash_counts']])
```

Column-level aggregators can be accessed using column indexing on the `relation` instance, using a syntax similar to the pandas `groupby` syntax:

```python
# List of unique values (creates a single column)
resegmented['crash_ids'] = relation['crash_id'].list()
# Apply an aggregator to multiple columns at once
resegmented[['crash_ids', 'severities']] = relation[['crash_id', 'severity']].list()
# Counts of unique values (creates a number of columns equal to the number of unique values)
severities = ['Fatal', 'Injury', 'Property Damage Only']
resegmented[severities] = relation['severity'].value_counts()[severities] # Returned as a dataframe

print(resegmented.head(3)[['route', 'beg', 'end', 'crash_ids', 'severities'] + severities])
```

#### Conflating *linear* data onto *linear* data

```python
# Conflating linear data onto linear data
relation = resegmented.lr.relate(pavement)

# Length-weighted mean of numerical data
resegmented['condition_rating'] = relation['condition_rating'].mean().round(3)
resegmented['condition_ratings'] = relation['condition_rating'].list()

print(resegmented.head(3)[['route', 'beg', 'end', 'condition_rating', 'condition_ratings']])

# Length-weighted mode of categorical data
resegmented['surface_type'] = relation['surface_type'].mode()
resegmented['surface_types'] = relation['surface_type'].list()

print(resegmented.head(3)[['route', 'beg', 'end', 'surface_type', 'surface_types']])
```

#### Creating new geometries from relations

We can create new geometries for linearly referenced point or linear dataframes by relating them to linearly referenced dataframes containing m-enabled geometries. When dissolving the roadways layer earlier, that method automatically created m-enabled geometries to retain dissolved event information. We can create them manually for other layers using the `add_geom_m` method, which applies event boundaries to the existing geometries.

```python
# Add m-enabled geometries to the roadways layer
roadways = roadways.lr.add_geom_m()

# Interpolate new crash geometries based on their LRS information
interpolated = crashes.lr.relate(roadways).interpolate()
print(list(crashes.geometry)[:3])
print(list(interpolated)[:3])

# Cut new roadway geometries from the parent LRS layer
cut = roadways.lr.relate(dissolved).cut()
print(list(roadways.geometry_m)[:3])
print(list(cut)[:3])
```

## LRS Creation and Compatibility

### Saving and Loading M-Enabled Geometry with WKT

Due to limited support of M-enabled geometries in shapely and geopandas, saving and loading data containing linref's LineStringM geometries is best done using WKT.

- **Loading Data.** When loading data, you can parse WKT with an M dimension using `df.lr.parse_geom_m_wkt(inplace=True)`. Be sure that the M-enabled geometry column is included in the LRS being used.

- **Saving Data** When saving data to a text-based format, LineStringM objects will automatically be converted to WKT. When saving data to a database format, you must indicate the type of the M-enabled geometry column using `dtype={'geometry_m': str}`, such as when using pandas' `to_sql` or geopandas' `to_postgis`.

### Processing Various Data Formats

Depending on the source, your data may be in a variety of states. Here is some general guidance on how to prepare data given a few case examples:

#### 1. Spatialized linear layer with no existing LRS

Load the data using geopandas and generate a new LRS on the data using the `df.lr.generate_linear_events` method described below.

#### 2. Spatialized linear or point layer with event begin and end or location measures

Load the data using geopandas and validate the existing LRS. First, check that all records have valid event information using `df.lr.invalid_events.sum() > 0` or by dropping invalid events with `df.lr.drop_invalid_events()`. Then, for linear events only, confirm that the data is properly chained with no disjoints within any unique route (i.e., dissolving the linear data produces single-part LineStrings) using `df.lr.is_chained`.

#### 3. Spatialized linear layer with M-enabled geometries

As of version 2.1, shapely provides some basic support of loading M-enabled geometries. Geopandas may require the `use_arrow=True` parameter when loading the data to be able to process these geometries without losing the M dimension. Load the data using geopandas and convert the shapely LineString objects containing M values which have minimal support to the linref LineStringM class using `df['geometry_m'] = lr.ext.parse_geoms_m_shapely(df.geometry)` or similar. Because the M dimension in the loaded geometry column may cause issues with some functionality, you can then remove the M dimension using `df['geometry'] = df.lr.geoms_m_reduced`. Finally, extract the begin and end measures from the processed M-enabled geometries for each event using `df.lr.extract_m_values(inplace=True)`.

### Generating a Linear Referencing System on Linear Data

If you have a dataset that contains linear geometries but is not linearly referenced, you can generate a new linear referencing system on that data using the `generate_linear_events` method. This will analyze your dataset, finding chains of contiguous geometries that share the same key column values. It then computes event begin and end measures according to the lengths of the chained geometries, defining each geometry's event boundaries.

This can be helpful when working with data that doesn't already include a linear referencing system (e.g., OpenStreetMaps roadway network data), enabling LRS-driven data engineering, analysis, and more using linref or another program like QGIS. Other associated data can then be projected to this new LRS, such as point or linear assets like street signs or project boundaries, creating a unified data system.

Note, the `generate_linear_events` method will typically add a new key column to the data called `'chain'`. This represents the index of the contiguous chain that a given geometry is a part of within its unique route defined by the key column(s). This is important to account for instances where spatial breaks may occur between chains of geometries to avoid creating a disjointed LRS. For example, when analyzing a roadway layer with the key column `street_name`, the route `'Main St'` may appear in multiple locations with separate chains of contiguous geometries. These chains would be assigned different chain indices and the `'chain'` column will be added to the dataframe's LRS as an additional key column to reflect this. If you are certain that this feature is not needed, you can avoid adding the chaining column with the parameter `add_chain=False`.

```python
# Generate a linear referencing system based on existing geometries
generated = roadways.lr.generate_linear_events(
    scale=5280, # Scales the distances calculated based on the coordinate reference system (e.g., unit conversion)
    decimals=0, # Round the computed event measures after scaling is applied
    add_chain=False, # Don't add chaining because we already know routes are not disjointed
    replace=True # Replaces existing LRS information for the sake of this example
)
print(generated.lr.get_group('I-5')[['route', 'beg', 'end']])
```

## Integrating Multiple Datasets

We can combine multiple event datasets into a single, unified dataframe using the `integrate` function. This analyzes a list of passed linearly referenced dataframes, creating a single version featuring the least common event intervals among them. New events can be tabularly joined back to each of their source dataframes using the created `integrated_index_[#]` columns.

```python
# Combine multiple datasets with a matching LRS
integrated = lr.integrate([roadways, pavement])

print(integrated.lr.get_group('SR-1'))
```

New geometries can be cut for these intervals by relating the integrated dataframe back to spatial dataframe on the same LRS.

```python
# Relate the integrated dataset back to the roadways layer
integrated['geometry_m'] = integrated.lr.relate(roadways).cut()

# A short-hand version of this pattern is available with the cut_from method
# This retrieves both M-enabled and non-M-enabled geometries in a single line
integrated.lr.cut_from(roadways, inplace=True)

print(integrated.columns)
```

## Additional Resources

- **API Documentation**: https://linref.readthedocs.io/
- **GitHub**: https://github.com/tariqshihadah/linref
- **Issues**: https://github.com/tariqshihadah/linref/issues
