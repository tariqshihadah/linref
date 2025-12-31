# Linref Usage Guide

This guide provides comprehensive documentation for using linref's redesigned API.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Setting Up LRS](#setting-up-lrs)
3. [Working with Events](#working-with-events)
4. [Dissolving and Merging](#dissolving-and-merging)
5. [Overlay Operations](#overlay-operations)
6. [Geometry Operations](#geometry-operations)
7. [Advanced Patterns](#advanced-patterns)
8. [Performance Tips](#performance-tips)

## Core Concepts

### Linear Referencing System (LRS)

An LRS defines the schema of your linearly referenced data. It's a lightweight configuration object that tells linref how to interpret your DataFrame columns.

```python
import linref as lr

# Create an LRS
my_lrs = lr.LRS(
    key_col=['route', 'direction'],  # Grouping columns
    beg_col='begin_mp',               # Begin mile point
    end_col='end_mp',                 # End mile point
    geom_col='geometry',              # Shapely geometry column
    geom_m_col='geometry_m',          # M-enabled geometry column
    closed='left_mod'                 # Range closure type
)
```

### EventsData

`EventsData` is the core computational engine that powers linref operations. While you typically interact with DataFrames using the `.lr` accessor, understanding `EventsData` helps with advanced operations.

```python
# Access EventsData from a DataFrame
events = df.lr.events

# EventsData properties
print(events.num_events)    # Number of events
print(events.is_linear)     # Is this linear or point data?
print(events.is_grouped)    # Are events grouped?
print(events.groups)        # Event group identifiers
print(events.begs)          # Begin locations
print(events.ends)          # End locations
```

### DataFrame Accessor Pattern

The `.lr` accessor extends pandas DataFrames with linear referencing capabilities:

```python
import pandas as pd

df = pd.DataFrame({
    'route': ['A', 'A', 'B'],
    'beg': [0.0, 1.0, 0.0],
    'end': [1.0, 2.0, 1.5],
    'volume': [100, 150, 120]
})

# Set LRS on the DataFrame
df = df.lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end',
    closed='right'
)

# Now all .lr methods are available
print(df.lr.is_linear)      # True
print(df.lr.is_grouped)     # True
dissolved = df.lr.dissolve()
```

## Setting Up LRS

### Method 1: Set LRS on Individual DataFrames

```python
df = df.lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end',
    closed='right'
)
```

### Method 2: Set Default LRS

Setting a default LRS is useful when working with multiple DataFrames that share the same schema:

```python
# Define default LRS
default_lrs = lr.LRS(
    key_col=['route', 'direction'],
    beg_col='begin_mp',
    end_col='end_mp',
    closed='left_mod'
)

lr.set_default_lrs(default_lrs)

# Now all DataFrames automatically have this LRS
df1 = pd.read_csv('roads.csv')
print(df1.lr.is_lrs_set)  # True
```

### Method 3: Copy LRS from Another DataFrame

```python
# Copy LRS settings from df1 to df2
df2 = df2.lr.lrs_like(df1)
```

### Modifying LRS

```python
# Modify existing LRS
df = df.lr.modify_lrs(beg_col='start_mp', end_col='finish_mp')

# Add a key column
df = df.lr.add_key('year')

# Remove a key column
df = df.lr.remove_key('year')
```

## Working with Events

### Selecting Events by Group

```python
# Get a specific group
route_a = df.lr.get_group('A')
route_a_2020 = df.lr.get_group(('A', 2020))

# Iterate over all groups
for group, group_df in df.lr.iter_groups():
    print(f"Group {group}: {len(group_df)} events")
```

### Querying by Location

```python
# Get events at specific locations
events = df.lr.events

# Check which events intersect a range
mask = events.intersect_range(beg=0.5, end=1.5)
intersecting = df[mask]

# Select events containing a point
mask = events.contains_point(loc=0.75)
containing = df[mask]
```

### Sorting Events

```python
# Sort by standard order (groups, then begin, then end)
df_sorted = df.lr.sort_standard()

# Get sorting index
df_sorted, sorter = df.lr.sort_standard(return_index=True)
```

## Dissolving and Merging

### Basic Dissolve

Dissolving merges consecutive events within each group:

```python
# Simple dissolve - merges all consecutive events
dissolved = df.lr.dissolve()

# Dissolve while retaining specific columns
dissolved = df.lr.dissolve(
    retain=['speed_limit', 'pavement_type'],
    sort=True
)

# Track which original events were merged
dissolved = df.lr.dissolve(
    inverse_index=True,
    inverse_col='original_indices'
)
```

### Understanding Dissolve

Dissolve only merges events that are:
1. In the same group (same key values)
2. Consecutive (no gaps between them)
3. Have the same retained attribute values (if `retain` is specified)

```python
# Example: These will be merged
# Route A: [0-1], [1-2], [2-3] -> [0-3]

# Example: These won't be merged (gap)
# Route A: [0-1], [2-3] -> [0-1], [2-3]

# Example: These won't be merged (different attributes)
# Route A, Speed 55: [0-1], [1-2]
# Route A, Speed 45: [2-3]
# Result: [0-2], [2-3]
```

## Overlay Operations

### Basic Overlay

Overlay operations create relationships between two linearly referenced datasets:

```python
# Two datasets with compatible LRS
roads = roads_df.lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end'
)

traffic = traffic_df.lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end'
)

# Overlay traffic data onto roads
result = roads.lr.overlay(
    traffic,
    left_cols=['lanes', 'surface_type'],
    right_cols=['aadt', 'truck_pct'],
    agg_funcs={'aadt': 'mean', 'truck_pct': 'mean'}
)
```

### Advanced Overlay with EventsRelation

For more control, use the `EventsRelation` object:

```python
# Create relationship between two datasets
relation = roads.lr.relate(traffic)

# Get overlay data (sparse matrix of overlap lengths)
overlay_matrix = relation.overlay(normalize=True, norm_by='left')

# Get intersection matrix (boolean)
intersect_matrix = relation.intersect()

# Count occurrences using count method
counts = relation.count()

# Get first or last values
first_speed = relation.first(data=traffic['speed'].values)

# Calculate sum or mean
total_crashes = relation.sum(data=traffic['crash_count'].values)
avg_speed = relation.mean(data=traffic['speed'].values)
```

### Overlay Use Cases

**Use Case 1: Count overlapping events**
```python
# Count how many crashes occurred on each road segment
relation = roads.lr.relate(crashes)
roads['crash_count'] = relation.count()
```

**Use Case 2: Simple aggregations**
```python
# Get speed limit from overlapping zones
relation = roads.lr.relate(speed_zones)
# Use first matching zone
roads['speed_limit'] = relation.first(data=speed_zones['speed_limit'].values)
# Or compute mean if multiple zones overlap
roads['avg_speed_limit'] = relation.mean(data=speed_zones['speed_limit'].values)
```

**Use Case 3: Sum values across overlaps**
```python
# Sum funding from all overlapping zones
relation = roads.lr.relate(funding_zones)
roads['total_funding'] = relation.sum(data=funding_zones['amount'].values)
```

## Geometry Operations

### Generating Linear Events from Geometries

If you have a GeoDataFrame with route geometries but no mile markers:

```python
import geopandas as gpd

# Start with geometries
routes = gpd.read_file('routes.shp')

# Set basic LRS
routes = routes.lr.set_lrs(
    key_col=['route_id'],
    geom_col='geometry'
)

# Generate mile markers based on geometry lengths
routes = routes.lr.generate_linear_events(
    beg_col='beg_mp',
    end_col='end_mp',
    chain_col='chain',
    scale=1/5280,  # Convert feet to miles
    decimals=3,
    add_geom_m=True
)

# Now routes has:
# - beg_mp, end_mp columns
# - chain column (for handling disjoint segments)
# - geometry_m column (M-enabled geometries)
```

### Creating Geometries from Mile Markers

If you have mile markers but no geometries:

```python
# Event data with mile markers
events_df = pd.DataFrame({
    'route': ['A', 'A', 'B'],
    'beg': [0.0, 1.0, 0.0],
    'end': [1.0, 2.0, 1.5],
    'volume': [100, 150, 120]
}).lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end'
)

# Reference routes with M-enabled geometries
ref_routes = ref_gdf.lr.set_lrs(
    key_col=['route'],
    geom_m_col='geometry_m'
)

# Cut geometries for events using cut_from method
events_gdf = events_df.lr.cut_from(ref_routes)
```

### Extracting M-Values from Geometries

```python
# Extract begin and end M-values from M-enabled geometries
df = df.lr.extract_m_values(
    beg_col='beg_mp',
    end_col='end_mp'
)
```

### Working with Chains

Chains help handle disjoint road segments (e.g., when a route splits or has gaps):

```python
# Add chain indices
df = df.lr.add_chaining(
    name='chain',
    enforce_m=True  # Require M-enabled geometries
)

# Get chain indices
chains = df.lr.get_chains()
```

## Advanced Patterns

### Pattern 1: Resegmentation

Create uniform segments from variable-length events:

```python
# Create 0.1-mile segments
segmented = df.lr.resegment(
    length=0.1,
    retain=['speed_limit', 'surface_type']
)

# Variable-length segments
import numpy as np
lengths = np.random.uniform(0.05, 0.15, len(df))
segmented = df.lr.resegment(
    length=lengths,
    fill='cut'  # Options: 'cut', 'left', 'right', 'extend'
)
```

### Pattern 2: Integration

Combine multiple event datasets into a unified framework:

```python
# Multiple datasets with matching LRS
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

# Integrate all datasets at once
integrated_df = pavement_df.lr.integrate(
    [traffic_df, crash_df],
    fill_gaps=False,
    split_at_locs=False
)
```

### Pattern 3: Point to Linear Conversion

Convert point events to linear events with buffers:

```python
# Point events
signs = signs_df.lr.set_lrs(
    key_col=['route'],
    loc_col='milepost'
)

# Convert to linear with buffer
buffered = signs.lr.extend(
    extend_begs=0.1,  # 0.1 miles before point
    extend_ends=0.1   # 0.1 miles after point
)

# Or convert explicitly
linear = signs.lr.point_to_linear(
    beg_col='beg',
    end_col='end',
    drop_loc=False
)
```

### Pattern 4: Shifting and Rounding

Adjust event locations:

```python
# Shift all events by 0.5 miles
shifted = df.lr.shift(shift=0.5)

# Round to 2 decimal places
rounded = df.lr.round(decimals=2)

# Extend events
extended = df.lr.extend(
    extend_begs=0.1,  # Extend 0.1 before
    extend_ends=0.2   # Extend 0.2 after
)
```

## Performance Tips

### 1. Use Default LRS

Set a default LRS once instead of configuring each DataFrame:

```python
# Do this once at the start
lr.set_default_lrs(my_lrs)

# All subsequent DataFrames inherit this configuration
df1 = pd.read_csv('file1.csv')  # Automatically has LRS
df2 = pd.read_csv('file2.csv')  # Automatically has LRS
```

### 2. Sort Before Dissolve

Pre-sorting can improve dissolve performance:

```python
df = df.lr.sort_standard()
dissolved = df.lr.dissolve()
```

### 3. Cache Relations for Multiple Operations

When performing multiple operations on the same relationship:

```python
# Enable caching (default)
relation = df1.lr.relate(df2, cache=True)

# Multiple operations reuse cached data
counts = relation.count()
first_vals = relation.first()
overlay = relation.overlay()

# Disable caching to save memory for one-time operations
relation = df1.lr.relate(df2, cache=False)
```

### 4. Use Chunking for Large Overlay Operations

Control memory usage with chunking in overlay:

```python
# Overlay with smaller chunk size for memory efficiency
overlay_matrix = df1.lr.overlay(df2, chunksize=500)  # Default is 1000
```

### 5. Drop Invalid Events

Remove events with missing data before processing:

```python
# Check for invalid events
valid_mask = df.lr.valid_events
print(f"Found {(~valid_mask).sum()} invalid events")

# Drop invalid events
df_clean = df.lr.drop_invalid_events()
```

## Common Pitfalls

### 1. LRS Compatibility

Ensure DataFrames have compatible LRS settings for overlay operations:

```python
# Both must have same number and type of key columns
df1 = df1.lr.set_lrs(key_col=['route', 'direction'], ...)
df2 = df2.lr.set_lrs(key_col=['route', 'direction'], ...)

# This will fail - incompatible keys
df3 = df3.lr.set_lrs(key_col=['route'], ...)  # Only one key
result = df1.lr.overlay(df3)  # Error!
```

### 2. Geometry Synchronization

Operations on begin/end columns don't automatically update geometries:

```python
# This modifies begin/end but not geometries
df = df.lr.shift(shift=0.5)

# To update geometries, regenerate them
df = df.lr.add_geom_m()
```

### 3. Monotonicity

Linear events should have begin <= end:

```python
# Enforce monotonicity
df = df.lr.set_monotonic()
```

### 4. Missing Groups

Relate and overlay operations require matching groups:

```python
# If df1 has route 'A' but df2 doesn't, no results for route 'A'
# Check LRS configuration first
print("Keys in df1:", df1.lr.key_col)
print("Keys in df2:", df2.lr.key_col)

# Iterate groups to check data
for group, group_df in df1.lr.iter_groups():
    print(f"Group {group}: {len(group_df)} events")
```

## Additional Resources

- **API Documentation**: https://linref.readthedocs.io/
- **Example Notebooks**: See `examples/` directory
- **GitHub**: https://github.com/tariqshihadah/linref
- **Issues**: https://github.com/tariqshihadah/linref/issues
