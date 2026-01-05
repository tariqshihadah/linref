# Linref Usage Guide

This guide covers the essential features and usage patterns for linref v1.0.

## Quick Start

### Installation

```bash
pip install linref
```

### Basic Example

```python
import pandas as pd
import linref as lr

# Create a DataFrame with linearly referenced events
df = pd.DataFrame({
    'route': ['A', 'A', 'B'],
    'beg': [0.0, 1.0, 0.0],
    'end': [1.0, 2.0, 1.5],
    'speed_limit': [55, 45, 65]
})

# Set up the LRS (Linear Referencing System)
df = df.lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end',
    closed='right'
)

# Use linear referencing methods
dissolved = df.lr.dissolve()
print(dissolved)
```

## Core Concepts

### Linear Referencing System (LRS)

The LRS defines your data's schema:

- **Key columns** (`key_col`) - Route identifiers (e.g., `['route']`, `['route', 'county']`)
- **Location columns** - Point events (`loc_col`) or linear events (`beg_col`, `end_col`)
- **Geometry columns** - Spatial data (`geom_col`, `geom_m_col` for M-enabled geometries)
- **Closure type** (`closed`) - How range endpoints are handled: `'left'`, `'right'`, `'both'`, `'neither'`, `'left_mod'`, `'right_mod'`

### Setting Up an Existing LRS

**Method 1: Set LRS per DataFrame**

```python
df = df.lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end',
    closed='right'
)
```

**Method 2: Set Default LRS**

```python
# Define default LRS once
default_lrs = lr.LRS(
    key_col=['route'],
    beg_col='begin_mp',
    end_col='end_mp',
    closed='left_mod'
)

# Set as default for all DataFrames
lr.set_default_lrs(default_lrs)
```

## Essential Operations

### Dissolving Events

Merge consecutive events with the same attributes:

```python
# Dissolve events, keeping specific columns
dissolved = df.lr.dissolve(
    retain=['speed_limit', 'pavement_type'],
    sort=True
)
```

### Selecting Events by Group

```python
# Get events for a specific group
route_a = df.lr.get_group('A')

# Iterate over all groups
for group, group_df in df.lr.iter_groups():
    print(f"Processing {group}: {len(group_df)} events")
```

### Resegmentation

Create uniform segments from variable-length events:

```python
# Create 0.1-mile segments
segmented = df.lr.resegment(length=0.1)
```

### Relating Datasets

Create relationships between two linearly referenced datasets:

```python
# Set up two datasets with matching LRS
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

# Create relationship and count crashes per road segment
relation = roads.lr.relate(crashes)
roads['num_crashes'] = relation.count()
```

### Aggregating Data Across Relationships

```python
# Create relation between datasets
relation = roads.lr.relate(crashes)

# Count occurrences
roads['crash_count'] = relation.count()

# Get first matching value
roads['severity'] = relation.first(data=crashes['severity'].values)

# Calculate mean across overlapping events
roads['avg_severity'] = relation.mean(data=crashes['severity'].values)

# Calculate sum
roads['total_damage'] = relation.sum(data=crashes['damage_cost'].values)
```

## Working with Point Events

### Point-Based Data

```python
# Point events with location column
signs = pd.DataFrame({
    'route': ['A', 'A', 'B'],
    'milepost': [0.5, 1.2, 0.8],
    'sign_type': ['Speed Limit', 'Stop', 'Yield']
}).lr.set_lrs(
    key_col=['route'],
    loc_col='milepost'
)
```

### Converting Point to Linear Events

```python
# Extend point events to create buffer zones
sign_zones = signs.lr.extend(
    extend_begs=0.1,  # 0.1 before
    extend_ends=0.1   # 0.1 after
)
```

## Geometry Operations

### Generating Linear Events from Geometries

If you have a GeoDataFrame with route geometries but no mile markers:

```python
import geopandas as gpd

# GeoDataFrame with route geometries
routes_gdf = gpd.read_file('routes.geojson')

routes = routes_gdf.lr.set_lrs(
    key_col=['route'],
    geom_col='geometry'
)

# Generate begin/end mile markers from geometry lengths
routes = routes.lr.generate_linear_events(
    beg_col='beg_mp',
    end_col='end_mp',
    scale=1/5280,  # Convert feet to miles
    decimals=3,
    add_geom_m=True
)
```

### Cutting Geometries

Cut geometries for your events from reference routes:

```python
# Events with mile markers, need geometries
events_df = events_df.lr.set_lrs(
    key_col=['route'],
    beg_col='beg',
    end_col='end'
)

# Reference routes with M-enabled geometries
ref_routes = ref_gdf.lr.set_lrs(
    key_col=['route'],
    geom_m_col='geometry_m'
)

# Cut geometries for events
events_gdf = events_df.lr.cut_from(ref_routes)
```

### Extracting M-Values

Extract begin and end M-values from M-enabled geometries:

```python
df = df.lr.extract_m_values(
    beg_col='extracted_beg',
    end_col='extracted_end'
)
```

## Integrating Multiple Datasets

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

## Common Patterns

### Pattern 1: Dissolving by Attributes

```python
# Dissolve consecutive events with same attributes
dissolved = df.lr.dissolve(
    retain=['speed_limit', 'pavement_type'],
    inverse_index=True,  # Track original events
    inverse_col='original_ids'
)
```

### Pattern 2: Crash Analysis

```python
# Count and analyze crashes on road segments
relation = roads.lr.relate(crashes)

# Count crashes per segment
roads['crash_count'] = relation.count()

# Get crash severity data
crash_severity = crashes['severity'].values
roads['avg_severity'] = relation.mean(data=crash_severity)
```

### Pattern 3: Uniform Segmentation

```python
# Create uniform 0.5-mile segments
segmented = roads.lr.resegment(length=0.5)
```

### Pattern 4: Buffering Point Events

```python
# Convert point events to linear with buffers
signs = signs_df.lr.set_lrs(
    key_col=['route'],
    loc_col='milepost'
)

# Extend to create buffer zones
sign_zones = signs.lr.extend(
    extend_begs=0.25,  # 0.25 miles before
    extend_ends=0.25   # 0.25 miles after
)
```

## Performance Tips

### 1. Sort Before Processing

```python
# Sort events before dissolve or other operations
df = df.lr.sort_standard()
```

### 2. Drop Invalid Events

```python
# Remove events with missing data
df_clean = df.lr.drop_invalid_events()
```

### 3. Use Chunking for Large Overlays

```python
# Control memory usage with chunking
overlay_matrix = df1.lr.overlay(df2, chunksize=500)
```

### 4. Cache Relations

```python
# Enable caching (default) for multiple operations
relation = df1.lr.relate(df2, cache=True)

# Multiple operations reuse cached data
counts = relation.count()
first_vals = relation.first()
```

## API Reference Summary

### DataFrame Methods (via `.lr` accessor)

- **Configuration**: `set_lrs()`, `modify_lrs()`, `clear_lrs()`
- **Selection**: `get_group()`, `iter_groups()`
- **Transformation**: `dissolve()`, `resegment()`, `extend()`, `shift()`, `round()`
- **Relationships**: `relate()`, `overlay()`, `integrate()`
- **Geometry**: `generate_linear_events()`, `add_geom_m()`, `extract_m_values()`, `cut_from()`, `interpolate_from()`
- **Validation**: `drop_invalid_events()`, `set_monotonic()`

### EventsRelation Methods

- **Aggregation**: `count()`, `first()`, `last()`, `sum()`, `mean()`
- **Geometry**: `cut()`, `interpolate()`
- **Data Transfer**: `distribute()`

## Troubleshooting

### "No LRS set"

Make sure you've called `.set_lrs()` on your DataFrame:

```python
df = df.lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end')
```

### "LRS columns not found"

Verify that the column names in your LRS match your DataFrame:

```python
print(df.columns)
print(df.lr.lrs)  # Check current LRS configuration
```

### Missing Groups in Relations

Relate and overlay operations require matching groups:

```python
# Check LRS configuration
print("Keys in df1:", df1.lr.key_col)
print("Keys in df2:", df2.lr.key_col)

# Iterate groups to verify data
for group, group_df in df1.lr.iter_groups():
    print(f"Group {group}: {len(group_df)} events")
```

## Additional Resources

- **API Documentation**: https://linref.readthedocs.io/
- **GitHub**: https://github.com/tariqshihadah/linref
- **Issues**: https://github.com/tariqshihadah/linref/issues
