# Migration Guide: v0.1.x to v1.0 (redesign)

This guide helps users migrate from linref v0.1.x to the redesigned v1.0 API.

## Overview of Changes

The v1.0 redesign represents a fundamental architectural change:

**v0.1.x**: Class-based API with `EventsCollection`, `EventsGroup`, `EventsMerge`, etc.

**v1.0**: DataFrame accessor pattern with `.lr` accessor and `LRS` configuration objects

## Quick Reference

| v0.1.x | v1.0 |
|--------|--------|
| `EventsCollection(df, ...)` | `df.lr.set_lrs(...)` |
| `ec['Route A']` | `df.lr.get_group('Route A')` |
| `ec.dissolve(attr=[...])` | `df.lr.dissolve(retain=[...])` |
| `ec1.merge(ec2)` | `df1.lr.relate(df2)` |
| `em.overlay_average(...)` | Use `relation.count()`, `.mean()`, `.sum()`, etc. |
| `EventsUnion([ec1, ec2])` | `df1.lr.integrate([df2, df3])` |
| `eg.intersecting(0.5, 1.5)` | Filter with pandas/events methods |

## Detailed Migration Steps

### 1. Creating Events Collections

**v0.1.x:**
```python
from linref import EventsCollection

ec = EventsCollection(
    df,
    keys=['Route', 'Year'],
    beg='Begin',
    end='End'
)
```

**v1.0:**
```python
import linref as lr

df = df.lr.set_lrs(
    key_col=['Route', 'Year'],
    beg_col='Begin',
    end_col='End',
    closed='right'  # Explicitly set closure type
)
```

**Key Differences:**
- Work directly with DataFrames instead of wrapping them
- Column parameters now use `_col` suffix for clarity
- `closed` parameter is now explicit (defaults were implicit in v0.1.x)

### 2. Selecting Events

**v0.1.x:**
```python
# Select specific group
eg = ec['Route 50', 2018]

# Select partial keys
ec_2018 = ec[:, 2018]

# Get intersecting events
df_intersecting = eg.intersecting(0.5, 1.5, closed='left_mod')

# Get events at point
df_at_point = eg.intersecting(0.75, closed='both')
```

**v1.0:**
```python
# Select specific group
group_df = df.lr.get_group(('Route 50', 2018))

# Select partial keys - use pandas filtering
df_2018 = df[df['Year'] == 2018]

# Get intersecting events
events = df.lr.events
mask = events.intersect_range(0.5, 1.5)
df_intersecting = df[mask]

# Get events at point
mask = events.contains_point(0.75)
df_at_point = df[mask]
```

**Key Differences:**
- `get_group()` requires tuple for multi-key groups
- Partial key selection uses standard pandas filtering
- Location queries done through `EventsData` object accessed via `.events`
- Returns boolean masks instead of filtered DataFrames

### 3. Dissolving Events

**v0.1.x:**
```python
ec_dissolved = ec.dissolve(
    attr=['Speed_Limit', 'Lanes'],
    aggs=['County']
)
```

**v1.0:**
```python
df_dissolved = df.lr.dissolve(
    retain=['Speed_Limit', 'Lanes', 'County'],
    sort=True,
    inverse_index=True,  # Track original events
    inverse_col='original_ids'
)
```

**Key Differences:**
- `attr` → `retain`: All columns to keep during dissolve
- No separate `aggs` parameter - all retain columns are preserved
- `inverse_index` provides mapping back to original events
- Returns DataFrame directly, not EventsCollection

### 4. Merging and Overlaying

**v0.1.x:**
```python
# Merge collections
em = ec1.merge(ec2)

# Overlay average
df_avg = em.overlay_average(cols=['Speed_Limit', 'Volume'])

# Distribute
df_dist = em.distribute(values=values, windows=windows)
```

**v1.0:**
```python
# Create relation and use aggregation methods
relation = df1.lr.relate(df2)

# Count overlapping events
df1['count'] = relation.count()

# Get volume from first matching event
df1['volume'] = relation.first(data=df2['Volume'].values)

# Calculate mean volume across overlapping events
df1['mean_volume'] = relation.mean(data=df2['Volume'].values)

# Calculate sum across overlapping events
df1['total_volume'] = relation.sum(data=df2['Volume'].values)
```

**Key Differences:**
- `.relate()` creates `EventsRelation` object
- Use relation methods like `.count()`, `.first()`, `.mean()`, `.sum()`
- Pass data arrays explicitly to aggregation methods
- Overlay results returned as arrays/matrices for flexibility

### 5. Resegmentation

**v0.1.x:**
```python
ec_windows = ec.to_windows(
    length=0.1,
    attr=['Speed_Limit'],
    aggs=['County']
)
```

**v1.0:**
```python
df_segmented = df.lr.resegment(
    length=0.1,
    retain=['Speed_Limit', 'County']
)
```

**Key Differences:**
- `to_windows()` → `resegment()`
- Similar `retain` pattern as dissolve
- Returns DataFrame directly

### 6. Union Operations

**v0.1.x:**
```python
from linref import EventsUnion

union = EventsUnion([ec1, ec2, ec3])
ec_unified = union.union()
```

**v1.0:**
```python
import linref as lr

# Get EventsData objects
events1 = df1.lr.events
events2 = df2.lr.events
events3 = df3.lr.events

# Integrate
integrated = lr.integrate(
    [events1, events2, events3],
    fill_gaps=False
)

# Convert to DataFrame
df_unified = integrated.to_frame(
    group_name=['route'],
    beg_name='beg',
    end_name='end'
)
```

**Key Differences:**
- `EventsUnion` → `integrate()` function
- Works with `EventsData` objects
- Result needs explicit conversion to DataFrame
- More control over gap filling and location splitting

### 7. Spatial Operations

**v0.1.x:**
```python
# Project events to routes
ec.project(routes_gdf)

# Build routes
ec.build_routes()
```

**v1.0:**
```python
# Generate linear events from geometries
df = df.lr.generate_linear_events(
    beg_col='beg',
    end_col='end',
    scale=1/5280,
    add_geom_m=True
)

# Extract M-values from geometries
df = df.lr.extract_m_values(
    beg_col='beg',
    end_col='end'
)

# Add M-enabled geometries
df = df.lr.add_geom_m(name='geometry_m')
```

**Key Differences:**
- More explicit geometry operations
- Separate methods for different geometry tasks
- Better support for M-enabled geometries

## Common Migration Patterns

### Pattern 1: Basic Workflow

**v0.1.x:**
```python
# Load and create collection
df = pd.read_csv('data.csv')
ec = EventsCollection(df, keys=['Route'], beg='Begin', end='End')

# Dissolve on attributes
ec_dissolved = ec.dissolve(attr=['Type'])

# Get specific route
eg = ec_dissolved['Route 50']

# Export
result_df = eg.events
```

**v1.0:**
```python
# Load and set LRS
df = pd.read_csv('data.csv')
df = df.lr.set_lrs(
    key_col=['Route'],
    beg_col='Begin',
    end_col='End',
    closed='right'
)

# Dissolve on attributes
df_dissolved = df.lr.dissolve(retain=['Type'])

# Get specific route
route_50 = df_dissolved.lr.get_group('Route 50')

# Already a DataFrame - ready to use!
```

### Pattern 2: Overlay Analysis

**v0.1.x:**
```python
# Create collections
ec_roads = EventsCollection(roads_df, ...)
ec_crashes = EventsCollection(crashes_df, ...)

# Merge and aggregate
em = ec_roads.merge(ec_crashes)
crash_counts = em.count(col='crash_id')

# Add to dataframe
roads_df['crash_count'] = crash_counts
```

**v1.0:**
```python
# Set up LRS
roads = roads_df.lr.set_lrs(...)
crashes = crashes_df.lr.set_lrs(...)

# Create relation and count
relation = roads.lr.relate(crashes)
roads['crash_count'] = relation.count()
```

### Pattern 3: Iterating Over Groups

**v0.1.x:**
```python
for key, eg in ec:
    print(f"Processing {key}")
    df = eg.events
    # Process df
```

**v1.0:**
```python
for group, group_df in df.lr.iter_groups():
    print(f"Processing {group}")
    # Process group_df directly
```

## Benefits of the New Design

### 1. DataFrame-Native

Work directly with pandas DataFrames throughout your workflow:

```python
# v1.0 - seamless pandas integration
df = df.lr.set_lrs(...)
df = df.lr.dissolve()
df = df[df['speed'] > 50]  # Standard pandas filtering
df = df.groupby('type').agg({'length': 'sum'})  # Standard aggregation
```

### 2. More Explicit Configuration

LRS objects make schema explicit and reusable:

```python
# Define once
road_lrs = lr.LRS(
    key_col=['route', 'direction'],
    beg_col='begin_mp',
    end_col='end_mp',
    closed='left_mod'
)

# Reuse across multiple DataFrames
df1 = df1.lr.set_lrs(road_lrs)
df2 = df2.lr.set_lrs(road_lrs)
df3 = df3.lr.set_lrs(road_lrs)

# Or set as default
lr.set_default_lrs(road_lrs)
```

### 3. Better Performance

Optimized EventsData core with improved algorithms:
- Faster dissolve operations
- More efficient overlay computations
- Better memory management with chunking

### 4. Enhanced Flexibility

More control over operations:

```python
# Multiple aggregation methods
relation = df1.lr.relate(df2)

# Use specific methods for aggregation
data = df2['some_column'].values
counts = relation.count()
sums = relation.sum(data=data)
means = relation.mean(data=data)
first_vals = relation.first(data=data)
last_vals = relation.last(data=data)
```

## Compatibility Notes

### What's Removed

- `EventsCollection` class
- `EventsGroup` class  
- `EventsMerge` class
- `EventsUnion` class
- Direct indexing with `ec[key]` syntax

### What's New

- `LRS` configuration objects
- `.lr` DataFrame accessor
- `EventsData` as computational core
- `EventsRelation` for overlay operations
- Default LRS settings
- Improved geometry handling

### What's Similar

- Core linear referencing algorithms
- Closure type handling
- Group-based operations
- Dissolve and overlay concepts

## Getting Help

If you encounter issues during migration:

1. **Check the examples**: See `examples/` directory for complete workflows
2. **Read the docs**: https://linref.readthedocs.io/
3. **File an issue**: https://github.com/tariqshihadah/linref/issues
4. **Review tests**: Check `linref/tests/` for usage patterns

## Migration Checklist

- [ ] Update `EventsCollection(df, ...)` to `df.lr.set_lrs(...)`
- [ ] Replace `ec[key]` with `df.lr.get_group(key)`
- [ ] Change `ec.dissolve(attr=[...])` to `df.lr.dissolve(retain=[...])`
- [ ] Update `ec1.merge(ec2)` to `df1.lr.relate(df2)`
- [ ] Replace `em.overlay_average(...)` with relation methods (`.count()`, `.mean()`, `.sum()`, etc.)
- [ ] Change `ec.to_windows(...)` to `df.lr.resegment(...)`
- [ ] Update `EventsUnion([...])` to `df1.lr.integrate([df2, df3])`
- [ ] Review closure types (now explicit)
- [ ] Update spatial operations to new geometry methods
- [ ] Test with your data to ensure correct behavior

## Example: Complete Migration

**v0.1.x Code:**
```python
from linref import EventsCollection
import pandas as pd

# Load data
roads = pd.read_csv('roads.csv')
crashes = pd.read_csv('crashes.csv')

# Create collections
ec_roads = EventsCollection(
    roads,
    keys=['route'],
    beg='begin_mp',
    end='end_mp'
)

ec_crashes = EventsCollection(
    crashes,
    keys=['route'],
    beg='crash_mp',
    end='crash_mp'  # Point events
)

# Dissolve roads by type
ec_roads = ec_roads.dissolve(attr=['road_type'])

# Merge and count crashes
em = ec_roads.merge(ec_crashes)
crash_counts = em.count()

# Get Route 50 results
eg = ec_roads['Route 50']
route_50_df = eg.events
route_50_df['crash_count'] = crash_counts[ec_roads.events['route'] == 'Route 50']
```

**v1.0 Code:**
```python
import linref as lr
import pandas as pd
import numpy as np

# Load data and set LRS
roads = pd.read_csv('roads.csv').lr.set_lrs(
    key_col=['route'],
    beg_col='begin_mp',
    end_col='end_mp',
    closed='right'
)

crashes = pd.read_csv('crashes.csv').lr.set_lrs(
    key_col=['route'],
    loc_col='crash_mp'
)

# Dissolve roads by type
roads = roads.lr.dissolve(retain=['road_type'])

# Count crashes per road segment
relation = roads.lr.relate(crashes)
roads['crash_count'] = relation.count()

# Get Route 50 results
route_50_df = roads.lr.get_group('Route 50')
```

**Key Improvements:**
- Fewer lines of code
- Direct DataFrame manipulation
- More explicit operations
- Better pandas integration
- Easier to understand flow
