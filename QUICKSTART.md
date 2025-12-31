# Linref Quick Start Guide

Get up and running with linref in 5 minutes!

## Installation

```bash
pip install linref
```

## Your First Linear Referencing Workflow

### Step 1: Import and Create Data

```python
import linref as lr
import pandas as pd

# Create a simple dataset with route events
df = pd.DataFrame({
    'route': ['I-80', 'I-80', 'I-80', 'I-5', 'I-5'],
    'begin_mp': [0.0, 1.5, 3.0, 0.0, 2.0],
    'end_mp': [1.5, 3.0, 5.0, 2.0, 4.0],
    'speed_limit': [65, 55, 65, 70, 55],
    'lanes': [4, 3, 4, 4, 3]
})

print(df)
```

### Step 2: Set Up Linear Referencing

```python
# Configure the Linear Referencing System (LRS)
df = df.lr.set_lrs(
    key_col=['route'],      # Route identifier
    beg_col='begin_mp',     # Begin milepost
    end_col='end_mp',       # End milepost
    closed='right'          # How to handle endpoints
)

# Check that LRS is configured
print(df.lr.is_linear)  # True
print(df.lr.is_grouped)  # True
```

### Step 3: Dissolve Consecutive Segments

```python
# Merge consecutive segments with the same attributes
dissolved = df.lr.dissolve(retain=['speed_limit', 'lanes'])

print(dissolved)
# Result: Consecutive segments with same speed/lanes are merged
```

### Step 4: Query by Location

```python
# Get events at specific locations
events = df.lr.events

# Find events that intersect mile marker 1.0 to 3.5
mask = events.intersect_range(beg=1.0, end=3.5)
intersecting = df[mask]

print(intersecting)
```

### Step 5: Work with Groups

```python
# Get all events for a specific route
i80_events = df.lr.get_group('I-80')
print(f"I-80 has {len(i80_events)} events")

# Iterate over all routes
for route, route_df in df.lr.iter_groups():
    total_length = (route_df['end_mp'] - route_df['begin_mp']).sum()
    print(f"{route}: {total_length} miles")
```

## Common Tasks

### Overlay Two Datasets

```python
# Two datasets with matching LRS
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

# Count crashes on each road segment
relation = roads.lr.relate(crashes)
roads['num_crashes'] = relation.count()
```

### Create Uniform Segments

```python
# Create 0.5-mile segments
segmented = df.lr.resegment(length=0.5)

print(segmented)
# All events are now exactly 0.5 miles long
```

### Work with Point Events

```python
# Point-based data (signs, milemarkers, etc.)
signs = pd.DataFrame({
    'route': ['I-80', 'I-80', 'I-5'],
    'milepost': [1.2, 3.5, 1.8],
    'sign_type': ['Speed Limit', 'Exit', 'Rest Area']
}).lr.set_lrs(
    key_col=['route'],
    loc_col='milepost'
)

# Add a buffer around each point
buffered = signs.lr.extend(
    extend_begs=0.1,  # 0.1 miles before
    extend_ends=0.1   # 0.1 miles after
)

print(buffered)
# Points are now 0.2-mile linear segments
```

### Generate Geometries

```python
import geopandas as gpd
from shapely.geometry import LineString

# Reference routes with geometries
ref_routes = gpd.GeoDataFrame({
    'route': ['I-80', 'I-5'],
    'geometry': [
        LineString([(0,0), (5,0)]),
        LineString([(0,2), (4,2)])
    ]
}).lr.set_lrs(
    key_col=['route'],
    geom_col='geometry'
)

# Generate mile markers from geometry lengths
ref_routes = ref_routes.lr.generate_linear_events(
    beg_col='beg',
    end_col='end',
    scale=1.0,  # Use as-is (or 1/5280 for feet to miles)
    add_geom_m=True  # Create M-enabled geometries
)

print(ref_routes)
```

## Best Practices

### 1. Set a Default LRS

If all your data uses the same schema:

```python
# Define once at the start of your script
default_lrs = lr.LRS(
    key_col=['route', 'direction'],
    beg_col='begin_mp',
    end_col='end_mp',
    closed='left_mod'
)

lr.set_default_lrs(default_lrs)

# All DataFrames automatically have this LRS
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')
# Both already configured!
```

### 2. Check Your Data

```python
# Verify LRS configuration
print(df.lr.study())

# Check for invalid events
if df.lr.invalid_events.any():
    print("Warning: Invalid events found!")
    df = df.lr.drop_invalid_events()
```

### 3. Sort Before Processing

```python
# Sort for better performance
df = df.lr.sort_standard()
```

### 4. Use Method Chaining

```python
# Chain operations together
result = (df
    .lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end')
    .lr.dissolve(retain=['type'])
    .lr.resegment(length=0.1)
)
```

## Next Steps

- **Read the full usage guide**: `USAGE_GUIDE.md`
- **Explore examples**: Check the `examples/` directory
- **Read the docs**: https://linref.readthedocs.io/
- **Migrate from v0.1.x**: See `MIGRATION_GUIDE.md`

## Quick Reference

### Set Up LRS
```python
df = df.lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end')
```

### Common Operations
```python
dissolved = df.lr.dissolve(retain=['attr'])
segmented = df.lr.resegment(length=0.1)
sorted_df = df.lr.sort_standard()
group_df = df.lr.get_group('Route A')
```

### Query by Location
```python
events = df.lr.events
mask = events.intersect_range(beg=0.5, end=1.5)
results = df[mask]
```

### Overlay Operations
```python
relation = df1.lr.events.relate(df2.lr.events)
overlay = relation.overlay()
aggregated = relation.aggregate(data, method='sum')
```

### Geometry Operations
```python
df = df.lr.generate_linear_events()  # From geometries
df = df.lr.extract_m_values()        # From M-geometries
df = df.lr.add_geom_m()              # Add M-geometries
```

## Troubleshooting

### "No LRS set"
Make sure you've called `.set_lrs()` on your DataFrame:
```python
df = df.lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end')
```

### "LRS not compatible"
Both DataFrames in an overlay must have compatible LRS (same keys):
```python
# Both need matching key columns
df1 = df1.lr.set_lrs(key_col=['route'], ...)
df2 = df2.lr.set_lrs(key_col=['route'], ...)  # Same key!
```

### "Invalid events"
Some events have missing data:
```python
# Find and remove invalid events
df = df.lr.drop_invalid_events()
```

### Performance Issues
Try sorting and using chunking:
```python
df = df.lr.sort_standard()
relation = df1.lr.events.relate(df2.lr.events)
result = relation.overlay(chunksize=500)
```

## Need Help?

- **Documentation**: https://linref.readthedocs.io/
- **Issues**: https://github.com/tariqshihadah/linref/issues
- **Examples**: See `examples/` directory in the repository
