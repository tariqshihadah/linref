# Linref Toy Datasets - Simplified API

## Quick Start

```python
import linref as lr

# Load any dataset with a single function
roads = lr.datasets.load('roadways')
crashes = lr.datasets.load('crashes')
pavement = lr.datasets.load('pavement')

# All datasets come with LRS pre-configured
dissolved = roads.lr.dissolve()
relation = roads.lr.relate(crashes)
```

## API

### `load(name)`

Load a toy dataset with LRS already configured.

**Parameters:**
- `name` (str): Dataset name - `'roadways'`, `'crashes'`, or `'pavement'`

**Returns:**
- GeoDataFrame (for roadways, crashes) or DataFrame (for pavement)
- LRS automatically configured, ready to use

**Examples:**

```python
# Load datasets
roads = lr.datasets.load('roadways')      # 10 segments, with geometry
crashes = lr.datasets.load('crashes')      # 20 points, with geometry  
pavement = lr.datasets.load('pavement')    # 14 segments, no geometry

# Use immediately
dissolved = roads.lr.dissolve(retain=['speed_limit'])
relation = roads.lr.relate(crashes)
roads['crash_count'] = relation.count()
```

### `list_datasets()`

Show all available datasets with descriptions.

```python
print(lr.datasets.list_datasets())
```

## Available Datasets

### 1. **roadways** - Linear Events with Geometry
- 10 segments across 3 routes (US-101, SR-1, I-5)
- Attributes: route, beg, end, traffic_volume, speed_limit
- LineString geometries included
- LRS: `key_col=['route'], beg_col='beg', end_col='end', geom_col='geometry'`

### 2. **crashes** - Point Events with Geometry
- 20 crash records
- Attributes: crash_id, route, location, severity, mode
- Point geometries included
- LRS: `key_col=['route'], loc_col='location', geom_col='geometry'`

### 3. **pavement** - Linear Events (no geometry)
- 14 segments across 3 routes
- Attributes: route, beg, end, condition_rating, surface_type
- No geometry (tabular data only)
- LRS: `key_col=['route'], beg_col='beg', end_col='end'`

## Data Files

All data stored in `linref/datasets/_data/`:
- `roadways.geojson` - GeoJSON with LineStrings
- `crashes.geojson` - GeoJSON with Points
- `pavement.csv` - CSV without geometry
