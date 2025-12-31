# Contributing to Linref

Thank you for your interest in contributing to linref! This guide will help you understand the project structure and contribute effectively.

## Architecture Overview

Linref v1.0 (redesign branch) uses a three-tier architecture:

### 1. LRS (Linear Referencing System)
- **Location**: `linref/ext/base.py` - `LRS` class
- **Purpose**: Schema definition objects that specify column mappings
- **Key attributes**: `key_col`, `loc_col`, `beg_col`, `end_col`, `geom_col`, `closed`
- **Immutable operations**: Returns new LRS objects instead of modifying in place (unless `inplace=True`)

### 2. EventsData
- **Location**: `linref/events/base.py` - `EventsData` class
- **Purpose**: Core computational engine for linear referencing operations
- **Key operations**: Available in `linref/events/` modules:
  - `modify.py` - dissolve, extend, shift, round, resegment
  - `relate.py` - overlay, intersect, relationships
  - `selection.py` - filtering and querying
  - `analyze.py` - analysis operations
  - `geometry.py` - spatial operations
  - `integration.py` - combining datasets
- **Data model**: Works with numpy arrays for performance

### 3. DataFrame Accessor (.lr)
- **Location**: `linref/ext/base.py` - `LRS_Accessor` class
- **Purpose**: User-facing pandas DataFrame extension
- **Pattern**: Uses pandas `@register_dataframe_accessor` decorator
- **Bridge**: Converts between DataFrames and EventsData

## Project Structure

```
linref/
├── __init__.py                 # Public API exports
├── errors.py                   # Custom exceptions
├── events/                     # EventsData core
│   ├── base.py                # EventsData class
│   ├── modify.py              # Modification operations
│   ├── relate.py              # Relationship operations
│   ├── selection.py           # Selection operations
│   ├── analyze.py             # Analysis operations
│   ├── geometry.py            # Geometry operations
│   ├── integration.py         # Integration operations
│   ├── common.py              # Common utilities
│   └── utility.py             # Helper functions
├── ext/                        # Extensions and accessor
│   ├── base.py                # LRS and LRS_Accessor
│   ├── validation.py          # Validation decorators
│   ├── spatial.py             # Spatial extensions
│   └── default.py             # Default settings
├── utility/                    # Utilities
│   ├── direction.py           # Direction extraction
│   └── utility.py             # General utilities
└── tests/                      # Test suite
    ├── test_events_*.py       # EventsData tests
    └── test_ext_*.py          # LRS_Accessor tests
```

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tariqshihadah/linref.git
   cd linref
   git checkout redesign
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .
   pip install pytest pytest-cov  # For testing
   ```

4. **Run tests**:
   ```bash
   pytest linref/tests/
   ```

## Code Style Guidelines

### General Principles
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public methods
- Keep methods focused and single-purpose
- Prefer composition over inheritance

### Naming Conventions

**DataFrame columns (parameter names)**:
- Use `_col` suffix: `key_col`, `beg_col`, `end_col`, `geom_col`
- Can be single label or list: `key_col=['route']` or `key_col=['route', 'direction']`

**EventsData attributes (no suffix)**:
- Direct access to data: `events.keys`, `events.begs`, `events.ends`
- Always returns numpy arrays

**Method patterns**:
- `get_*()` - Retrieve data
- `set_*()` - Configure settings
- `add_*()` - Add columns/features
- `*_like()` - Copy configuration
- `build_*()` - Generate complex objects

### Parameter Patterns

**Standard parameters**:
```python
def method(
    self,
    # Required positional arguments
    required_arg,
    # Optional arguments with defaults
    optional_arg=None,
    # Boolean flags (default False unless otherwise logical)
    inplace: bool = False,
    replace: bool = False,
    # Output control
    return_index: bool = False,
    return_relation: bool = False
) -> pd.DataFrame | None:
```

**`inplace` parameter**:
- When `True`: Modifies object in place, returns `None`
- When `False` (default): Returns new object, original unchanged
- Always document return type clearly

## Adding New Features

### 1. Adding a New EventsData Method

**Location**: `linref/events/<appropriate_module>.py`

```python
# In linref/events/modify.py
def new_operation(events, param1, param2=None, inplace=False):
    """
    Short description.
    
    Parameters
    ----------
    events : EventsData
        Input events data object.
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2.
    inplace : bool, default False
        Whether to modify events in place.
        
    Returns
    -------
    EventsData or None
        Modified events if inplace=False, else None.
    """
    # Validate input
    if not isinstance(events, base.EventsData):
        raise TypeError("Input must be EventsData instance.")
    
    # Select object to modify
    events = events if inplace else events.copy()
    
    # Perform operation
    # ... modify events._begs, events._ends, etc.
    
    # Return
    return None if inplace else events
```

**Then add to EventsData class** in `linref/events/base.py`:

```python
def new_operation(self, param1, param2=None, inplace=False):
    """Wrapper for module-level function."""
    return modify.new_operation(
        self, 
        param1=param1, 
        param2=param2, 
        inplace=inplace
    )
```

### 2. Adding a New DataFrame Accessor Method

**Location**: `linref/ext/base.py` - in `LRS_Accessor` class

```python
@_method_require(is_linear=True)  # Add requirements as needed
def new_method(
    self,
    param1,
    param2=None,
    inplace: bool = False
) -> pd.DataFrame | None:
    """
    Short description.
    
    Parameters
    ----------
    param1 : type
        Description.
    param2 : type, optional
        Description.
    inplace : bool, default False
        Whether to apply changes in place.
        
    Returns
    -------
    DataFrame or None
        Modified DataFrame if inplace=False, else None.
    """
    # Get events data
    events = self.get_events()
    
    # Perform operation
    result_events = events.new_operation(param1, param2)
    
    # Apply to DataFrame
    df = self.df if inplace else self.copy_df()
    df[self.beg_col] = result_events.begs
    df[self.end_col] = result_events.ends
    
    return None if inplace else df
```

### 3. Adding Tests

**Location**: `linref/tests/test_<module>.py`

```python
import unittest
import pandas as pd
import linref as lr
from linref.events import EventsData

class TestNewFeature(unittest.TestCase):
    """Test the new feature."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'route': ['A', 'A', 'B'],
            'beg': [0.0, 1.0, 0.0],
            'end': [1.0, 2.0, 1.5]
        }).lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end'
        )
        
    def test_basic_operation(self):
        """Test basic operation."""
        result = self.df.lr.new_method(param1='value')
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        # More specific assertions
        
    def test_inplace_operation(self):
        """Test inplace operation."""
        df_copy = self.df.copy()
        result = df_copy.lr.new_method(param1='value', inplace=True)
        # Assertions
        self.assertIsNone(result)
        # Check df_copy was modified
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty DataFrame
        # Invalid parameters
        # etc.

if __name__ == '__main__':
    unittest.main()
```

## Validation Decorators

Use decorators to validate preconditions:

```python
from linref.events.utility import _method_require
from linref.ext.validation import _method_deprecates_geometry

# For EventsData methods
@_method_require(is_linear=True, is_empty=False)
def method(self, ...):
    pass

# For DataFrame accessor methods
@_method_require(is_linear=True)  # Checks self.is_linear
def method(self, ...):
    pass

@_method_deprecates_geometry  # Warns if geometries present
def method(self, ...):
    pass
```

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def method(self, param1, param2=None, inplace=False):
    """
    One-line summary.
    
    More detailed description if needed. Can span multiple lines.
    Explain what the method does, when to use it, and any important
    behavior or limitations.
    
    Parameters
    ----------
    param1 : type
        Description of param1. Include valid values, units, or
        constraints.
    param2 : type, optional
        Description of optional param2.
    inplace : bool, default False
        Whether to modify in place. If True, returns None.
        
    Returns
    -------
    return_type or None
        Description of return value. Explain what is returned
        when inplace=False vs inplace=True.
        
    Raises
    ------
    ExceptionType
        When and why this exception is raised.
        
    See Also
    --------
    related_method : Related functionality.
    
    Examples
    --------
    >>> df = df.lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end')
    >>> result = df.lr.method(param1='value')
    """
```

### Example Code

- Always include working examples in docstrings
- Examples should be self-contained
- Use realistic parameter values
- Show both simple and advanced usage

## Testing Guidelines

### Test Coverage

Aim for comprehensive coverage:
- Normal operations
- Edge cases (empty data, single event, etc.)
- Error handling (invalid parameters, incompatible data)
- Inplace vs. copy operations
- Different LRS configurations

### Test Data

Use simple, predictable test data:

```python
def setUp(self):
    # Simple linear events
    self.df = pd.DataFrame({
        'route': ['A', 'A', 'B', 'B'],
        'beg': [0.0, 1.0, 0.0, 2.0],
        'end': [1.0, 2.0, 2.0, 4.0],
        'value': [10, 20, 30, 40]
    }).lr.set_lrs(...)
```

### Assertion Patterns

```python
# Type checks
self.assertIsInstance(result, pd.DataFrame)
self.assertIsNone(result)  # For inplace operations

# Value checks
self.assertEqual(result['beg'].tolist(), [0.0, 2.0])
self.assertTrue((result['value'] == 10).all())

# Shape checks
self.assertEqual(len(result), 2)
self.assertEqual(result.shape[1], 4)

# LRS checks
self.assertTrue(result.lr.is_linear)
self.assertEqual(result.lr.key_col, ['route'])
```

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following style guidelines
   - Add tests for new features
   - Update documentation

3. **Run tests**:
   ```bash
   pytest linref/tests/
   ```

4. **Update documentation**:
   - Update relevant docstrings
   - Add examples if adding new methods
   - Update README or guides if needed

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub

## Common Patterns

### Working with EventsData

```python
# Get EventsData from DataFrame
events = df.lr.events

# Check properties
if events.is_linear and not events.is_empty:
    # Safe to proceed
    pass

# Iterate over groups
for group, group_events in events.iter_groups():
    # Process each group
    pass

# Convert back to DataFrame
new_df = events.to_frame(
    group_name=['route'],
    beg_name='beg',
    end_name='end'
)
```

### Handling Geometry Synchronization

Methods that modify begin/end values should warn about geometry desync:

```python
from linref.ext.validation import _method_deprecates_geometry

@_method_deprecates_geometry
def shift(self, shift, inplace=False):
    # This will warn users that geometries may become invalid
    pass
```

### Error Handling

```python
from linref.errors import LRSConfigurationError, LRSCompatibilityError

# Configuration errors
if not self.is_lrs_set:
    raise LRSConfigurationError("No LRS configured for DataFrame")

# Compatibility errors
if not df1.lr.is_grouped or not df2.lr.is_grouped:
    raise LRSCompatibilityError("Both DataFrames must have grouped LRS")

# Validation errors
if len(self.key_col) != len(other.lr.key_col):
    raise LRSCompatibilityError(
        f"Key column count mismatch: {len(self.key_col)} vs "
        f"{len(other.lr.key_col)}"
    )
```

## Questions or Issues?

- **Documentation**: Check `USAGE_GUIDE.md` and existing code
- **Examples**: See `examples/` directory
- **Tests**: Review `linref/tests/` for patterns
- **Issues**: Open an issue on GitHub for questions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE.rst).
