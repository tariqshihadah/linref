# Linref Test Suite

This directory contains unit tests for the `linref` package.

## Test Structure

```
tests/
├── __init__.py
├── README.md
├── test_ext_base.py      # Tests for linref.ext.base module
└── data/                 # Test data directory
```

## Running Tests

### Using unittest (standard library)

```bash
# Run all tests in the module
python -m unittest linref.tests.test_ext_base

# Run with verbose output
python -m unittest linref.tests.test_ext_base -v

# Run a specific test class
python -m unittest linref.tests.test_ext_base.TestLRSInit

# Run a specific test method
python -m unittest linref.tests.test_ext_base.TestLRSInit.test_lrs_init_linear
```

### Using pytest (if installed)

```bash
# Run all tests
pytest linref/tests/

# Run with verbose output
pytest linref/tests/ -v

# Run a specific file
pytest linref/tests/test_ext_base.py

# Run tests matching a pattern
pytest linref/tests/ -k "test_lrs"
```

## Test Coverage

### `test_ext_base.py`

Tests for the `linref.ext.base` module, covering:

#### LRS Class Tests
- **TestLRSInit**: LRS object initialization
  - Empty, linear, point, and spatial LRS creation
  - Multiple key columns
  - Invalid parameter handling
  
- **TestLRSProperties**: LRS properties and methods
  - Property checks (`is_linear`, `is_point`, `is_grouped`, etc.)
  - Parameter management (`set_params`, `add_key`, `remove_key`)
  - Copying and equality comparison
  - Study method for validating DataFrames

#### LRS_Accessor Class Tests
- **TestLRSAccessorInit**: Accessor initialization
  - Accessor availability on DataFrames
  - Setting and clearing LRS
  - LRS configuration via kwargs
  
- **TestLRSAccessorProperties**: Accessor properties
  - Column name access
  - Data extraction (keys, begs, ends, locs, geoms)
  - Event validation
  - Event lengths calculation
  
- **TestLRSAccessorMethods**: Accessor methods
  - LRS modification
  - Key column management
  - Group selection and iteration
  - Standard sorting

#### Event Operations Tests
- **TestEventOperations**: Event manipulation
  - Extending and shifting events
  - Rounding event locations
  - Converting point to linear events
  - Dissolving consecutive events
  - Resegmenting events

#### Utility Tests
- **TestCompatibilityFunctions**: Compatibility checking
  - DataFrame LRS compatibility validation
  - Error handling for incompatible LRS configurations

#### Configuration Tests
- **TestDefaultLRS**: Default LRS settings
  - Setting and clearing default LRS
  
- **TestGeometrySyncBehavior**: Geometry synchronization
  - Sync behavior configuration
  - Invalid parameter handling
  
- **TestStudyMethod**: LRS study functionality
  - Complete LRS validation
  - Missing column detection

## Adding New Tests

When adding new tests to this suite:

1. **Follow the naming convention**: `test_<module_name>.py`
2. **Use descriptive test names**: Start with `test_` and describe what's being tested
3. **Organize by class**: Group related tests into `TestCase` classes
4. **Add docstrings**: Briefly describe what each test validates
5. **Use setUp/tearDown**: For common test data preparation and cleanup
6. **Test edge cases**: Include tests for error conditions and boundary cases

### Example Test Structure

```python
class TestNewFeature(unittest.TestCase):
    """Test suite for new feature."""
    
    def setUp(self):
        """Prepare test fixtures."""
        self.test_data = ...
    
    def test_basic_functionality(self):
        """Test that basic feature works as expected."""
        result = feature(self.test_data)
        self.assertEqual(result, expected)
    
    def test_error_handling(self):
        """Test that invalid input raises appropriate error."""
        with self.assertRaises(ValueError):
            feature(invalid_input)
    
    def tearDown(self):
        """Clean up after tests."""
        pass
```

## Test Data

Test data files should be placed in the `tests/data/` directory. Use CSV, JSON, or GeoJSON formats for event and geometry data.

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    python -m unittest discover -s linref/tests -p "test_*.py"
```

## Contributing

When contributing new features to linref:

1. Write tests before implementing the feature (TDD approach)
2. Ensure all existing tests still pass
3. Aim for high test coverage of new code
4. Update this README if adding new test modules
