"""
Unit tests for linref.ext.base module.

Tests cover:
- LRS class initialization and configuration
- LRS_Accessor class and DataFrame integration
- Event operations (dissolve, resegment, integrate, etc.)
- Geometry operations (cutting, interpolation, projection)
- Relational operations between DataFrames
"""

import unittest
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
# Import from main package to ensure accessor registration
import linref
from linref import LRS, LRS_Accessor, integrate
from linref.ext.base import test_compatibility
from linref.errors import LRSConfigurationError, LRSCompatibilityError


class TestLRSInit(unittest.TestCase):
    """Test LRS class initialization and basic properties."""

    def test_lrs_init_empty(self):
        """Test creating an empty LRS object with minimal parameters."""
        lrs = LRS()
        self.assertIsInstance(lrs, LRS)
        self.assertTrue(len(lrs.key_col) == 0)
        self.assertIsNone(lrs.loc_col)
        self.assertIsNone(lrs.beg_col)
        self.assertIsNone(lrs.end_col)
        self.assertIsNone(lrs.geom_col)
        self.assertIsNone(lrs.geom_m_col)
        self.assertIsNone(lrs.closed)

    def test_lrs_init_linear(self):
        """Test creating a linear LRS object."""
        lrs = LRS(
            key_col=['route_id'],
            beg_col='begin_mp',
            end_col='end_mp',
            closed='right'
        )
        self.assertIsInstance(lrs, LRS)
        self.assertEqual(lrs.key_col, ['route_id'])
        self.assertEqual(lrs.beg_col, 'begin_mp')
        self.assertEqual(lrs.end_col, 'end_mp')
        self.assertEqual(lrs.closed, 'right')
        self.assertTrue(lrs.is_linear)
        self.assertFalse(lrs.is_point)

    def test_lrs_init_point(self):
        """Test creating a point LRS object."""
        lrs = LRS(
            key_col=['route_id'],
            loc_col='milepost'
        )
        self.assertIsInstance(lrs, LRS)
        self.assertEqual(lrs.key_col, ['route_id'])
        self.assertEqual(lrs.loc_col, 'milepost')
        self.assertTrue(lrs.is_point)
        self.assertTrue(lrs.is_located)
        self.assertFalse(lrs.is_linear)
        self.assertIsNone(lrs.closed)

    def test_lrs_init_spatial(self):
        """Test creating a spatial LRS object."""
        lrs = LRS(
            key_col=['route_id'],
            beg_col='begin_mp',
            end_col='end_mp',
            geom_col='geometry',
            geom_m_col='geometry_m',
            closed='right'
        )
        self.assertTrue(lrs.is_spatial)
        self.assertTrue(lrs.is_spatial_m)

    def test_lrs_init_multiple_keys(self):
        """Test creating LRS with multiple key columns."""
        lrs = LRS(key_col=['route_id', 'direction', 'year'])
        self.assertEqual(len(lrs.key_col), 3)
        self.assertTrue(lrs.is_grouped)

    def test_lrs_init_invalid_closed(self):
        """Test that invalid closed parameter raises error."""
        with self.assertRaises(LRSConfigurationError):
            LRS(closed='invalid_value')
        with self.assertRaises(ValueError):
            LRS(
                key_col=['route'],
                beg_col='beg',
                end_col='end',
                closed='invalid_value'
            )


class TestLRSProperties(unittest.TestCase):
    """Test LRS class properties and methods."""

    def setUp(self):
        """Set up common LRS objects for testing."""
        self.lrs_linear = LRS(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='right'
        )
        self.lrs_point = LRS(
            key_col=['route'],
            loc_col='loc'
        )
        self.lrs_spatial = LRS(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            geom_col='geometry',
            geom_m_col='geometry_m',
            closed='right'
        )

    def test_lrs_is_properties(self):
        """Test LRS is_* properties."""
        # Linear LRS
        self.assertTrue(self.lrs_linear.is_linear)
        self.assertFalse(self.lrs_linear.is_point)
        self.assertTrue(self.lrs_linear.is_grouped)
        self.assertFalse(self.lrs_linear.is_spatial)
        
        # Point LRS
        self.assertFalse(self.lrs_point.is_linear)
        self.assertTrue(self.lrs_point.is_point)
        self.assertTrue(self.lrs_point.is_located)
        
        # Spatial LRS
        self.assertTrue(self.lrs_spatial.is_spatial)
        self.assertTrue(self.lrs_spatial.is_spatial_m)

    def test_lrs_params_property(self):
        """Test that params property returns correct dictionary."""
        params = self.lrs_linear.params
        self.assertIsInstance(params, dict)
        self.assertEqual(params['key_col'], ['route'])
        self.assertEqual(params['beg_col'], 'beg')
        self.assertEqual(params['end_col'], 'end')
        self.assertEqual(params['closed'], 'right')

    def test_lrs_copy(self):
        """Test copying LRS objects."""
        lrs_copy = self.lrs_linear.copy()
        self.assertEqual(lrs_copy.params, self.lrs_linear.params)
        self.assertIsNot(lrs_copy, self.lrs_linear)
        
        # Deep copy
        lrs_deep = self.lrs_linear.copy(deep=True)
        self.assertEqual(lrs_deep.params, self.lrs_linear.params)

    def test_lrs_set_params(self):
        """Test setting LRS parameters."""
        lrs = self.lrs_linear.copy()
        lrs_modified = lrs.set_params(beg_col='start', end_col='finish', inplace=False)
        
        # Original unchanged
        self.assertEqual(lrs.beg_col, 'beg')
        self.assertEqual(lrs.end_col, 'end')
        
        # New object modified
        self.assertEqual(lrs_modified.beg_col, 'start')
        self.assertEqual(lrs_modified.end_col, 'finish')

    def test_lrs_set_params_inplace(self):
        """Test setting LRS parameters in place."""
        lrs = self.lrs_linear.copy()
        result = lrs.set_params(beg_col='start', inplace=True)
        
        self.assertIsNone(result)
        self.assertEqual(lrs.beg_col, 'start')

    def test_lrs_add_key(self):
        """Test adding key columns to LRS."""
        lrs = self.lrs_linear.copy()
        lrs_modified = lrs.add_key('year', inplace=False)
        
        self.assertEqual(len(lrs.key_col), 1)
        self.assertEqual(len(lrs_modified.key_col), 2)
        self.assertIn('year', lrs_modified.key_col)

    def test_lrs_remove_key(self):
        """Test removing key columns from LRS."""
        lrs = LRS(key_col=['route', 'direction', 'year'])
        lrs_modified = lrs.remove_key('direction', inplace=False)
        
        self.assertEqual(len(lrs_modified.key_col), 2)
        self.assertNotIn('direction', lrs_modified.key_col)

    def test_lrs_remove_key_missing_raises(self):
        """Test that removing non-existent key raises error."""
        lrs = self.lrs_linear.copy()
        with self.assertRaises(KeyError):
            lrs.remove_key('nonexistent', errors='raise', inplace=False)

    def test_lrs_remove_key_missing_ignore(self):
        """Test that removing non-existent key with errors='ignore' works."""
        lrs = self.lrs_linear.copy()
        lrs_modified = lrs.remove_key('nonexistent', errors='ignore', inplace=False)
        self.assertEqual(lrs_modified.key_col, lrs.key_col)

    def test_lrs_equality(self):
        """Test LRS equality comparison."""
        lrs1 = LRS(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        lrs2 = LRS(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        lrs3 = LRS(key_col=['route'], beg_col='start', end_col='end', closed='right')
        
        self.assertEqual(lrs1, lrs2)
        self.assertNotEqual(lrs1, lrs3)
        self.assertNotEqual(lrs1, "not an LRS")

    def test_lrs_str_repr(self):
        """Test LRS string representation."""
        lrs = self.lrs_linear
        lrs_str = str(lrs)
        self.assertIn('LRS(', lrs_str)
        self.assertIn('key_col=', lrs_str)
        self.assertIn('beg_col=', lrs_str)
        
        lrs_repr = repr(lrs)
        self.assertEqual(lrs_str, lrs_repr)

    def test_lrs_study(self):
        """Test LRS study method on DataFrame."""
        df = pd.DataFrame({
            'route': ['A', 'B'],
            'beg': [0, 1],
            'end': [1, 2]
        })
        
        study = self.lrs_linear.study(df)
        
        self.assertIsInstance(study, dict)
        self.assertTrue(study['keys']['valid'])
        self.assertTrue(study['linear']['valid'])
        self.assertFalse(study['geometry']['valid'])


class TestLRSAccessorInit(unittest.TestCase):
    """Test LRS_Accessor initialization and DataFrame integration."""

    def test_accessor_exists(self):
        """Test that .lr accessor is available on DataFrames."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        self.assertTrue(hasattr(df, 'lr'))
        self.assertIsInstance(df.lr, LRS_Accessor)

    def test_accessor_no_lrs_set(self):
        """Test accessor behavior when no LRS is set."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        self.assertTrue(df.lr.is_lrs_empty)

    def test_set_lrs(self):
        """Test setting LRS on DataFrame."""
        df = pd.DataFrame({
            'route': ['A', 'B'],
            'beg': [0, 1],
            'end': [1, 2]
        })
        lrs = LRS(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        df_with_lrs = df.lr.set_lrs(lrs, inplace=False)
        
        self.assertFalse(df_with_lrs.lr.is_lrs_empty)
        self.assertEqual(df_with_lrs.lr.lrs, lrs)
        # Original unchanged
        self.assertTrue(df.lr.is_lrs_empty)

    def test_set_lrs_inplace(self):
        """Test setting LRS in place."""
        df = pd.DataFrame({
            'route': ['A', 'B'],
            'beg': [0, 1],
            'end': [1, 2]
        })
        lrs = LRS(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        result = df.lr.set_lrs(lrs, inplace=True)
        
        self.assertIsNone(result)
        self.assertTrue(df.lr.is_lrs_set)
        self.assertEqual(df.lr.lrs, lrs)

    def test_set_lrs_kwargs(self):
        """Test setting LRS using keyword arguments."""
        df = pd.DataFrame({
            'route': ['A', 'B'],
            'beg': [0, 1],
            'end': [1, 2]
        })
        df_with_lrs = df.lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='right'
        )
        
        self.assertTrue(df_with_lrs.lr.is_lrs_set)
        self.assertTrue(df_with_lrs.lr.is_linear)

    def test_clear_lrs(self):
        """Test clearing LRS from DataFrame."""
        df = pd.DataFrame({'route': ['A'], 'beg': [0], 'end': [1]})
        df = df.lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        
        self.assertTrue(df.lr.is_lrs_set)
        
        df_cleared = df.lr.clear_lrs(inplace=False)
        self.assertTrue(df_cleared.lr.is_lrs_empty)


class TestLRSAccessorProperties(unittest.TestCase):
    """Test LRS_Accessor properties."""

    def setUp(self):
        """Set up test DataFrames."""
        self.df_linear = pd.DataFrame({
            'route': ['A', 'A', 'B', 'B'],
            'beg': [0.0, 1.0, 0.0, 2.0],
            'end': [1.0, 2.0, 2.0, 4.0],
            'attr': ['x', 'y', 'z', 'w']
        }).lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='right'
        )
        
        self.df_point = pd.DataFrame({
            'route': ['A', 'A', 'B'],
            'loc': [0.5, 1.5, 1.0],
            'attr': ['x', 'y', 'z']
        }).lr.set_lrs(
            key_col=['route'],
            loc_col='loc'
        )

    def test_accessor_column_properties(self):
        """Test column name properties."""
        self.assertEqual(self.df_linear.lr.key_col, ['route'])
        self.assertEqual(self.df_linear.lr.beg_col, 'beg')
        self.assertEqual(self.df_linear.lr.end_col, 'end')
        self.assertEqual(self.df_linear.lr.closed, 'right')
        
        self.assertEqual(self.df_point.lr.loc_col, 'loc')

    def test_lrs_cols_property(self):
        """Test lrs_cols property."""
        lrs_cols = self.df_linear.lr.lrs_cols
        self.assertIn('route', lrs_cols)
        self.assertIn('beg', lrs_cols)
        self.assertIn('end', lrs_cols)
        self.assertNotIn('attr', lrs_cols)

    def test_other_cols_property(self):
        """Test other_cols property."""
        other_cols = self.df_linear.lr.other_cols
        self.assertIn('attr', other_cols)
        self.assertNotIn('route', other_cols)
        self.assertNotIn('beg', other_cols)

    def test_is_properties(self):
        """Test is_* properties on accessor."""
        # Linear DataFrame
        self.assertTrue(self.df_linear.lr.is_linear)
        self.assertTrue(self.df_linear.lr.is_grouped)
        self.assertFalse(self.df_linear.lr.is_point)
        
        # Point DataFrame
        self.assertTrue(self.df_point.lr.is_point)
        self.assertTrue(self.df_point.lr.is_located)
        self.assertFalse(self.df_point.lr.is_linear)

    def test_data_properties(self):
        """Test data extraction properties."""
        # Keys
        keys = self.df_linear.lr.keys
        self.assertEqual(len(keys), 4)
        
        # Begins and ends
        begs = self.df_linear.lr.begs
        ends = self.df_linear.lr.ends
        np.testing.assert_array_equal(begs, [0.0, 1.0, 0.0, 2.0])
        np.testing.assert_array_equal(ends, [1.0, 2.0, 2.0, 4.0])
        
        # Locations
        locs = self.df_point.lr.locs
        np.testing.assert_array_equal(locs, [0.5, 1.5, 1.0])

    def test_event_lengths(self):
        """Test event_lengths property."""
        lengths = self.df_linear.lr.event_lengths
        np.testing.assert_array_equal(lengths, [1.0, 1.0, 2.0, 2.0])

    def test_valid_events(self):
        """Test valid_events property."""
        df = pd.DataFrame({
            'route': ['A', 'A', None, 'B'],
            'beg': [0.0, 1.0, 2.0, None],
            'end': [1.0, 2.0, 3.0, 4.0]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        
        valid = df.lr.valid_events
        self.assertTrue(valid.iloc[0])
        self.assertTrue(valid.iloc[1])
        self.assertFalse(valid.iloc[2])  # Missing key
        self.assertFalse(valid.iloc[3])  # Missing begin

    def test_accessor_str_repr(self):
        """Test accessor string representation."""
        accessor_str = str(self.df_linear.lr)
        self.assertIn('LRS_Accessor', accessor_str)
        self.assertIn('GR', accessor_str)  # Grouped
        self.assertIn('LN', accessor_str)  # Linear


class TestLRSAccessorMethods(unittest.TestCase):
    """Test LRS_Accessor methods."""

    def setUp(self):
        """Set up test DataFrames."""
        self.df = pd.DataFrame({
            'route': ['A', 'A', 'B', 'B'],
            'beg': [0.0, 1.0, 0.0, 2.0],
            'end': [1.0, 2.0, 2.0, 4.0],
            'attr': [10, 20, 30, 40]
        }).lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='right'
        )

    def test_modify_lrs(self):
        """Test modifying LRS parameters."""
        df_modified = self.df.lr.modify_lrs(beg_col='start', end_col='finish', inplace=False)
        
        self.assertEqual(df_modified.lr.beg_col, 'start')
        self.assertEqual(df_modified.lr.end_col, 'finish')
        # Original unchanged
        self.assertEqual(self.df.lr.beg_col, 'beg')

    def test_add_key(self):
        """Test adding key column to accessor."""
        df_modified = self.df.lr.add_key('year', inplace=False)
        
        self.assertIn('year', df_modified.lr.key_col)
        self.assertNotIn('year', self.df.lr.key_col)

    def test_remove_key(self):
        """Test removing key column from accessor."""
        df = self.df.lr.add_key(['dir', 'year'], inplace=False)
        df_modified = df.lr.remove_key('dir', inplace=False)
        
        self.assertNotIn('dir', df_modified.lr.key_col)
        self.assertIn('year', df_modified.lr.key_col)

    def test_lrs_like(self):
        """Test copying LRS from another DataFrame."""
        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = self.df.copy()
        
        df1_with_lrs = df1.lr.lrs_like(df2, inplace=False)
        
        self.assertEqual(df1_with_lrs.lr.lrs, df2.lr.lrs)

    def test_get_group(self):
        """Test retrieving specific group."""
        group_a = self.df.lr.get_group('A')
        
        self.assertEqual(len(group_a), 2)
        self.assertTrue(all(group_a['route'] == 'A'))

    def test_iter_groups(self):
        """Test iterating over groups."""
        groups = list(self.df.lr.iter_groups())
        
        self.assertEqual(len(groups), 2)

    def test_sort_standard(self):
        """Test standard sorting."""
        df_unsorted = pd.DataFrame({
            'route': ['B', 'A', 'A', 'B'],
            'beg': [2.0, 1.0, 0.0, 0.0],
            'end': [4.0, 2.0, 1.0, 2.0]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        
        df_sorted = df_unsorted.lr.sort_standard(inplace=False)
        
        # Check that routes are in order
        self.assertEqual(df_sorted.iloc[0]['route'], 'A')
        self.assertEqual(df_sorted.iloc[0]['beg'], 0.0)


class TestEventOperations(unittest.TestCase):
    """Test event manipulation operations."""

    def setUp(self):
        """Set up test DataFrames."""
        self.df = pd.DataFrame({
            'route': ['A', 'A', 'A', 'B', 'B'],
            'beg': [0.0, 1.0, 2.0, 0.0, 2.0],
            'end': [1.0, 2.0, 3.0, 2.0, 4.0],
            'attr': ['x', 'x', 'y', 'z', 'z'],
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
                LineString([(2, 0), (3, 0)]),
                LineString([(0, 0), (2, 0)]),
                LineString([(2, 0), (4, 0)]),
            ]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', geom_col='geometry', closed='right')

    def test_extend(self):
        """Test extending event bounds."""
        df_extended = self.df.lr.extend(extend_begs=0.1, extend_ends=0.1, inplace=False)
        
        # Check that events were extended
        self.assertAlmostEqual(df_extended.iloc[0]['beg'], -0.1)
        self.assertAlmostEqual(df_extended.iloc[0]['end'], 1.1)

    def test_shift(self):
        """Test shifting events."""
        df_shifted = self.df.lr.shift(shift=10.0, inplace=False)
        
        # Check that events were shifted
        self.assertAlmostEqual(df_shifted.iloc[0]['beg'], 10.0)
        self.assertAlmostEqual(df_shifted.iloc[0]['end'], 11.0)

    def test_round(self):
        """Test rounding event locations."""
        df = pd.DataFrame({
            'route': ['A'],
            'beg': [0.1234],
            'end': [1.5678]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        
        df_rounded = df.lr.round(decimals=2, inplace=False)
        
        self.assertAlmostEqual(df_rounded.iloc[0]['beg'], 0.12)
        self.assertAlmostEqual(df_rounded.iloc[0]['end'], 1.57)

    def test_point_to_linear(self):
        """Test converting point events to linear."""
        df_point = pd.DataFrame({
            'route': ['A', 'B'],
            'loc': [1.5, 2.5]
        }).lr.set_lrs(key_col=['route'], loc_col='loc')
        
        df_linear = df_point.lr.point_to_linear(
            beg_col='beg',
            end_col='end',
            inplace=False
        )
        
        self.assertTrue(df_linear.lr.is_linear)
        self.assertEqual(df_linear.iloc[0]['beg'], 1.5)
        self.assertEqual(df_linear.iloc[0]['end'], 1.5)

    def test_dissolve(self):
        """Test dissolving consecutive events."""
        df_dissolved = self.df.lr.dissolve(retain=['attr'])
        
        # Should have fewer rows after dissolve
        self.assertLess(len(df_dissolved), len(self.df))
        
        # Check that consecutive events with same attributes were merged
        route_a = df_dissolved[df_dissolved['route'] == 'A']
        # First two events in route A have attr='x' and should be merged
        self.assertEqual(route_a['attr'].iloc[0], 'x')
        # Check merged event bounds
        self.assertEqual(df_dissolved.iloc[0]['beg'], 0.0)
        self.assertEqual(df_dissolved.iloc[0]['end'], 2.0)

    def test_dissolve_geometry(self):
        """Test dissolving events with geometry."""
        df_dissolved = self.df.lr.dissolve(retain=['attr'], merge_geom=True)
        
        # Check that geometry was dissolved correctly
        route_a = df_dissolved[df_dissolved['route'] == 'A']
        self.assertTrue(route_a.iloc[0]['geometry'].equals(
            LineString([(0, 0), (1, 0), (2, 0)])
        ))

    def test_resegment(self):
        """Test resegmenting events."""
        df_resegmented = self.df.lr.resegment(length=0.5, fill='cut', cut_geom=False)
        
        # Should have more rows after resegmenting
        self.assertGreater(len(df_resegmented), len(self.df))
        
        # Check that events are approximately the target length
        lengths = df_resegmented.lr.event_lengths
        self.assertTrue(all(lengths <= 0.5))

    def test_resegment_geometry(self):
        """Test resegmenting events with geometry cutting."""
        df_resegmented = self.df.lr.add_geom_m().lr.resegment(length=0.5, fill='cut', cut_geom=True)
        
        # Check that geometries were cut correctly
        for idx, row in df_resegmented.iterrows():
            event_length = row['end'] - row['beg']
            geom_length = row['geometry'].length
            self.assertAlmostEqual(event_length, geom_length)


class TestCompatibilityFunctions(unittest.TestCase):
    """Test compatibility checking functions."""

    def test_test_compatibility_valid(self):
        """Test compatibility check with valid DataFrames."""
        df1 = pd.DataFrame({
            'route': ['A'],
            'beg': [0.0],
            'end': [1.0]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        
        df2 = pd.DataFrame({
            'route': ['B'],
            'beg': [0.0],
            'end': [1.0]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        
        # Should not raise
        result = test_compatibility([df1, df2])
        self.assertEqual(result, [df1, df2])

    def test_test_compatibility_no_lrs(self):
        """Test compatibility check with DataFrame without LRS."""
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'b': [2]}).lr.set_lrs(key_col=['b'], beg_col='x', end_col='y')
        
        with self.assertRaises(LRSCompatibilityError):
            test_compatibility([df1, df2])

    def test_test_compatibility_different_key_count(self):
        """Test compatibility check with different key column counts."""
        df1 = pd.DataFrame({
            'route': ['A'],
            'beg': [0.0],
            'end': [1.0]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        
        df2 = pd.DataFrame({
            'route': ['B'],
            'dir': ['N'],
            'beg': [0.0],
            'end': [1.0]
        }).lr.set_lrs(key_col=['route', 'dir'], beg_col='beg', end_col='end', closed='right')
        
        with self.assertRaises(LRSCompatibilityError):
            test_compatibility([df1, df2])


class TestDefaultLRS(unittest.TestCase):
    """Test default LRS functionality."""

    def test_set_default_lrs(self):
        """Test setting default LRS."""
        default_lrs = LRS(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        LRS_Accessor.set_default_lrs(default_lrs)
        
        # New DataFrames should have this LRS by default
        df = pd.DataFrame({'route': ['A'], 'beg': [0], 'end': [1]})
        self.assertEqual(df.lr.lrs, default_lrs)
        
        # Clean up
        LRS_Accessor.clear_default_lrs()

    def test_clear_default_lrs(self):
        """Test clearing default LRS."""
        default_lrs = LRS(key_col=['route'], beg_col='beg', end_col='end')
        LRS_Accessor.set_default_lrs(default_lrs)
        LRS_Accessor.clear_default_lrs()
        
        df = pd.DataFrame({'a': [1]})
        self.assertEqual(df.lr.lrs, LRS())  # Should be empty LRS


class TestGeometrySyncBehavior(unittest.TestCase):
    """Test geometry synchronization behavior settings."""

    def test_set_default_geometry_sync(self):
        """Test setting default geometry sync behavior."""
        LRS_Accessor.set_default_geometry_sync('error')
        # This should be reflected in new accessor instances
        # Actual behavior testing would require geometry operations
        
        # Clean up
        LRS_Accessor.set_default_geometry_sync('warn')

    def test_set_default_geometry_sync_invalid(self):
        """Test that invalid sync behavior raises error."""
        with self.assertRaises(ValueError):
            LRS_Accessor.set_default_geometry_sync('invalid')

    def test_set_geometry_sync(self):
        """Test setting geometry sync behavior on instance."""
        df = pd.DataFrame({'a': [1]})
        df.lr.set_geometry_sync('error')
        # Behavior would be tested with actual geometry operations


class TestStudyMethod(unittest.TestCase):
    """Test the study method."""

    def test_study_complete_lrs(self):
        """Test study method with complete LRS."""
        df = pd.DataFrame({
            'route': ['A'],
            'beg': [0.0],
            'end': [1.0],
            'geometry': [LineString([(0, 0), (1, 1)])]
        })
        df = gpd.GeoDataFrame(df, geometry='geometry')
        df = df.lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            geom_col='geometry',
            closed='right'
        )
        
        study = df.lr.study()
        
        self.assertTrue(study['keys']['valid'])
        self.assertTrue(study['linear']['valid'])
        self.assertTrue(study['geometry']['valid'])

    def test_study_missing_columns(self):
        """Test study method with missing columns."""
        df = pd.DataFrame({'route': ['A'], 'beg': [0.0]})
        df = df.lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',  # This column doesn't exist
            closed='right'
        )
        
        study = df.lr.study()
        
        self.assertTrue(study['keys']['valid'])
        self.assertFalse(study['linear']['valid'])
        self.assertIn('end', study['linear']['missing'])


class TestProjectMethod(unittest.TestCase):
    """Test the project method for projecting point events onto linear events."""

    def setUp(self):
        """Set up test data with M-enabled geometries."""
        from linref.events.geometry import LineStringM
        
        # Create linear events (roads) with M-enabled geometries
        roads_data = pd.DataFrame({
            'route': ['US-101', 'US-101', 'SR-1'],
            'beg': [0.0, 10.0, 0.0],
            'end': [10.0, 20.0, 15.0]
        })
        
        # Add 2D geometries
        roads_data['geometry'] = [
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),  
            LineString([(0, 10), (15, 10)])
        ]
        
        # Add M-enabled geometries using LineStringM
        roads_data['geometry_m'] = [
            LineStringM(LineString([(0, 0), (10, 0)]), m=[0.0, 10.0]),
            LineStringM(LineString([(10, 0), (20, 0)]), m=[10.0, 20.0]),
            LineStringM(LineString([(0, 10), (15, 10)]), m=[0.0, 15.0])
        ]
        
        self.roads = gpd.GeoDataFrame(roads_data, geometry='geometry', crs='EPSG:4326')
        self.roads = self.roads.lr.set_lrs(
            key_col=['route'],
            loc_col='milepost',
            beg_col='beg',
            end_col='end',
            geom_col='geometry',
            geom_m_col='geometry_m',
            closed='left_mod'
        )
        
        # Create point events to project
        self.points = gpd.GeoDataFrame({
            'event_id': [1, 2, 3],
            'severity': ['High', 'Low', 'Medium'],
            'geometry': [
                Point(5, 0.05),      # Near US-101, MP ~5
                Point(15, 0.02),     # Near US-101, MP ~15
                Point(7, 10.1)       # Near SR-1, MP ~7
            ]
        }, crs='EPSG:4326')

    def test_project_basic(self):
        """Test basic projection of points onto lines."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore geographic CRS warning
            projected = self.roads.lr.project(self.points, buffer=1.0)
        
        # Check that all points were projected
        self.assertEqual(len(projected), 3)
        
        # Check that LRS columns were added
        self.assertIn('route', projected.columns)
        self.assertIn('milepost', projected.columns)
        self.assertIn('project_distance', projected.columns)
        
        # Check that original columns are preserved
        self.assertIn('event_id', projected.columns)
        self.assertIn('severity', projected.columns)
        
        # Check approximate milepost values
        self.assertAlmostEqual(projected.iloc[0]['milepost'], 5.0, places=0)
        self.assertAlmostEqual(projected.iloc[1]['milepost'], 15.0, places=0)
        self.assertAlmostEqual(projected.iloc[2]['milepost'], 7.0, places=0)

    def test_project_replace_false(self):
        """Test that replace=False raises error when columns exist."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            projected = self.roads.lr.project(self.points, buffer=1.0, replace=False)
        
        # Try to project again with replace=False (should fail)
        with self.assertRaises(ValueError) as context:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.roads.lr.project(projected, buffer=1.0, replace=False)
        
        # Check that error mentions protected columns
        error_msg = str(context.exception)
        self.assertIn('route', error_msg)
        self.assertIn('milepost', error_msg)

    def test_project_replace_true(self):
        """Test that replace=True successfully replaces existing columns."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            projected = self.roads.lr.project(self.points, buffer=1.0)
            
            # Modify the projected data
            projected_modified = projected.copy()
            projected_modified['route'] = 'FAKE_ROUTE'
            projected_modified['milepost'] = 999.0
            
            # Re-project with replace=True
            reprojected = self.roads.lr.project(projected_modified, buffer=1.0, replace=True)
        
        # Verify columns were replaced
        self.assertNotIn('FAKE_ROUTE', reprojected['route'].values)
        self.assertTrue(all(reprojected['milepost'] != 999.0))

    def test_project_dropna_false(self):
        """Test that dropna=False keeps unmatched points."""
        # Create points with one far away
        points_with_far = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'geometry': [
                Point(5, 0.05),    # Near US-101
                Point(100, 100),   # Far away - no match
                Point(7, 10.1)     # Near SR-1
            ]
        }, crs='EPSG:4326')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            projected = self.roads.lr.project(points_with_far, buffer=1.0, dropna=False)
        
        # Should keep all 3 rows
        self.assertEqual(len(projected), 3)
        
        # The far point should have NaN values
        self.assertTrue(projected.iloc[1]['route'] is None or pd.isna(projected.iloc[1]['route']))

    def test_project_dropna_true(self):
        """Test that dropna=True removes unmatched points."""
        # Create points with one far away
        points_with_far = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'geometry': [
                Point(5, 0.05),    # Near US-101
                Point(100, 100),   # Far away - no match
                Point(7, 10.1)     # Near SR-1
            ]
        }, crs='EPSG:4326')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            projected = self.roads.lr.project(points_with_far, buffer=1.0, dropna=True)
        
        # Should only have 2 rows (matched points)
        self.assertEqual(len(projected), 2)


# Run tests
if __name__ == '__main__':
    unittest.main()
