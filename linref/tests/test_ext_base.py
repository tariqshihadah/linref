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
from linref.options import options, set_default_lrs
from linref.ext.base import check_compatibility
from linref.errors import LRSConfigurationError, LRSCompatibilityError


class TestLRSInit(unittest.TestCase):
    """Test LRS class initialization and basic properties."""

    def test_lrs_init_empty(self):
        """Test creating an empty LRS object with minimal parameters."""
        lrs = LRS()
        self.assertIsInstance(lrs, LRS)
        self.assertTrue(len(lrs.key_col) == 0)
        self.assertIsNone(lrs.chain_col)
        self.assertIsNone(lrs.loc_col)
        self.assertIsNone(lrs.beg_col)
        self.assertIsNone(lrs.end_col)
        self.assertIsNone(lrs.geom_col)
        self.assertIsNone(lrs.geom_m_col)
        self.assertIsNone(lrs.closed)
        self.assertFalse(lrs.is_chained)

    def test_lrs_init_linear(self):
        """Test creating a linear LRS object."""
        lrs = LRS(
            key_col=['route_id'],
            chain_col='chain',
            beg_col='begin_mp',
            end_col='end_mp',
            closed='right'
        )
        self.assertIsInstance(lrs, LRS)
        self.assertEqual(lrs.key_col, ['route_id'])
        self.assertEqual(lrs.chain_col, 'chain')
        self.assertNotIn('chain', lrs.key_col)
        self.assertTrue(lrs.is_chained)
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
        self.assertIn('chain_col', params)
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
        
        # Copy preserves chain_col
        lrs_chain = LRS(key_col=['route'], chain_col='chain')
        self.assertEqual(lrs_chain.copy().chain_col, 'chain')
        self.assertEqual(lrs_chain.copy(deep=True).chain_col, 'chain')

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
        
        # chain_col via set_params
        lrs2 = lrs.set_params(chain_col='chain', inplace=False)
        self.assertIsNone(lrs.chain_col)
        self.assertEqual(lrs2.chain_col, 'chain')

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
        lrs1 = LRS(key_col=['route'], chain_col='chain', beg_col='beg', end_col='end', closed='right')
        lrs2 = LRS(key_col=['route'], chain_col='chain', beg_col='beg', end_col='end', closed='right')
        lrs3 = LRS(key_col=['route'], chain_col=None, beg_col='start', end_col='end', closed='right')
        lrs4 = LRS(key_col=['route'], chain_col=None, beg_col='beg', end_col='end', closed='right')
        lrs5 = LRS(key_col=['route'], chain_col=None, beg_col='beg', end_col='end', closed='right')
        
        self.assertEqual(lrs1, lrs2)
        self.assertNotEqual(lrs1, lrs3)
        self.assertNotEqual(lrs1, "not an LRS")
        self.assertEqual(lrs4, lrs5)
        self.assertNotEqual(lrs1, lrs4)

    def test_lrs_str_repr(self):
        """Test LRS string representation."""
        lrs = self.lrs_linear
        lrs_str = str(lrs)
        self.assertIn('LRS(', lrs_str)
        self.assertIn('key_col=', lrs_str)
        self.assertIn('chain_col=', lrs_str)
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
        self.assertFalse(study['chaining']['defined'])


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
        df_extended = self.df.lr.extend(extend_begs=0.1, extend_ends=0.1, inplace=False, geometry_sync='ignore')
        
        # Check that events were extended
        self.assertAlmostEqual(df_extended.iloc[0]['beg'], -0.1)
        self.assertAlmostEqual(df_extended.iloc[0]['end'], 1.1)

    def test_shift(self):
        """Test shifting events."""
        df_shifted = self.df.lr.shift(shift=10.0, inplace=False, geometry_sync='ignore')
        
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

    def test_separate(self):
        """Test separating overlapping events via accessor."""
        df_overlap = pd.DataFrame({
            'route': ['A', 'A', 'A'],
            'beg': [0.0, 3.0, 7.0],
            'end': [5.0, 8.0, 12.0],
            'attr': ['x', 'y', 'z']
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')

        result = df_overlap.lr.separate()

        # No overlaps in output
        for i in range(len(result) - 1):
            self.assertLessEqual(result.iloc[i]['end'], result.iloc[i + 1]['beg'])
        # LRS is preserved
        self.assertTrue(result.lr.lrs == df_overlap.lr.lrs)
        self.assertEqual(result.lr.beg_col, 'beg')
        # Attributes are preserved
        self.assertEqual(list(result['attr']), ['x', 'y', 'z'])

    def test_separate_drop_short(self):
        """Test separate with drop_short removes eclipsed events."""
        df_eclipsed = pd.DataFrame({
            'route': ['A', 'A'],
            'beg': [0.0, 2.0],
            'end': [10.0, 5.0]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='right')

        result = df_eclipsed.lr.separate(drop_short=True)
        self.assertEqual(len(result), 1)
        self.assertTrue(result.lr.lrs == df_eclipsed.lr.lrs)


class TestCompatibilityFunctions(unittest.TestCase):
    """Test compatibility checking functions."""

    def test_check_compatibility_valid(self):
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
        result = check_compatibility([df1, df2])
        self.assertEqual(result, [df1, df2])

    def test_check_compatibility_no_lrs(self):
        """Test compatibility check with DataFrame without LRS."""
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'b': [2]}).lr.set_lrs(key_col=['b'], beg_col='x', end_col='y')
        
        with self.assertRaises(LRSCompatibilityError):
            check_compatibility([df1, df2])

    def test_check_compatibility_different_key_count(self):
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
            check_compatibility([df1, df2])


class TestDefaultLRS(unittest.TestCase):
    """Test default LRS functionality."""

    def tearDown(self):
        options.reset()

    def test_set_default_lrs(self):
        """Test setting default LRS."""
        default_lrs = LRS(key_col=['route'], beg_col='beg', end_col='end', closed='right')
        set_default_lrs(default_lrs)
        
        # New DataFrames should have this LRS by default
        df = pd.DataFrame({'route': ['A'], 'beg': [0], 'end': [1]})
        self.assertEqual(df.lr.lrs, default_lrs)

    def test_set_default_lrs_with_kwargs(self):
        """Test setting default LRS via keyword arguments."""
        set_default_lrs(key_col=['route'], beg_col='beg', end_col='end')
        expected = LRS(key_col=['route'], beg_col='beg', end_col='end')
        self.assertEqual(options.default_lrs, expected)

    def test_set_default_lrs_rejects_none(self):
        """Test that setting default LRS to None raises ValueError."""
        with self.assertRaises(ValueError):
            options.default_lrs = None

    def test_set_default_lrs_rejects_invalid(self):
        """Test that setting default LRS to non-LRS raises ValueError."""
        with self.assertRaises(ValueError):
            options.default_lrs = 'not an LRS'

    def test_options_reset(self):
        """Test resetting options restores defaults."""
        set_default_lrs(key_col=['route'], beg_col='beg', end_col='end')
        options.reset()
        
        df = pd.DataFrame({'a': [1]})
        self.assertEqual(df.lr.lrs, LRS())  # Should be empty LRS

    def test_options_read_default_lrs(self):
        """Test reading default LRS from options."""
        default_lrs = LRS(key_col=['route'], beg_col='beg', end_col='end')
        set_default_lrs(default_lrs)
        self.assertEqual(options.default_lrs, default_lrs)

    def test_package_level_options_access(self):
        """Test that linref.options provides access to defaults."""
        default_lrs = LRS(key_col=['route'], beg_col='beg', end_col='end')
        set_default_lrs(default_lrs)
        self.assertEqual(linref.options.default_lrs, default_lrs)


class TestGeometrySyncBehavior(unittest.TestCase):
    """Test geometry synchronization behavior settings."""

    def tearDown(self):
        options.reset()

    def test_set_default_geometry_sync(self):
        """Test setting default geometry sync behavior."""
        options.default_geometry_sync = 'error'
        df = pd.DataFrame({'a': [1]})
        self.assertEqual(df.lr.geometry_sync, 'error')

    def test_set_default_geometry_sync_invalid(self):
        """Test that invalid sync behavior raises error."""
        with self.assertRaises(ValueError):
            options.default_geometry_sync = 'invalid'

    def test_set_geometry_sync_on_instance(self):
        """Test setting geometry sync behavior on instance via property."""
        df = pd.DataFrame({'a': [1]})
        df.lr.geometry_sync = 'error'
        self.assertEqual(df.lr.geometry_sync, 'error')


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
        from linref.geometry import LineStringM
        
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
        
        self.roads = gpd.GeoDataFrame(roads_data, geometry='geometry', crs='EPSG:3857')
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
        }, crs='EPSG:3857')

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
        }, crs='EPSG:3857')
        
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
        }, crs='EPSG:3857')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            projected = self.roads.lr.project(points_with_far, buffer=1.0, dropna=True)
        
        # Should only have 2 rows (matched points)
        self.assertEqual(len(projected), 2)


class TestIntegrateMethod(unittest.TestCase):
    """Test the integrate method for combining multiple DataFrames."""

    def setUp(self):
        """Set up test DataFrames with overlapping and non-overlapping events."""
        # DataFrame 1: Road pavement conditions
        self.df1 = pd.DataFrame({
            'route': ['A', 'A', 'B'],
            'beg': [0.0, 5.0, 0.0],
            'end': [5.0, 10.0, 8.0],
            'condition': ['Good', 'Fair', 'Good']
        }).lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='left_mod'
        )
        
        # DataFrame 2: Speed limits (overlapping with df1)
        self.df2 = pd.DataFrame({
            'route': ['A', 'A', 'B'],
            'beg': [0.0, 6.0, 0.0],
            'end': [6.0, 12.0, 10.0],
            'speed_limit': [35, 45, 55]
        }).lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='left_mod'
        )
        
        # DataFrame 3: Traffic volume (different segmentation)
        self.df3 = pd.DataFrame({
            'route': ['A', 'A', 'B'],
            'beg': [0.0, 7.0, 2.0],
            'end': [7.0, 11.0, 9.0],
            'volume': [1000, 1500, 2000]
        }).lr.set_lrs(
            key_col=['route'],
            beg_col='beg',
            end_col='end',
            closed='left_mod'
        )

    def test_integrate_basic_two_dataframes(self):
        """Test basic integration of two DataFrames."""
        integrated = self.df1.lr.integrate(self.df2)
        
        # Check that result is a DataFrame
        self.assertIsInstance(integrated, pd.DataFrame)
        
        # Check that LRS columns are present
        self.assertIn('route', integrated.columns)
        self.assertIn('beg', integrated.columns)
        self.assertIn('end', integrated.columns)
        
        # Check that inverse index columns are present
        self.assertIn('integrated_index_0', integrated.columns)
        self.assertIn('integrated_index_1', integrated.columns)
        
        # Check that integration created more segments (least common intervals)
        self.assertGreater(len(integrated), len(self.df1))
        self.assertGreater(len(integrated), len(self.df2))

    def test_integrate_function_multiple_dataframes(self):
        """Test integrate function with multiple DataFrames."""
        integrated = integrate([self.df1, self.df2, self.df3])
        
        # Check correct number of inverse index columns
        self.assertIn('integrated_index_0', integrated.columns)
        self.assertIn('integrated_index_1', integrated.columns)
        self.assertIn('integrated_index_2', integrated.columns)
        
        # Check that we get even more granular segments
        self.assertGreater(len(integrated), len(self.df1))

    def test_integrate_custom_inverse_col_string(self):
        """Test integrate with custom inverse column name (string)."""
        integrated = self.df1.lr.integrate(self.df2, inverse_col='source_idx')
        
        # Check that custom column names are used
        self.assertIn('source_idx_0', integrated.columns)
        self.assertIn('source_idx_1', integrated.columns)
        
        # Default names should not be present
        self.assertNotIn('integrated_index_0', integrated.columns)

    def test_integrate_custom_inverse_col_list(self):
        """Test integrate with custom inverse column names (list)."""
        integrated = integrate(
            [self.df1, self.df2, self.df3],
            inverse_col=['pavement_idx', 'speed_idx', 'volume_idx']
        )
        
        # Check that custom column names are used
        self.assertIn('pavement_idx', integrated.columns)
        self.assertIn('speed_idx', integrated.columns)
        self.assertIn('volume_idx', integrated.columns)

    def test_integrate_inverse_col_list_wrong_length(self):
        """Test that inverse_col list with wrong length raises error."""
        with self.assertRaises(ValueError) as context:
            integrate([self.df1, self.df2, self.df3], inverse_col=['a', 'b'])
        
        self.assertIn('length must match', str(context.exception))

    def test_integrate_fill_gaps_false(self):
        """Test integrate with fill_gaps=False (default)."""
        # Create DataFrames with gaps
        df_gap1 = pd.DataFrame({
            'route': ['A'],
            'beg': [0.0],
            'end': [5.0],
            'attr': [1]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='left_mod')
        
        df_gap2 = pd.DataFrame({
            'route': ['A'],
            'beg': [7.0],
            'end': [10.0],
            'attr': [2]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='left_mod')
        
        integrated = df_gap1.lr.integrate(df_gap2, fill_gaps=False)
        
        # Should only have segments where there's overlap or original events
        # Gap between 5.0 and 7.0 should not be filled
        gap_segments = integrated[(integrated['beg'] >= 5.0) & (integrated['end'] <= 7.0)]
        # There should be no segment spanning the gap
        self.assertEqual(len(gap_segments), 0)

    def test_integrate_fill_gaps_true(self):
        """Test integrate with fill_gaps=True."""
        # Create DataFrames with gaps
        df_gap1 = pd.DataFrame({
            'route': ['A'],
            'beg': [0.0],
            'end': [5.0],
            'attr': [1]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='left_mod')
        
        df_gap2 = pd.DataFrame({
            'route': ['A'],
            'beg': [7.0],
            'end': [10.0],
            'attr': [2]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='left_mod')
        
        integrated = df_gap1.lr.integrate(df_gap2, fill_gaps=True)
        
        # Should have a segment filling the gap (5.0 to 7.0)
        # Total extent should be from min beg to max end
        self.assertAlmostEqual(integrated['beg'].min(), 0.0)
        self.assertAlmostEqual(integrated['end'].max(), 10.0)
        
        # Should have exactly 3 segments (0-5, 5-7 gap, 7-10)
        self.assertEqual(len(integrated), 3)

    def test_integrate_split_at_locs_false(self):
        """Test integrate with split_at_locs=False (default)."""
        # Create a linear and point event dataframe
        df_linear = pd.DataFrame({
            'route': ['A'],
            'beg': [0.0],
            'end': [10.0],
            'attr': [1]
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='left_mod')
        
        df_point = pd.DataFrame({
            'route': ['A'],
            'loc': [5.0],
            'attr': [2]
        }).lr.set_lrs(key_col=['route'], loc_col='loc')
        
        # Without split_at_locs, point events should not cause splits
        # This should only work with linear events
        integrated = df_linear.lr.integrate([df_linear], split_at_locs=False)
        
        # Result should match the input structure
        self.assertEqual(len(integrated), 1)

    def test_integrate_incompatible_lrs(self):
        """Test that incompatible LRS raises error."""
        df_incompatible = pd.DataFrame({
            'route': ['A'],
            'start': [0.0],
            'finish': [10.0]
        }).lr.set_lrs(
            key_col=['route'],
            beg_col='start',
            end_col='finish',
            closed='right'  # Different closed parameter
        )
        
        with self.assertRaises(LRSCompatibilityError):
            self.df1.lr.integrate(df_incompatible)

    def test_integrate_no_lrs_set(self):
        """Test that DataFrame without LRS raises error."""
        df_no_lrs = pd.DataFrame({
            'route': ['A'],
            'beg': [0.0],
            'end': [10.0]
        })
        
        with self.assertRaises(LRSCompatibilityError):
            integrate([self.df1, df_no_lrs])

    def test_integrate_point_events_error(self):
        """Test that point-only DataFrames raise error."""
        df_point = pd.DataFrame({
            'route': ['A'],
            'loc': [5.0]
        }).lr.set_lrs(key_col=['route'], loc_col='loc')
        
        with self.assertRaises(LRSCompatibilityError):
            self.df1.lr.integrate(df_point)

    def test_integrate_accessor_with_list(self):
        """Test accessor integrate method with list of DataFrames."""
        integrated = self.df1.lr.integrate([self.df2, self.df3])
        
        # Should have 3 inverse index columns (df1 + df2 + df3)
        self.assertIn('integrated_index_0', integrated.columns)
        self.assertIn('integrated_index_1', integrated.columns)
        self.assertIn('integrated_index_2', integrated.columns)

    def test_integrate_preserves_index_mapping(self):
        """Test that inverse index columns correctly map to original indices."""
        # Use simple DataFrames with known indices
        df_a = pd.DataFrame({
            'route': ['X'],
            'beg': [0.0],
            'end': [5.0],
            'data_a': ['A']
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='left_mod')
        
        df_b = pd.DataFrame({
            'route': ['X'],
            'beg': [3.0],
            'end': [8.0],
            'data_b': ['B']
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='left_mod')
        
        integrated = integrate([df_a, df_b])
        
        # Check that inverse indices are valid (not all NaN)
        self.assertFalse(integrated['integrated_index_0'].isna().all())
        self.assertFalse(integrated['integrated_index_1'].isna().all())
        
        # There should be segments referencing both dataframes in the overlap
        overlap_segs = integrated[
            integrated['integrated_index_0'].notna() & 
            integrated['integrated_index_1'].notna()
        ]
        self.assertGreater(len(overlap_segs), 0)


class TestAccessorStatePersistence(unittest.TestCase):
    """Test that accessor state persists across re-instantiation (pandas 3.0+)."""

    def setUp(self):
        self.df = pd.DataFrame({
            'route': ['A', 'B'],
            'beg': [0.0, 1.0],
            'end': [1.0, 2.0],
        })
        self.lrs = LRS(key_col=['route'], beg_col='beg', end_col='end', closed='right')

    def test_lrs_survives_reinstantiation(self):
        """LRS state persists when a new accessor is constructed on the same df."""
        self.df.lr.set_lrs(self.lrs, inplace=True)
        # Simulate pandas 3.0: construct a brand-new accessor on the same df
        new_acc = LRS_Accessor(self.df)
        self.assertEqual(new_acc.lrs, self.lrs)
        self.assertEqual(new_acc.lrs.beg_col, 'beg')
        self.assertTrue(new_acc.is_linear)

    def test_geometry_sync_survives_reinstantiation(self):
        """Geometry sync state persists across accessor re-construction."""
        self.df.lr.geometry_sync = 'error'
        new_acc = LRS_Accessor(self.df)
        self.assertEqual(new_acc.geometry_sync, 'error')

    def test_lrs_propagates_through_copy(self):
        """LRS state propagates through df.copy() via attrs."""
        df = self.df.lr.set_lrs(self.lrs, inplace=False)
        df_copy = df.copy()
        self.assertEqual(df_copy.lr.lrs, self.lrs)
        self.assertTrue(df_copy.lr.is_linear)

    def test_consecutive_lr_accesses_share_state(self):
        """Multiple df.lr accesses see the same state (the core pandas 3.0 fix)."""
        self.df.lr.set_lrs(self.lrs, inplace=True)
        # These are potentially different accessor instances in pandas 3.0
        lrs1 = self.df.lr.lrs
        lrs2 = self.df.lr.lrs
        self.assertEqual(lrs1, lrs2)
        self.assertEqual(lrs1.beg_col, 'beg')

    def test_set_lrs_then_dissolve(self):
        """The primary failure scenario from pandas 3.0: set_lrs then use method."""
        df = pd.DataFrame({
            'route': ['A', 'A', 'A'],
            'beg': [0.0, 1.0, 2.0],
            'end': [1.0, 2.0, 3.0],
            'value': [10, 10, 20],
        })
        df.lr.set_lrs(self.lrs, inplace=True)
        # This should work — dissolve sees the LRS even if accessor is re-created
        result = df.lr.dissolve(retain=['value'])
        self.assertIsNotNone(result)
        self.assertTrue(result.lr.is_linear)

    def test_default_lrs_used_for_new_df(self):
        """Default LRS is still used for DataFrames without stored attrs."""
        default = LRS(key_col=['x'], beg_col='a', end_col='b')
        set_default_lrs(default)
        try:
            fresh_df = pd.DataFrame({'x': [1], 'a': [0], 'b': [1]})
            self.assertEqual(fresh_df.lr.lrs, default)
        finally:
            options.reset()

    def test_inplace_false_returns_independent_state(self):
        """Non-inplace set_lrs returns a df with independent LRS state."""
        df_with_lrs = self.df.lr.set_lrs(self.lrs, inplace=False)
        # Original should still be empty
        self.assertTrue(self.df.lr.is_lrs_empty)
        # Returned df should have the LRS
        self.assertEqual(df_with_lrs.lr.lrs, self.lrs)


class TestChainCol(unittest.TestCase):
    """Test chain_col as a first-class LRS parameter."""

    def setUp(self):
        """Set up test data with disjointed geometries for chaining tests."""
        from linref.geometry import LineStringM
        
        # Create data with disjointed geometries within a route:
        # Route A: two contiguous segments + one disjointed segment
        # Route B: one segment
        self.df = gpd.GeoDataFrame({
            'route': ['A', 'A', 'A', 'B'],
            'beg': [0.0, 5.0, 20.0, 0.0],
            'end': [5.0, 10.0, 25.0, 8.0],
            'geometry': [
                LineString([(0, 0), (5, 0)]),     # A chain 0
                LineString([(5, 0), (10, 0)]),    # A chain 0
                LineString([(20, 0), (25, 0)]),   # A chain 1 (disjointed)
                LineString([(0, 10), (8, 10)]),   # B chain 0
            ],
            'geometry_m': [
                LineStringM(LineString([(0, 0), (5, 0)]), m=[0.0, 5.0]),
                LineStringM(LineString([(5, 0), (10, 0)]), m=[5.0, 10.0]),
                LineStringM(LineString([(20, 0), (25, 0)]), m=[20.0, 25.0]),
                LineStringM(LineString([(0, 10), (8, 10)]), m=[0.0, 8.0]),
            ]
        }, geometry='geometry')

    # --- Accessor key_col dynamic behavior ---

    def test_key_col_excludes_absent_chain(self):
        """key_col should not include chain_col when column is absent."""
        df = self.df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        # chain column doesn't exist yet
        self.assertEqual(df.lr.key_col, ['route'])

    def test_key_col_includes_present_chain(self):
        """key_col should include chain_col when column exists."""
        df = self.df.copy()
        df['chain'] = [0, 0, 1, 0]
        df = df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        self.assertEqual(df.lr.key_col, ['route', 'chain'])

    def test_base_key_col_always_excludes_chain(self):
        """base_key_col should always exclude chain_col."""
        df = self.df.copy()
        df['chain'] = [0, 0, 1, 0]
        df = df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        self.assertEqual(df.lr.base_key_col, ['route'])
        # Even though chain exists, base_key_col excludes it
        self.assertNotIn('chain', df.lr.base_key_col)

    def test_no_double_append_chain_in_key_col(self):
        """If chain_col is already in key_col, don't duplicate it."""
        df = self.df.copy()
        df['chain'] = [0, 0, 1, 0]
        df = df.lr.set_lrs(
            key_col=['route', 'chain'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        # chain should appear only once
        self.assertEqual(df.lr.key_col.count('chain'), 1)

    def test_missing_key_cols_excludes_chain(self):
        """missing_key_cols should not report absent chain_col."""
        df = self.df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        # chain column doesn't exist, but shouldn't be in missing_key_cols
        self.assertEqual(df.lr.missing_key_cols, [])

    # --- add_chaining with chain_col ---

    def test_add_chaining_defaults_from_lrs(self):
        """add_chaining should use chain_col from LRS as default name."""
        df = self.df.lr.set_lrs(
            key_col=['route'], chain_col='my_chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        result = df.lr.add_chaining()
        # Should use 'my_chain' from LRS
        self.assertIn('my_chain', result.columns)
        self.assertEqual(result.lr.lrs.chain_col, 'my_chain')

    def test_add_chaining_sets_chain_col_on_lrs(self):
        """add_chaining should set chain_col on LRS when not defined."""
        df = self.df.lr.set_lrs(
            key_col=['route'],
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        result = df.lr.add_chaining(name='chain')
        # chain_col should be set on LRS
        self.assertEqual(result.lr.lrs.chain_col, 'chain')
        # chain should NOT be in key_col
        self.assertNotIn('chain', result.lr.lrs.key_col)
        # But accessor key_col should include it (column now exists)
        self.assertIn('chain', result.lr.key_col)

    def test_add_chaining_computes_correct_chains(self):
        """add_chaining should correctly identify disjointed chains."""
        df = self.df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        result = df.lr.add_chaining()
        chains = result['chain']
        # Route A: first two are chain 0, third is chain 1
        self.assertEqual(chains.iloc[0], 0)
        self.assertEqual(chains.iloc[1], 0)
        self.assertEqual(chains.iloc[2], 1)
        # Route B: single chain 0
        self.assertEqual(chains.iloc[3], 0)

    def test_add_chaining_legacy_migration(self):
        """add_chaining migrates chain from key_col to chain_col."""
        df = self.df.copy()
        df['chain'] = [0, 0, 1, 0]
        df = df.lr.set_lrs(
            key_col=['route', 'chain'],
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        # chain is in key_col (legacy pattern)
        self.assertIn('chain', df.lr.lrs.key_col)
        result = df.lr.add_chaining(name='chain', replace=True)
        # After add_chaining, chain should be in chain_col, not key_col
        self.assertEqual(result.lr.lrs.chain_col, 'chain')
        self.assertNotIn('chain', result.lr.lrs.key_col)

    # --- Legacy pattern backward compatibility ---

    def test_legacy_chain_in_key_col_still_works(self):
        """Chain column in key_col (without chain_col) should work."""
        df = self.df.copy()
        df['chain'] = [0, 0, 1, 0]
        df = df.lr.set_lrs(
            key_col=['route', 'chain'],
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        # Should work normally — chain is a regular key column
        self.assertEqual(df.lr.key_col, ['route', 'chain'])
        self.assertTrue(df.lr.is_grouped)

    # --- is_contiguous / is_disjointed ---

    def test_is_contiguous_false_without_chain_column(self):
        """is_contiguous should be False when disjointed segments exist without chaining."""
        df = self.df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        # No chain column in data — disjoint segments in route A
        self.assertFalse(df.lr.is_contiguous)

    def test_is_contiguous_true_with_correct_chains(self):
        """is_contiguous should be True when chain column correctly separates disjoint segments."""
        df = self.df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        result = df.lr.add_chaining()
        # After add_chaining, data should be properly chained
        self.assertTrue(result.lr.is_contiguous)

    def test_is_contiguous_false_with_incorrect_chains(self):
        """is_contiguous should be False when chain column doesn't properly separate disjoint segments."""
        df = self.df.copy()
        # Assign all to chain 0 — incorrect, since route A has disjointed segments
        df['chain'] = [0, 0, 0, 0]
        df = df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        self.assertFalse(df.lr.is_contiguous)

    def test_get_chains_include_chain_all_zero_when_correct(self):
        """get_chains(include_chain=True) should return all zeros when chains are correct."""
        df = self.df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        result = df.lr.add_chaining()
        chains = result.lr.get_chains(include_chain=True)
        self.assertTrue(all(chains == 0))

    def test_get_chains_include_chain_nonzero_when_incorrect(self):
        """get_chains(include_chain=True) should return non-zero when chains are wrong."""
        df = self.df.copy()
        df['chain'] = [0, 0, 0, 0]
        df = df.lr.set_lrs(
            key_col=['route'], chain_col='chain',
            beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m',
            closed='left_mod'
        )
        chains = df.lr.get_chains(include_chain=True)
        # Route A chain 0 still has disjointed segments
        self.assertFalse(all(chains == 0))


class TestSetMonotonic(unittest.TestCase):
    """Test LRS_Accessor.set_monotonic method."""

    def setUp(self):
        """Set up test DataFrame with mixed monotonic/non-monotonic events."""
        self.df = pd.DataFrame({
            'route': ['A', 'A', 'B'],
            'beg': [0.0, 2.0, 4.0],
            'end': [1.0, 1.0, 2.0],
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(2, 0), (1, 0)]),
                LineString([(4, 0), (2, 0)]),
            ]
        }).lr.set_lrs(
            key_col=['route'], beg_col='beg', end_col='end',
            geom_col='geometry', closed='right'
        )

    def test_set_monotonic_with_geometry_reversal(self):
        """Test that bounds are enforced and geometries reversed by default."""
        result = self.df.lr.set_monotonic()
        # Bounds enforced
        self.assertTrue(np.all(result.lr.begs <= result.lr.ends))
        # Monotonic event unchanged
        self.assertEqual(list(result.iloc[0]['geometry'].coords), [(0, 0), (1, 0)])
        # Non-monotonic event geometries reversed
        self.assertEqual(list(result.iloc[1]['geometry'].coords), [(1, 0), (2, 0)])
        self.assertEqual(list(result.iloc[2]['geometry'].coords), [(2, 0), (4, 0)])

    def test_set_monotonic_without_geometry_reversal(self):
        """Test that bounds are enforced but geometries preserved when
        reverse_geom=False."""
        result = self.df.lr.set_monotonic(reverse_geom=False)
        # Bounds enforced
        self.assertTrue(np.all(result.lr.begs <= result.lr.ends))
        np.testing.assert_array_equal(result['beg'].values, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result['end'].values, [1.0, 2.0, 4.0])
        # Geometries unchanged
        self.assertEqual(list(result.iloc[1]['geometry'].coords), [(2, 0), (1, 0)])
        self.assertEqual(list(result.iloc[2]['geometry'].coords), [(4, 0), (2, 0)])


class TestCluster(unittest.TestCase):
    """Test cluster method on LRS_Accessor."""

    def test_point_events_basic(self):
        """Test clustering point events with clear proximity groups."""
        df = pd.DataFrame({
            'route': ['A', 'A', 'A', 'A'],
            'mp': [1.0, 1.01, 5.0, 5.005],
        }).lr.set_lrs(key_col=['route'], loc_col='mp')

        result = df.lr.cluster(max_gap=0.02)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0, 1, 1])

    def test_point_events_grouped(self):
        """Test that clustering respects group boundaries."""
        df = pd.DataFrame({
            'route': ['A', 'A', 'B', 'B'],
            'mp': [1.0, 1.01, 1.0, 1.01],
        }).lr.set_lrs(key_col=['route'], loc_col='mp')

        result = df.lr.cluster(max_gap=0.02)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0, 1, 1])

    def test_point_events_transitive(self):
        """Test transitive clustering: A near B, B near C → all in one cluster."""
        df = pd.DataFrame({
            'route': ['A', 'A', 'A'],
            'mp': [1.0, 1.015, 1.03],
        }).lr.set_lrs(key_col=['route'], loc_col='mp')

        # With tolerance=0.02, 1.0↔1.015 overlap, 1.015↔1.03 overlap,
        # but 1.0↔1.03 do NOT directly overlap. Connected components
        # should still group all three together.
        result = df.lr.cluster(max_gap=0.02)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0, 0])

    def test_linear_events_overlapping(self):
        """Test clustering linear events that overlap after buffering."""
        df = pd.DataFrame({
            'route': ['A', 'A', 'A'],
            'beg': [0.0, 0.9, 5.0],
            'end': [1.0, 2.0, 6.0],
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='both')

        # First two events already overlap (0-1 and 0.9-2), tolerance extends further
        result = df.lr.cluster(max_gap=0.1)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0, 1])

    def test_linear_events_gap_within_tolerance(self):
        """Test that linear events with gap smaller than tolerance cluster."""
        df = pd.DataFrame({
            'route': ['A', 'A'],
            'beg': [0.0, 1.05],
            'end': [1.0, 2.0],
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='both')

        # Gap of 0.05, tolerance of 0.1 should bridge it
        result = df.lr.cluster(max_gap=0.1)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0])

    def test_custom_column_name(self):
        """Test using a custom name for the cluster column."""
        df = pd.DataFrame({
            'route': ['A', 'A'],
            'mp': [1.0, 5.0],
        }).lr.set_lrs(key_col=['route'], loc_col='mp')

        result = df.lr.cluster(max_gap=0.1, name='group_id')
        self.assertIn('group_id', result.columns)
        self.assertNotIn('cluster', result.columns)
        np.testing.assert_array_equal(result['group_id'].values, [0, 1])

    def test_inplace(self):
        """Test inplace modification."""
        df = pd.DataFrame({
            'route': ['A', 'A'],
            'mp': [1.0, 5.0],
        }).lr.set_lrs(key_col=['route'], loc_col='mp')

        result = df.lr.cluster(max_gap=0.1, inplace=True)
        self.assertIsNone(result)
        np.testing.assert_array_equal(df['cluster'].values, [0, 1])

    def test_no_positional_data_raises(self):
        """Test that error is raised when no location or linear data exists."""
        df = pd.DataFrame({
            'route': ['A', 'A'],
        }).lr.set_lrs(key_col=['route'])

        with self.assertRaises(LRSConfigurationError):
            df.lr.cluster(max_gap=0.1)

    def test_zero_tolerance(self):
        """Test with zero tolerance: coincident points cluster, others don't."""
        df = pd.DataFrame({
            'route': ['A', 'A', 'A'],
            'mp': [1.0, 1.0, 2.0],
        }).lr.set_lrs(key_col=['route'], loc_col='mp')

        # Zero tolerance with enforce_edges defaulting to True means
        # coincident points cluster, but non-coincident points don't.
        result = df.lr.cluster(max_gap=0.0)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0, 1])

    def test_small_tolerance_coincident(self):
        """Test that coincident points cluster with any positive tolerance."""
        df = pd.DataFrame({
            'route': ['A', 'A', 'A'],
            'mp': [1.0, 1.0, 2.0],
        }).lr.set_lrs(key_col=['route'], loc_col='mp')

        result = df.lr.cluster(max_gap=0.001)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0, 1])

    def test_enforce_edges_on_linear_events(self):
        """Test enforce_edges=True on linear events includes touching boundaries."""
        df = pd.DataFrame({
            'route': ['A', 'A'],
            'beg': [0.0, 1.0],
            'end': [1.0, 2.0],
        }).lr.set_lrs(key_col=['route'], beg_col='beg', end_col='end', closed='both')

        # With enforce_edges=True, adjacent events sharing edge at 1.0 cluster
        result = df.lr.cluster(max_gap=0.0, enforce_edges=True)
        np.testing.assert_array_equal(result['cluster'].values, [0, 0])

        # With enforce_edges=False, they don't cluster
        result = df.lr.cluster(max_gap=0.0, enforce_edges=False)
        np.testing.assert_array_equal(result['cluster'].values, [0, 1])


# Run tests
if __name__ == '__main__':
    unittest.main()
