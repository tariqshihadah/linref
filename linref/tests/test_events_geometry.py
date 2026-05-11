"""
Unit tests for linref.events.geometry module.

Tests cover:
- LineStringM class initialization and properties
- substring_m_coords function including edge cases
- Floating point precision handling
- Adjacent substring boundary consistency
"""

import unittest
import numpy as np
from shapely.geometry import LineString, Point
from linref.geometry import LineStringM, substring_m_coords
from linref.errors import GeometryMeasureError


class TestLineStringM(unittest.TestCase):
    """Test LineStringM class initialization and basic properties."""

    def test_init_without_m(self):
        """Test creating a LineStringM without M values."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        lsm = LineStringM(geom)
        self.assertIsInstance(lsm, LineStringM)
        self.assertEqual(lsm.geom, geom)
        self.assertIsNone(lsm.m)

    def test_init_with_m(self):
        """Test creating a LineStringM with M values."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([0.0, 1.0, 2.0])
        lsm = LineStringM(geom, m)
        self.assertIsInstance(lsm, LineStringM)
        self.assertEqual(lsm.geom, geom)
        np.testing.assert_array_equal(lsm.m, m)

    def test_init_with_invalid_m_length(self):
        """Test that M values must match coordinate length."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([0.0, 1.0])  # Wrong length
        with self.assertRaises(ValueError):
            LineStringM(geom, m)

    def test_init_with_non_monotonic_m(self):
        """Test that M values must be monotonically increasing."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([0.0, 2.0, 1.0])  # Not monotonic
        with self.assertRaises(GeometryMeasureError):
            LineStringM(geom, m)

    def test_beg_m_end_m(self):
        """Test beg_m and end_m properties."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([0.0, 5.0, 10.0])
        lsm = LineStringM(geom, m)
        self.assertEqual(lsm.beg_m, 0.0)
        self.assertEqual(lsm.end_m, 10.0)

    def test_m_to_distance(self):
        """Test m_to_distance method and caching."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        m = np.array([0.0, 10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # Test exact M values at vertices
        self.assertAlmostEqual(lsm.m_to_distance(0.0), 0.0)
        self.assertAlmostEqual(lsm.m_to_distance(10.0), 1.0)
        self.assertAlmostEqual(lsm.m_to_distance(20.0), 2.0)
        self.assertAlmostEqual(lsm.m_to_distance(30.0), 3.0)
        
        # Test interpolated M values
        self.assertAlmostEqual(lsm.m_to_distance(5.0), 0.5)
        self.assertAlmostEqual(lsm.m_to_distance(15.0), 1.5)
        self.assertAlmostEqual(lsm.m_to_distance(25.0), 2.5)
        
        # Test that cache is created
        self.assertTrue(hasattr(lsm, '_cached_cumdist'))
        
        # Test that cache is invalidated when geometry changes
        new_geom = LineString([(0, 0), (2, 0)])
        lsm.geom = new_geom
        self.assertFalse(hasattr(lsm, '_cached_cumdist'))

    def test_distance_to_m(self):
        """Test distance_to_m method."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        m = np.array([0.0, 10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # Test exact distances at vertices
        self.assertAlmostEqual(lsm.distance_to_m(0.0), 0.0)
        self.assertAlmostEqual(lsm.distance_to_m(1.0), 10.0)
        self.assertAlmostEqual(lsm.distance_to_m(2.0), 20.0)
        self.assertAlmostEqual(lsm.distance_to_m(3.0), 30.0)
        
        # Test interpolated distances
        self.assertAlmostEqual(lsm.distance_to_m(0.5), 5.0)
        self.assertAlmostEqual(lsm.distance_to_m(1.5), 15.0)
        self.assertAlmostEqual(lsm.distance_to_m(2.5), 25.0)

    def test_m_to_distance_roundtrip(self):
        """Test round-trip conversion: M -> distance -> M."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        m = np.array([0.0, 5.0, 15.0, 25.0, 40.0])
        lsm = LineStringM(geom, m)
        
        # Test round-trip at various M values
        test_m_values = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 32.5, 40.0]
        for m_val in test_m_values:
            distance = lsm.m_to_distance(m_val)
            m_back = lsm.distance_to_m(distance)
            self.assertAlmostEqual(m_val, m_back, places=10,
                msg=f"Round-trip failed for M={m_val}: got {m_back}")

    def test_distance_to_m_roundtrip(self):
        """Test round-trip conversion: distance -> M -> distance."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        m = np.array([0.0, 5.0, 15.0, 25.0, 40.0])
        lsm = LineStringM(geom, m)
        
        # Test round-trip at various distances
        test_distances = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        for dist in test_distances:
            m_val = lsm.distance_to_m(dist)
            dist_back = lsm.m_to_distance(m_val)
            self.assertAlmostEqual(dist, dist_back, places=10,
                msg=f"Round-trip failed for distance={dist}: got {dist_back}")

    def test_m_to_distance_nonuniform(self):
        """Test m_to_distance with non-uniform M values."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([0.0, 5.0, 25.0])  # Non-uniform spacing
        lsm = LineStringM(geom, m)
        
        # Test interpolation in first segment
        self.assertAlmostEqual(lsm.m_to_distance(2.5), 0.5)
        # Test interpolation in second segment
        self.assertAlmostEqual(lsm.m_to_distance(15.0), 1.5)

    def test_distance_to_m_nonuniform(self):
        """Test distance_to_m with non-uniform M values."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([0.0, 5.0, 25.0])  # Non-uniform spacing
        lsm = LineStringM(geom, m)
        
        # Test interpolation in first segment (M changes by 5 over distance 1)
        self.assertAlmostEqual(lsm.distance_to_m(0.5), 2.5)
        # Test interpolation in second segment (M changes by 20 over distance 1)
        self.assertAlmostEqual(lsm.distance_to_m(1.5), 15.0)

    def test_m_to_distance_normalized(self):
        """Test m_to_norm_distance method."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        m = np.array([0.0, 10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # Test normalized distances
        self.assertAlmostEqual(lsm.m_to_norm_distance(0.0), 0.0)
        self.assertAlmostEqual(lsm.m_to_norm_distance(15.0), 0.5)
        self.assertAlmostEqual(lsm.m_to_norm_distance(30.0), 1.0)

    def test_distance_to_m_normalized(self):
        """Test distance_to_m with normalized distances."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        m = np.array([0.0, 10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # Test normalized distances
        self.assertAlmostEqual(lsm.distance_to_m(0.0, normalized=True), 0.0)
        self.assertAlmostEqual(lsm.distance_to_m(0.5, normalized=True), 15.0)
        self.assertAlmostEqual(lsm.distance_to_m(1.0, normalized=True), 30.0)

    def test_m_to_distance_snapping(self):
        """Test m_to_distance with snapping enabled."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # Test snapping to beginning
        self.assertAlmostEqual(lsm.m_to_distance(5.0, snap=True), 0.0)
        # Test snapping to end
        self.assertAlmostEqual(lsm.m_to_distance(35.0, snap=True), 2.0)
        # Test error without snapping
        with self.assertRaises(ValueError):
            lsm.m_to_distance(5.0, snap=False)
        with self.assertRaises(ValueError):
            lsm.m_to_distance(35.0, snap=False)

    def test_distance_to_m_snapping(self):
        """Test distance_to_m with snapping enabled."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # Test snapping to beginning
        self.assertAlmostEqual(lsm.distance_to_m(-0.5, snap=True), 10.0)
        # Test snapping to end
        self.assertAlmostEqual(lsm.distance_to_m(2.5, snap=True), 30.0)
        # Test error without snapping
        with self.assertRaises(ValueError):
            lsm.distance_to_m(-0.5, snap=False)
        with self.assertRaises(ValueError):
            lsm.distance_to_m(2.5, snap=False)

    def test_m_to_distance_diagonal_line(self):
        """Test m_to_distance on a diagonal line."""
        geom = LineString([(0, 0), (1, 1), (2, 2)])
        m = np.array([0.0, 10.0, 20.0])
        lsm = LineStringM(geom, m)
        
        # Distance should be sqrt(2) per segment
        expected_segment_length = np.sqrt(2)
        
        self.assertAlmostEqual(lsm.m_to_distance(0.0), 0.0)
        self.assertAlmostEqual(lsm.m_to_distance(10.0), expected_segment_length)
        self.assertAlmostEqual(lsm.m_to_distance(20.0), 2 * expected_segment_length)
        self.assertAlmostEqual(lsm.m_to_distance(5.0), expected_segment_length / 2)

    def test_distance_to_m_diagonal_line(self):
        """Test distance_to_m on a diagonal line."""
        geom = LineString([(0, 0), (1, 1), (2, 2)])
        m = np.array([0.0, 10.0, 20.0])
        lsm = LineStringM(geom, m)
        
        expected_segment_length = np.sqrt(2)
        
        self.assertAlmostEqual(lsm.distance_to_m(0.0), 0.0)
        self.assertAlmostEqual(lsm.distance_to_m(expected_segment_length), 10.0)
        self.assertAlmostEqual(lsm.distance_to_m(2 * expected_segment_length), 20.0)
        self.assertAlmostEqual(lsm.distance_to_m(expected_segment_length / 2), 5.0)

    def test_m_to_distance_errors(self):
        """Test error handling in m_to_distance."""
        geom = LineString([(0, 0), (1, 0)])
        
        # Test with undefined M values
        lsm_no_m = LineStringM(geom)
        with self.assertRaises(ValueError):
            lsm_no_m.m_to_distance(5.0)

    def test_distance_to_m_errors(self):
        """Test error handling in distance_to_m."""
        geom = LineString([(0, 0), (1, 0)])
        
        # Test with undefined M values
        lsm_no_m = LineStringM(geom)
        with self.assertRaises(ValueError):
            lsm_no_m.distance_to_m(0.5)

    def test_conversion_precision(self):
        """Test that conversions maintain high precision."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        m = np.array([0.0, 10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # Test many values for precision
        for i in range(100):
            m_val = i * 0.3
            if m_val <= 30.0:
                distance = lsm.m_to_distance(m_val)
                m_back = lsm.distance_to_m(distance)
                self.assertAlmostEqual(m_val, m_back, places=10)

    def test_conversion_performance_consistency(self):
        """Test that conversions produce consistent results with caching."""
        geom = LineString([(0, 0), (1, 0), (2, 0), (3, 0)])
        m = np.array([0.0, 10.0, 20.0, 30.0])
        lsm = LineStringM(geom, m)
        
        # First call creates cache
        dist1 = lsm.m_to_distance(15.0)
        # Second call uses cache
        dist2 = lsm.m_to_distance(15.0)
        
        self.assertEqual(dist1, dist2)
        
        # Test distance_to_m as well
        m1 = lsm.distance_to_m(1.5)
        m2 = lsm.distance_to_m(1.5)
        
        self.assertEqual(m1, m2)


class TestSubstringMCoords(unittest.TestCase):
    """Test substring_m_coords function including edge cases."""

    def setUp(self):
        """Set up common test data."""
        # Simple straight line
        self.coords_simple = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.m_simple = np.array([0.0, 10.0, 20.0, 30.0])
        
        # Line with non-uniform M values
        self.coords_nonuniform = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.m_nonuniform = np.array([0.0, 5.0, 15.0])
        
        # Diagonal line
        self.coords_diagonal = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        self.m_diagonal = np.array([0.0, 10.0, 20.0])

    def test_full_extent(self):
        """Test extracting the full extent of a linestring."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 0.0, 3.0
        )
        np.testing.assert_array_almost_equal(coords, self.coords_simple)
        np.testing.assert_array_almost_equal(m, self.m_simple)

    def test_substring_at_vertices(self):
        """Test extracting a substring between existing vertices."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 1.0, 2.0
        )
        expected_coords = np.array([[1.0, 0.0], [2.0, 0.0]])
        expected_m = np.array([10.0, 20.0])
        np.testing.assert_array_almost_equal(coords, expected_coords)
        np.testing.assert_array_almost_equal(m, expected_m)

    def test_substring_between_vertices(self):
        """Test extracting a substring with interpolated endpoints."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 0.5, 2.5
        )
        # Start should be interpolated between (0,0) and (1,0)
        # End should be interpolated between (2,0) and (3,0)
        self.assertEqual(len(coords), 4)  # Interpolated start, two vertices, interpolated end
        np.testing.assert_array_almost_equal(coords[0], [0.5, 0.0])
        np.testing.assert_array_almost_equal(coords[-1], [2.5, 0.0])
        # Check M values are interpolated correctly
        self.assertAlmostEqual(m[0], 5.0)  # Halfway between 0 and 10
        self.assertAlmostEqual(m[-1], 25.0)  # Halfway between 20 and 30

    def test_adjacent_substrings_boundary_consistency(self):
        """Test that adjacent substrings have matching boundaries."""
        # First substring from 0 to 1.5
        coords1, m1 = substring_m_coords(
            self.coords_simple, self.m_simple, 0.0, 1.5
        )
        # Second substring from 1.5 to 3.0
        coords2, m2 = substring_m_coords(
            self.coords_simple, self.m_simple, 1.5, 3.0
        )
        # The end of first should match the start of second
        np.testing.assert_array_almost_equal(coords1[-1], coords2[0])
        self.assertAlmostEqual(m1[-1], m2[0])

    def test_multiple_adjacent_substrings(self):
        """Test multiple adjacent substrings all have consistent boundaries."""
        splits = [0.0, 0.7, 1.3, 2.1, 3.0]
        substrings = []
        
        for i in range(len(splits) - 1):
            coords, m = substring_m_coords(
                self.coords_simple, self.m_simple, splits[i], splits[i+1]
            )
            substrings.append((coords, m))
        
        # Check all boundaries match
        for i in range(len(substrings) - 1):
            coords1, m1 = substrings[i]
            coords2, m2 = substrings[i+1]
            np.testing.assert_array_almost_equal(
                coords1[-1], coords2[0],
                err_msg=f"Boundary mismatch between substring {i} and {i+1}"
            )
            self.assertAlmostEqual(
                m1[-1], m2[0],
                msg=f"M value mismatch between substring {i} and {i+1}"
            )

    def test_start_beyond_end(self):
        """Test that start > end raises ValueError."""
        with self.assertRaises(ValueError):
            substring_m_coords(self.coords_simple, self.m_simple, 2.0, 1.0)

    def test_start_at_end_point(self):
        """Test substring starting at the end of the linestring."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 3.0, 3.0
        )
        # Should return 2 identical points (valid for LineString creation)
        self.assertEqual(len(coords), 2)
        np.testing.assert_array_almost_equal(coords[0], coords[1])
        np.testing.assert_array_almost_equal(coords[0], self.coords_simple[-1])
        self.assertEqual(m[0], m[1])

    def test_end_at_start_point(self):
        """Test substring ending at the start of the linestring."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 0.0, 0.0
        )
        # Should return 2 identical points (valid for LineString creation)
        self.assertEqual(len(coords), 2)
        np.testing.assert_array_almost_equal(coords[0], coords[1])
        np.testing.assert_array_almost_equal(coords[0], self.coords_simple[0])
        self.assertEqual(m[0], m[1])

    def test_start_before_beginning(self):
        """Test substring with start < 0."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, -1.0, 1.5
        )
        # Should start at the beginning
        np.testing.assert_array_almost_equal(coords[0], self.coords_simple[0])
        self.assertEqual(m[0], self.m_simple[0])

    def test_end_beyond_end(self):
        """Test substring with end beyond the linestring length."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 1.5, 10.0
        )
        # Should end at the last point
        np.testing.assert_array_almost_equal(coords[-1], self.coords_simple[-1])
        self.assertEqual(m[-1], self.m_simple[-1])

    def test_normalized_mode(self):
        """Test substring extraction with normalized distances."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 0.25, 0.75, normalized=True
        )
        # 0.25 of 3.0 = 0.75, 0.75 of 3.0 = 2.25
        self.assertAlmostEqual(coords[0][0], 0.75)
        self.assertAlmostEqual(coords[-1][0], 2.25)

    def test_floating_point_precision(self):
        """Test that floating point errors don't create non-monotonic M values."""
        # Create a scenario prone to floating point errors
        coords = np.array([[0.0, 0.0], [0.3, 0.0], [0.7, 0.0], [1.0, 0.0]])
        m = np.array([0.0, 0.3, 0.7, 1.0])
        
        # Extract substring that might have precision issues
        result_coords, result_m = substring_m_coords(
            coords, m, 0.1, 0.9
        )
        
        # Check M values are monotonically non-decreasing
        m_diffs = np.diff(result_m)
        self.assertTrue(np.all(m_diffs >= 0), 
                       f"M values not monotonic: {result_m}, diffs: {m_diffs}")

    def test_diagonal_line_interpolation(self):
        """Test interpolation on a diagonal line."""
        # Extract from 0.5 to 1.5 on a diagonal
        coords, m = substring_m_coords(
            self.coords_diagonal, self.m_diagonal, 
            0.5 * np.sqrt(2), 1.5 * np.sqrt(2)
        )
        
        # Check that coordinates and M values are properly interpolated
        self.assertGreater(len(coords), 2)
        # M values should be monotonic
        self.assertTrue(np.all(np.diff(m) >= 0))

    def test_custom_tolerance(self):
        """Test using a custom tolerance parameter."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 0.0, 3.0, tolerance=1e-8
        )
        # Should still work with different tolerance
        self.assertGreater(len(coords), 0)
        self.assertTrue(np.all(np.diff(m) >= 0))

    def test_very_small_substring(self):
        """Test extracting a very small substring."""
        coords, m = substring_m_coords(
            self.coords_simple, self.m_simple, 1.0, 1.001
        )
        # Should have at least 2 points
        self.assertGreaterEqual(len(coords), 2)
        # M values should be close
        self.assertLess(m[-1] - m[0], 1.0)

    def test_nonuniform_m_interpolation(self):
        """Test interpolation with non-uniform M values."""
        # M values: [0, 5, 15] for coords at x=[0, 1, 2]
        coords, m = substring_m_coords(
            self.coords_nonuniform, self.m_nonuniform, 0.5, 1.5
        )
        
        # Start M should be interpolated: at x=0.5, M should be 2.5 (halfway to 5)
        self.assertAlmostEqual(m[0], 2.5, places=5)
        # End M should be interpolated: at x=1.5, M should be 10.0 (halfway between 5 and 15)
        self.assertAlmostEqual(m[-1], 10.0, places=5)

    def test_m_values_always_increase(self):
        """Test that M values never decrease even with floating point errors."""
        # Run many random substrings to catch potential precision issues
        np.random.seed(42)
        for _ in range(100):
            start = np.random.uniform(0, 2.5)
            end = np.random.uniform(start, 3.0)
            coords, m = substring_m_coords(
                self.coords_simple, self.m_simple, start, end
            )
            # Check monotonicity
            if len(m) > 1:
                m_diffs = np.diff(m)
                self.assertTrue(
                    np.all(m_diffs >= 0),
                    f"Non-monotonic M values for start={start}, end={end}: {m}"
                )

    def test_result_always_valid_for_linestring(self):
        """Test that results always have at least 2 coordinates for valid LineString."""
        test_cases = [
            (0.0, 0.0),  # Start at beginning
            (1.5, 1.5),  # Middle point
            (3.0, 3.0),  # End point
            (0.0, 3.0),  # Full extent
            (0.5, 2.5),  # Partial
        ]
        
        for start, end in test_cases:
            coords, m = substring_m_coords(
                self.coords_simple, self.m_simple, start, end
            )
            # Must have at least 2 coordinates for valid LineString
            self.assertGreaterEqual(
                len(coords), 2, 
                f"Result for substring({start}, {end}) has only {len(coords)} coordinates"
            )
            # Verify we can create a LineString from the result
            try:
                line = LineString(coords)
                self.assertIsInstance(line, LineString)
            except Exception as e:
                self.fail(f"Failed to create LineString from substring({start}, {end}): {e}")


class TestSubstringMCoordsEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios for substring_m_coords."""

    def test_two_point_line(self):
        """Test substring on a line with only two points."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        m = np.array([0.0, 10.0])
        
        result_coords, result_m = substring_m_coords(coords, m, 0.25, 0.75)
        
        # Should interpolate correctly
        self.assertEqual(len(result_coords), 2)
        self.assertAlmostEqual(result_coords[0][0], 0.25)
        self.assertAlmostEqual(result_coords[-1][0], 0.75)
        self.assertAlmostEqual(result_m[0], 2.5)
        self.assertAlmostEqual(result_m[-1], 7.5)

    def test_single_segment_full_extraction(self):
        """Test extracting a full single segment."""
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        m = np.array([0.0, 1.0])
        
        result_coords, result_m = substring_m_coords(
            coords, m, 0.0, np.sqrt(2)
        )
        
        np.testing.assert_array_almost_equal(result_coords, coords)
        np.testing.assert_array_almost_equal(result_m, m)

    def test_zero_length_substring(self):
        """Test extracting a zero-length substring at a point."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        m = np.array([0.0, 10.0, 20.0])
        
        result_coords, result_m = substring_m_coords(coords, m, 0.5, 0.5)
        
        # Should return 2 identical points (valid for LineString creation)
        self.assertEqual(len(result_coords), 2)
        np.testing.assert_array_almost_equal(result_coords[0], result_coords[1])
        self.assertAlmostEqual(result_coords[0][0], 0.5)
        self.assertAlmostEqual(result_m[0], 5.0)
        self.assertAlmostEqual(result_m[0], result_m[1])

    def test_boundary_at_vertex_exact(self):
        """Test that boundaries at exact vertices produce consistent results."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        m = np.array([0.0, 10.0, 20.0, 30.0])
        
        # Split at vertex (x=1.0)
        coords1, m1 = substring_m_coords(coords, m, 0.0, 1.0)
        coords2, m2 = substring_m_coords(coords, m, 1.0, 3.0)
        
        # Boundaries should match exactly
        np.testing.assert_array_equal(coords1[-1], coords2[0])
        self.assertEqual(m1[-1], m2[0])
        # Values should be exactly the vertex values
        np.testing.assert_array_equal(coords1[-1], [1.0, 0.0])
        self.assertEqual(m1[-1], 10.0)


class TestLineStringMReverse(unittest.TestCase):
    """Test LineStringM.reverse() method."""

    def test_reverse_raises_not_implemented(self):
        """Test that reverse raises NotImplementedError."""
        geom = LineString([(0, 0), (1, 0), (2, 0)])
        m = np.array([0.0, 1.0, 2.0])
        lsm = LineStringM(geom, m)
        with self.assertRaises(NotImplementedError):
            lsm.reverse()


if __name__ == '__main__':
    unittest.main()
