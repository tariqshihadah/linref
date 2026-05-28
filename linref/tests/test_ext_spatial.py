import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import linref
from linref.ext.spatial import generate_intersections


class TestGenerateIntersections(unittest.TestCase):
    """Test the standalone generate_intersections function."""

    def setUp(self):
        """Create routes with segments for testing."""
        self.gdf = gpd.GeoDataFrame(
            {
                'route_id': ['A', 'A', 'B', 'B'],
                'geometry': [
                    LineString([(0, 0), (10, 0)]),    # A seg 1
                    LineString([(10, 0), (20, 0)]),   # A seg 2
                    LineString([(5, -5), (5, 5)]),    # B seg 1 crosses A1
                    LineString([(15, -5), (15, 5)]),  # B seg 2 crosses A2
                ],
            },
            crs='EPSG:4326',
        )

    def test_touches_predicate(self):
        """Default predicate='touches' finds endpoint-sharing pairs."""
        result = generate_intersections(self.gdf)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(result.crs, self.gdf.crs)
        for col in ('geometry', 'index_left', 'index_right'):
            self.assertIn(col, result.columns)
        # A1 and A2 touch at (10,0); no other pairs touch
        self.assertEqual(len(result), 1)
        self.assertEqual(
            list(result.geometry), [Point(10, 0)]
        )
        np.testing.assert_array_equal(
            list(result['index_left'].values), [0]
        )
        np.testing.assert_array_equal(
            list(result['index_right'].values), [1]
        )

    def test_crosses_predicate(self):
        """predicate='crosses' finds interior crossings only."""
        result = generate_intersections(self.gdf, predicate='crosses')
        # A1×B1 and A2×B2 cross; A1×A2 only touch → 2
        self.assertEqual(len(result), 2)
        self.assertEqual(
            list(result.geometry), [Point(5, 0), Point(15, 0)]
        )
        np.testing.assert_array_equal(
            list(result['index_left'].values), [0, 1]
        )
        np.testing.assert_array_equal(
            list(result['index_right'].values), [2, 3]
        )

    def test_intersects_predicate(self):
        """predicate='intersects' finds all intersecting pairs."""
        result = generate_intersections(self.gdf, predicate='intersects')
        # A1×A2 (touch), A1×B1 (cross), A2×B2 (cross) = 3
        self.assertEqual(len(result), 3)
        self.assertEqual(
            list(result.geometry), [Point(5, 0), Point(10, 0), Point(15, 0)]
        )

    def test_no_intersection(self):
        """Parallel lines and single geometry return empty result."""
        parallel = gpd.GeoDataFrame(
            {'geometry': [
                LineString([(0, 0), (10, 0)]),
                LineString([(0, 5), (10, 5)]),
            ]},
            crs='EPSG:4326',
        )
        self.assertEqual(len(generate_intersections(parallel)), 0)
        single = parallel.iloc[[0]]
        self.assertEqual(len(generate_intersections(single)), 0)

    def test_index_labels_preserved(self):
        """Result references original index labels."""
        gdf = gpd.GeoDataFrame(
            {'geometry': [
                LineString([(0, 0), (5, 0)]),
                LineString([(5, 0), (10, 0)]),
            ]},
            index=[10, 20],
            crs='EPSG:4326',
        )
        result = generate_intersections(gdf)
        self.assertEqual(result['index_left'].iloc[0], 10)
        self.assertEqual(result['index_right'].iloc[0], 20)

    def test_exclude_groups(self):
        """Group exclusion removes same-group pairs."""
        result_all = generate_intersections(
            self.gdf, exclude_groups=None, predicate='intersects'
        )
        self.assertEqual(len(result_all), 3)
        # Exclude same route: removes A1×A2 → 2
        result_excl = generate_intersections(
            self.gdf, exclude_groups='route_id', predicate='intersects'
        )
        self.assertEqual(len(result_excl), 2)

    def test_exclude_groups_missing_column_raises(self):
        """Nonexistent column raises ValueError."""
        with self.assertRaises(ValueError):
            generate_intersections(self.gdf, exclude_groups='bad_col')

    def test_invalid_input_raises(self):
        """Non-GeoDataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            generate_intersections(pd.DataFrame({'a': [1]}))


class TestLRSAccessorGenerateIntersections(unittest.TestCase):
    """Test the LRS_Accessor.generate_intersections method."""

    def setUp(self):
        """Create a GeoDataFrame with LRS configured."""
        self.gdf = gpd.GeoDataFrame(
            {
                'route_id': ['A', 'A', 'B', 'B'],
                'geometry': [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(10, 0), (20, 0)]),
                    LineString([(5, -5), (5, 5)]),
                    LineString([(15, -5), (15, 5)]),
                ],
            },
            crs='EPSG:4326',
        )
        self.gdf.lr.set_lrs(
            key_col='route_id', geom_col='geometry', inplace=True
        )

    def test_default_uses_key_col(self):
        """Default exclude_groups=True uses LRS key columns."""
        # Default excludes same-route pairs; with 'intersects': A1×B1, A2×B2 = 2
        result = self.gdf.lr.generate_intersections(predicate='intersects')
        self.assertEqual(len(result), 2)

    def test_exclude_groups_false(self):
        """exclude_groups=False includes same-route pairs."""
        result = self.gdf.lr.generate_intersections(
            exclude_groups=False, predicate='intersects'
        )
        self.assertEqual(len(result), 3)

    def test_predicate_forwarded(self):
        """predicate parameter is forwarded correctly."""
        touches = self.gdf.lr.generate_intersections(predicate='touches')
        crosses = self.gdf.lr.generate_intersections(predicate='crosses')
        # touches: A1×A2 is same-route (excluded) → 0
        self.assertEqual(len(touches), 0)
        # crosses: A1×B1, A2×B2 are cross-route → 2
        self.assertEqual(len(crosses), 2)

    def test_ungrouped_lrs(self):
        """Ungrouped LRS with default resolves to no exclusion."""
        gdf = gpd.GeoDataFrame(
            {'geometry': [
                LineString([(0, 0), (5, 0)]),
                LineString([(5, 0), (10, 0)]),
            ]},
            crs='EPSG:4326',
        )
        gdf.lr.set_lrs(geom_col='geometry', inplace=True)
        # touches with no exclusion → 1
        self.assertEqual(len(gdf.lr.generate_intersections()), 1)


# Run tests
if __name__ == '__main__':
    unittest.main()
