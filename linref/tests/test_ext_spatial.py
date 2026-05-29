import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import linref
from linref.ext.spatial import generate_intersection_pairs, generate_intersection_nodes


class TestGenerateIntersectionPairs(unittest.TestCase):
    """Test the standalone generate_intersection_pairs function."""

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
        intersections, index_left, index_right = generate_intersection_pairs(self.gdf)
        # A1 and A2 touch at (10,0); no other pairs touch
        self.assertEqual(len(intersections), 1)
        self.assertEqual(intersections[0], Point(10, 0))
        np.testing.assert_array_equal(index_left, [0])
        np.testing.assert_array_equal(index_right, [1])

    def test_crosses_predicate(self):
        """predicate='crosses' finds interior crossings only."""
        intersections, index_left, index_right = generate_intersection_pairs(
            self.gdf, predicate='crosses'
        )
        # A1×B1 and A2×B2 cross; A1×A2 only touch → 2
        self.assertEqual(len(intersections), 2)
        self.assertEqual(list(intersections), [Point(5, 0), Point(15, 0)])
        np.testing.assert_array_equal(index_left, [0, 1])
        np.testing.assert_array_equal(index_right, [2, 3])

    def test_intersects_predicate(self):
        """predicate='intersects' finds all intersecting pairs."""
        intersections, index_left, index_right = generate_intersection_pairs(
            self.gdf, predicate='intersects'
        )
        # A1×A2 (touch), A1×B1 (cross), A2×B2 (cross) = 3
        self.assertEqual(len(intersections), 3)
        self.assertEqual(
            list(intersections), [Point(5, 0), Point(10, 0), Point(15, 0)]
        )

    def test_no_intersection(self):
        """Parallel lines and single geometry return empty arrays."""
        parallel = gpd.GeoDataFrame(
            {'geometry': [
                LineString([(0, 0), (10, 0)]),
                LineString([(0, 5), (10, 5)]),
            ]},
            crs='EPSG:4326',
        )
        intersections, index_left, index_right = generate_intersection_pairs(parallel)
        self.assertEqual(len(intersections), 0)
        single = parallel.iloc[[0]]
        intersections, _, _ = generate_intersection_pairs(single)
        self.assertEqual(len(intersections), 0)

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
        _, index_left, index_right = generate_intersection_pairs(gdf)
        self.assertEqual(index_left[0], 10)
        self.assertEqual(index_right[0], 20)

    def test_exclude_groups(self):
        """Group exclusion removes same-group pairs."""
        intersections_all, _, _ = generate_intersection_pairs(
            self.gdf, exclude_groups=None, predicate='intersects'
        )
        self.assertEqual(len(intersections_all), 3)
        # Exclude same route: removes A1×A2 → 2
        intersections_excl, _, _ = generate_intersection_pairs(
            self.gdf, exclude_groups='route_id', predicate='intersects'
        )
        self.assertEqual(len(intersections_excl), 2)

    def test_exclude_groups_missing_column_raises(self):
        """Nonexistent column raises ValueError."""
        with self.assertRaises(ValueError):
            generate_intersection_pairs(self.gdf, exclude_groups='bad_col')

    def test_invalid_input_raises(self):
        """Non-GeoDataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            generate_intersection_pairs(pd.DataFrame({'a': [1]}))


class TestGenerateIntersectionNodes(unittest.TestCase):
    """Test the standalone generate_intersection_nodes function."""

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

    def test_returns_tuple_of_arrays(self):
        """Result is a tuple of (geometry_array, indices_list)."""
        result = generate_intersection_nodes(self.gdf, predicate='touches')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        geoms, indices = result
        self.assertIsInstance(geoms, np.ndarray)
        self.assertIsInstance(indices, list)

    def test_touches_groups_correctly(self):
        """touches: A1×A2 at (10,0) → indices [0, 1]."""
        geoms, indices = generate_intersection_nodes(self.gdf, predicate='touches')
        self.assertEqual(len(geoms), 1)
        self.assertEqual(geoms[0], Point(10, 0))
        self.assertEqual(indices[0], [0, 1])

    def test_crosses_groups_correctly(self):
        """crosses: A1×B1 at (5,0), A2×B2 at (15,0)."""
        geoms, indices = generate_intersection_nodes(self.gdf, predicate='crosses')
        # Sort by x coordinate for stable comparison
        order = np.argsort([g.x for g in geoms])
        geoms = geoms[order]
        indices = [indices[i] for i in order]
        self.assertEqual(len(geoms), 2)
        self.assertEqual(geoms[0], Point(5, 0))
        self.assertEqual(indices[0], [0, 2])
        self.assertEqual(geoms[1], Point(15, 0))
        self.assertEqual(indices[1], [1, 3])

    def test_intersects_merges_shared_point(self):
        """Three lines meeting at one point produce a single node."""
        gdf = gpd.GeoDataFrame(
            {'geometry': [
                LineString([(0, 0), (5, 0)]),
                LineString([(5, 0), (10, 0)]),
                LineString([(5, -5), (5, 5)]),
            ]},
            crs='EPSG:4326',
        )
        geoms, indices = generate_intersection_nodes(gdf, predicate='intersects')
        # All three meet at (5,0) → single node with indices [0, 1, 2]
        self.assertEqual(len(geoms), 1)
        self.assertEqual(geoms[0], Point(5, 0))
        self.assertEqual(indices[0], [0, 1, 2])

    def test_exclude_groups(self):
        """Group exclusion removes same-group pairs before grouping."""
        geoms, indices = generate_intersection_nodes(
            self.gdf, exclude_groups='route_id', predicate='intersects'
        )
        # Excludes A1×A2 → only cross-route pairs remain: 2 nodes
        self.assertEqual(len(geoms), 2)

    def test_empty_result(self):
        """No intersections returns empty arrays."""
        parallel = gpd.GeoDataFrame(
            {'geometry': [
                LineString([(0, 0), (10, 0)]),
                LineString([(0, 5), (10, 5)]),
            ]},
            crs='EPSG:4326',
        )
        geoms, indices = generate_intersection_nodes(parallel)
        self.assertEqual(len(geoms), 0)
        self.assertEqual(indices, [])

    def test_single_geometry(self):
        """Single geometry returns empty result."""
        single = gpd.GeoDataFrame(
            {'geometry': [LineString([(0, 0), (10, 0)])]},
            crs='EPSG:4326',
        )
        geoms, indices = generate_intersection_nodes(single)
        self.assertEqual(len(geoms), 0)

    def test_index_labels_preserved(self):
        """Result references original index labels, not positional."""
        gdf = gpd.GeoDataFrame(
            {'geometry': [
                LineString([(0, 0), (5, 0)]),
                LineString([(5, 0), (10, 0)]),
            ]},
            index=[10, 20],
            crs='EPSG:4326',
        )
        geoms, indices = generate_intersection_nodes(gdf)
        self.assertEqual(indices[0], [10, 20])

    def test_invalid_input_raises(self):
        """Non-GeoDataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            generate_intersection_nodes(pd.DataFrame({'a': [1]}))


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
        result = self.gdf.lr.generate_intersections(
            predicate='intersects', project=False,
        )
        self.assertEqual(len(result), 2)

    def test_exclude_groups_false(self):
        """exclude_groups=False includes same-route pairs."""
        result = self.gdf.lr.generate_intersections(
            exclude_groups=False, predicate='intersects', project=False,
        )
        self.assertEqual(len(result), 3)

    def test_predicate_forwarded(self):
        """predicate parameter is forwarded correctly."""
        touches = self.gdf.lr.generate_intersections(
            predicate='touches', project=False,
        )
        crosses = self.gdf.lr.generate_intersections(
            predicate='crosses', project=False,
        )
        self.assertEqual(len(touches), 0)
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
        result = gdf.lr.generate_intersections(project=False)
        self.assertEqual(len(result), 1)

    def test_returns_geodataframe_with_indices(self):
        """Result is a GeoDataFrame with indices column, no index_left/right."""
        result = self.gdf.lr.generate_intersections(
            predicate='crosses', project=False,
        )
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertIn('indices', result.columns)
        self.assertIn('geometry', result.columns)
        self.assertNotIn('index_left', result.columns)
        self.assertNotIn('index_right', result.columns)
        for idx_list in result['indices']:
            self.assertIsInstance(idx_list, list)

    def test_grouped_indices_correct(self):
        """Verify indices are correctly grouped per intersection point."""
        result = self.gdf.lr.generate_intersections(
            predicate='crosses', project=False,
        )
        result = result.sort_values(
            'geometry', key=lambda s: s.apply(lambda g: g.x),
        ).reset_index(drop=True)
        self.assertEqual(result['indices'].iloc[0], [0, 2])
        self.assertEqual(result['indices'].iloc[1], [1, 3])

    def test_empty_result(self):
        """Empty result returns GeoDataFrame with correct schema."""
        result = self.gdf.lr.generate_intersections(
            predicate='touches', project=False,
        )
        self.assertEqual(len(result), 0)
        self.assertIn('geometry', result.columns)
        self.assertIn('indices', result.columns)

    def test_project_expand(self):
        """project=True, expand=True produces one row per line per node."""
        from linref.geometry import LineStringM
        gdf = gpd.GeoDataFrame(
            {
                'route_id': ['A', 'A', 'B'],
                'beg': [0.0, 10.0, 0.0],
                'end': [10.0, 20.0, 10.0],
                'geometry': [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(10, 0), (20, 0)]),
                    LineString([(5, -5), (5, 5)]),
                ],
                'geometry_m': [
                    LineStringM(LineString([(0, 0), (10, 0)]), m=[0.0, 10.0]),
                    LineStringM(LineString([(10, 0), (20, 0)]), m=[10.0, 20.0]),
                    LineStringM(LineString([(5, -5), (5, 5)]), m=[0.0, 10.0]),
                ],
            },
            crs='EPSG:4326',
        )
        gdf.lr.set_lrs(
            key_col='route_id', loc_col='beg', beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m', inplace=True,
        )
        result = gdf.lr.generate_intersections(
            predicate='crosses', project=True, expand=True,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result['route_id']), {'A', 'B'})
        np.testing.assert_array_almost_equal(
            sorted(result['beg'].values), [5.0, 5.0]
        )

    def test_project_no_expand(self):
        """project=True, expand=False produces one row per node."""
        from linref.geometry import LineStringM
        gdf = gpd.GeoDataFrame(
            {
                'route_id': ['A', 'A', 'B'],
                'beg': [0.0, 10.0, 0.0],
                'end': [10.0, 20.0, 10.0],
                'geometry': [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(10, 0), (20, 0)]),
                    LineString([(5, -5), (5, 5)]),
                ],
                'geometry_m': [
                    LineStringM(LineString([(0, 0), (10, 0)]), m=[0.0, 10.0]),
                    LineStringM(LineString([(10, 0), (20, 0)]), m=[10.0, 20.0]),
                    LineStringM(LineString([(5, -5), (5, 5)]), m=[0.0, 10.0]),
                ],
            },
            crs='EPSG:4326',
        )
        gdf.lr.set_lrs(
            key_col='route_id', loc_col='beg', beg_col='beg', end_col='end',
            geom_col='geometry', geom_m_col='geometry_m', inplace=True,
        )
        result = gdf.lr.generate_intersections(
            predicate='crosses', project=True, expand=False,
        )
        # One row; which route is non-deterministic for equidistant matches
        self.assertEqual(len(result), 1)
        self.assertIn(result['route_id'].iloc[0], {'A', 'B'})
        self.assertAlmostEqual(result['beg'].iloc[0], 5.0)


# Run tests
if __name__ == '__main__':
    unittest.main()
