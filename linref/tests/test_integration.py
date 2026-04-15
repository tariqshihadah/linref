"""
End-to-end integration tests for linref v1.0.

Tests cover the primary workflows described in the release plan:
load → set LRS → dissolve → relate → integrate, plus spatial projection,
geometry operations, and the HIN safety analysis workflow.

All tests use the built-in toy datasets (roadways, crashes, pavement) and
are independently runnable in any order.
"""

import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

import linref as lr
from linref import LRS, integrate


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests using built-in toy datasets."""

    @classmethod
    def setUpClass(cls):
        """Load toy datasets with LRS pre-configured."""
        cls.roads = lr.datasets.load('roadways', set_lrs=True)
        cls.crashes = lr.datasets.load('crashes', set_lrs=True)
        cls.pavement = lr.datasets.load('pavement', set_lrs=True)

    # ------------------------------------------------------------------
    # 1. Dataset loading and LRS configuration
    # ------------------------------------------------------------------

    def test_load_and_configure_lrs(self):
        """Load datasets with and without set_lrs; verify shape, columns,
        and LRS configuration."""
        # Without LRS -------------------------------------------------
        roads_raw = lr.datasets.load('roadways')
        crashes_raw = lr.datasets.load('crashes')
        pavement_raw = lr.datasets.load('pavement')

        self.assertEqual(roads_raw.shape[0], 10)
        self.assertEqual(crashes_raw.shape[0], 20)
        self.assertEqual(pavement_raw.shape[0], 14)

        # Raw frames should have a default (unconfigured) LRS
        self.assertIsNone(roads_raw.lr.beg_col)
        self.assertIsNone(roads_raw.lr.end_col)
        self.assertEqual(roads_raw.lr.key_col, [])

        # With LRS ----------------------------------------------------
        roads = self.roads.copy()
        crashes = self.crashes.copy()
        pavement = self.pavement.copy()

        for df in (roads, crashes, pavement):
            self.assertTrue(df.lr.is_lrs_set)
            self.assertEqual(df.lr.key_col, ['route'])

        # Linear-event specific checks
        self.assertEqual(roads.lr.beg_col, 'beg')
        self.assertEqual(roads.lr.end_col, 'end')
        self.assertEqual(roads.lr.lrs.closed, 'left_mod')

        # Point-event specific checks
        self.assertEqual(crashes.lr.loc_col, 'loc')

        # Route coverage should be identical across datasets
        self.assertEqual(
            set(roads['route'].unique()),
            set(crashes['route'].unique()),
        )
        self.assertEqual(
            set(roads['route'].unique()),
            set(pavement['route'].unique()),
        )

    # ------------------------------------------------------------------
    # 2. Dissolve and resegment
    # ------------------------------------------------------------------

    def test_dissolve_and_resegment(self):
        """Sort → dissolve → resegment and verify mileage conservation,
        segment lengths, and route coverage."""
        roads = self.roads.copy()
        sorted_roads = roads.lr.sort_standard()
        dissolved = sorted_roads.lr.dissolve()

        # Dissolve should merge 10 segments into 3 (one per route)
        self.assertEqual(dissolved.shape[0], 3)
        self.assertEqual(
            set(dissolved['route']),
            set(roads['route'].unique()),
        )

        # Original total mileage
        orig_miles = (roads['end'] - roads['beg']).sum()

        # Dissolved mileage must equal original
        dissolved_miles = (dissolved['end'] - dissolved['beg']).sum()
        self.assertAlmostEqual(orig_miles, dissolved_miles, places=6)

        # Resegment to 5-mile chunks
        reseg = dissolved.lr.resegment(length=5)
        reseg_miles = (reseg['end'] - reseg['beg']).sum()

        # Total mileage preserved
        self.assertAlmostEqual(orig_miles, reseg_miles, places=6)

        # All segments should be ≤ 5 miles (with float tolerance)
        max_seg = (reseg['end'] - reseg['beg']).max()
        self.assertLessEqual(max_seg, 5.0 + 1e-9)

        # No gaps within each route
        for route in reseg['route'].unique():
            sub = reseg[reseg['route'] == route].sort_values('beg')
            begs = sub['beg'].values
            ends = sub['end'].values
            # Each segment's beg should equal the previous segment's end
            np.testing.assert_array_almost_equal(begs[1:], ends[:-1])

    # ------------------------------------------------------------------
    # 3. Point-to-linear relate
    # ------------------------------------------------------------------

    def test_point_to_linear_relate(self):
        """Relate crashes (point) to resegmented roads (linear) and verify
        aggregation methods preserve all data."""
        roads = self.roads.copy()
        dissolved = roads.lr.sort_standard().lr.dissolve()
        reseg = dissolved.lr.resegment(length=5)

        relation = reseg.lr.relate(self.crashes)

        # count — total must equal number of crashes
        counts = relation.count()
        self.assertEqual(len(counts), reseg.shape[0])
        self.assertEqual(counts.sum(), 20)

        # list — flattened list must contain all crash IDs
        crash_lists = relation['crash_id'].list()
        all_ids = [cid for sublist in crash_lists for cid in sublist]
        self.assertEqual(
            sorted(all_ids),
            sorted(self.crashes['crash_id'].tolist()),
        )

        # sum — crash_id sum across segments should equal overall sum
        # (crash_id is numeric)
        id_sum = relation['crash_id'].sum()
        self.assertAlmostEqual(
            id_sum.sum(),
            self.crashes['crash_id'].sum(),
            places=6,
        )

        # mean — crash_id mean values should stay within original range
        id_mean = relation['crash_id'].mean()
        orig_min = self.crashes['crash_id'].min()
        orig_max = self.crashes['crash_id'].max()
        for val in id_mean:
            if not np.isnan(val):
                self.assertGreaterEqual(val, orig_min)
                self.assertLessEqual(val, orig_max)

    # ------------------------------------------------------------------
    # 4. Linear-to-linear relate
    # ------------------------------------------------------------------

    def test_linear_to_linear_relate(self):
        """Relate pavement (linear) to resegmented roads (linear) and verify
        length-weighted aggregations."""
        roads = self.roads.copy()
        dissolved = roads.lr.sort_standard().lr.dissolve()
        reseg = dissolved.lr.resegment(length=5)

        relation = reseg.lr.relate(self.pavement)

        # mean condition_rating — length-weighted
        means = relation['condition_rating'].mean()
        self.assertEqual(len(means), reseg.shape[0])
        # All values should be within the original range
        orig_min = self.pavement['condition_rating'].min()
        orig_max = self.pavement['condition_rating'].max()
        for val in means:
            if not np.isnan(val):
                self.assertGreaterEqual(val, orig_min - 1e-9)
                self.assertLessEqual(val, orig_max + 1e-9)

        # mode surface_type — should come from the original value set
        modes = relation['surface_type'].mode()
        valid_types = set(self.pavement['surface_type'].unique())
        for val in modes:
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                self.assertIn(val, valid_types)

    # ------------------------------------------------------------------
    # 5. Integrate multiple datasets
    # ------------------------------------------------------------------

    def test_integrate_multiple_datasets(self):
        """Integrate resegmented roads and pavement into least-common-interval
        events."""
        roads = self.roads.copy()
        dissolved = roads.lr.sort_standard().lr.dissolve()
        reseg = dissolved.lr.resegment(length=5)

        integrated = integrate([reseg, self.pavement])

        # Should have more segments than either input alone
        self.assertGreater(integrated.shape[0], reseg.shape[0])

        # Route coverage preserved
        self.assertEqual(
            set(integrated['route'].unique()),
            set(roads['route'].unique()),
        )

        # Verify least-common-interval: no integrated segment should straddle
        # any boundary from either input dataset.
        for route in integrated['route'].unique():
            int_sub = integrated[integrated['route'] == route]
            reseg_sub = reseg[reseg['route'] == route]
            pav_sub = self.pavement[self.pavement['route'] == route]

            # Collect all input boundaries
            input_edges = sorted(set(
                list(reseg_sub['beg']) + list(reseg_sub['end']) +
                list(pav_sub['beg']) + list(pav_sub['end'])
            ))

            for _, row in int_sub.iterrows():
                beg, end = row['beg'], row['end']
                for edge in input_edges:
                    # No input edge should fall strictly inside an integrated
                    # segment (within tolerance)
                    if beg + 1e-9 < edge < end - 1e-9:
                        self.fail(
                            f"Integrated segment [{beg}, {end}] on {route} "
                            f"straddles input edge {edge}"
                        )

        # Total mileage preserved
        orig_miles = (reseg['end'] - reseg['beg']).sum()
        int_miles = (integrated['end'] - integrated['beg']).sum()
        self.assertAlmostEqual(orig_miles, int_miles, places=6)

    # ------------------------------------------------------------------
    # 6. Spatial projection
    # ------------------------------------------------------------------

    def test_spatial_projection(self):
        """Project synthetic point features onto the road network and verify
        projected LRS values."""
        roads = self.roads.copy()
        roads_m = roads.lr.add_geom_m()

        # Build synthetic points near known road geometries
        points = gpd.GeoDataFrame(
            {'name': ['P1', 'P2', 'P3']},
            geometry=[
                Point(1.0, 0.2),   # near US-101
                Point(5.0, 1.0),   # near mid-network
                Point(8.0, 1.5),   # near end-of-network
            ],
            crs='EPSG:4326',
        )

        projected = roads_m.lr.project(points, buffer=2, nearest=True)

        # All points should have been matched
        self.assertEqual(projected.shape[0], 3)
        self.assertIn('route', projected.columns)
        self.assertIn('loc', projected.columns)
        self.assertIn('project_distance', projected.columns)

        # Each projected loc should be within the matched route's range
        for _, row in projected.iterrows():
            route = row['route']
            loc = row['loc']
            route_roads = roads[roads['route'] == route]
            route_min = route_roads['beg'].min()
            route_max = route_roads['end'].max()
            self.assertGreaterEqual(loc, route_min - 1e-9)
            self.assertLessEqual(loc, route_max + 1e-9)

        # Project distances should be non-negative
        self.assertTrue((projected['project_distance'] >= 0).all())

    # ------------------------------------------------------------------
    # 7. Geometry operations (interpolate + cut)
    # ------------------------------------------------------------------

    def test_geometry_operations(self):
        """Add M-geometries, then use interpolate_from and cut_from to
        generate derived geometries."""
        roads = self.roads.copy()
        roads_m = roads.lr.add_geom_m()

        # Verify M-geometry column was added
        self.assertIn('geometry_m', roads_m.columns)
        from linref.events.geometry import LineStringM
        for geom in roads_m['geometry_m']:
            self.assertIsInstance(geom, LineStringM)

        # Dissolve + resegment for cut target
        dissolved = roads_m.lr.sort_standard().lr.dissolve()
        dissolved_m = dissolved.lr.add_geom_m()
        reseg = dissolved.lr.resegment(length=5)

        # cut_from: cut resegmented geometries from dissolved geometries
        reseg_cut = reseg.lr.cut_from(dissolved_m)
        self.assertEqual(reseg_cut.shape[0], reseg.shape[0])
        for geom in reseg_cut.geometry:
            self.assertIsInstance(geom, (LineString, type(None)))
            if geom is not None:
                self.assertFalse(geom.is_empty)

        # interpolate_from: create point geometries for crashes
        crashes = self.crashes.copy()
        interp = crashes.lr.interpolate_from(roads_m)
        self.assertEqual(interp.shape[0], crashes.shape[0])
        for geom in interp.geometry:
            self.assertIsInstance(geom, Point)
            self.assertFalse(geom.is_empty)

    # ------------------------------------------------------------------
    # 8. HIN safety analysis workflow
    # ------------------------------------------------------------------

    def test_hin_workflow(self):
        """Full HIN workflow: dissolve → resegment → relate → distribute
        with linear decay."""
        roads = self.roads.copy()
        crashes = self.crashes.copy()

        dissolved = roads.lr.sort_standard().lr.dissolve()
        reseg = dissolved.lr.resegment(length=0.5)

        relation = reseg.lr.relate(crashes)

        # Distribute with linear decay
        scores = relation.distribute(
            decay_size=2,
            decay_func='linear',
        )

        self.assertEqual(len(scores), reseg.shape[0])

        # All scores non-negative
        self.assertTrue((scores >= -1e-12).all())

        # Total distributed score should equal number of crashes
        # (distribute normalizes per-event and sums, so total ≈ num crashes)
        self.assertAlmostEqual(scores.sum(), 20.0, places=4)

        # Simple count comparison
        counts = relation.count()
        self.assertEqual(counts.sum(), 20)

        # Segments with crashes should generally have higher scores
        # than segments far from any crash (a sanity check, not strict)
        has_crash = counts > 0
        if has_crash.any() and (~has_crash).any():
            mean_with = scores[has_crash].mean()
            mean_without = scores[~has_crash].mean()
            self.assertGreater(mean_with, mean_without)


if __name__ == '__main__':
    unittest.main()
