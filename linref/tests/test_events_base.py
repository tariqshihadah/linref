"""
Unit tests for linref.events.base module.

Tests cover:
- EventsData sorting (sort, argsort, sort_standard)
- EventsData analysis (find_same, find_inside)
"""

import unittest
import numpy as np
from linref.events.base import EventsData


class TestSorting(unittest.TestCase):
    """Tests for EventsData sorting methods."""

    def test_sort_ascending_and_descending(self):
        """Sort by single key ascending and descending."""
        e = EventsData(begs=[5, 1, 3, 2], ends=[8, 4, 6, 5])
        result = e.sort(by='begs')
        np.testing.assert_array_equal(result.begs, [1, 2, 3, 5])
        np.testing.assert_array_equal(result.ends, [4, 5, 6, 8])
        result = e.sort(by='begs', ascending=False)
        np.testing.assert_array_equal(result.begs, [5, 3, 2, 1])

    def test_sort_multi_key_mixed_ascending(self):
        """Sort by multiple keys with mixed ascending/descending."""
        e = EventsData(begs=[0, 0, 5, 5], ends=[10, 6, 9, 15])
        result = e.sort(by=['begs', 'ends'], ascending=[True, False])
        np.testing.assert_array_equal(result.begs, [0, 0, 5, 5])
        np.testing.assert_array_equal(result.ends, [10, 6, 15, 9])

    def test_sort_groups_descending(self):
        """Sort by non-numeric groups descending."""
        e = EventsData(begs=[3, 1, 2], ends=[6, 4, 5], groups=['C', 'A', 'B'])
        result = e.sort(by='groups', ascending=False)
        np.testing.assert_array_equal(result.groups, ['C', 'B', 'A'])
        np.testing.assert_array_equal(result.begs, [3, 2, 1])

    def test_sort_return_index(self):
        """Sort with return_index returns the sort indices."""
        e = EventsData(begs=[5, 1, 3], ends=[8, 4, 6])
        result, idx = e.sort(by='begs', return_index=True)
        np.testing.assert_array_equal(idx, [1, 2, 0])
        np.testing.assert_array_equal(result.begs, [1, 3, 5])

    def test_argsort_multi_key(self):
        """argsort with multiple keys and mixed ascending."""
        e = EventsData(begs=[1, 1, 2, 2], ends=[5, 3, 8, 6])
        idx = e.argsort(by=['begs', 'ends'], ascending=[True, False])
        np.testing.assert_array_equal(idx, [0, 1, 2, 3])
        idx = e.argsort(by=['begs', 'ends'], ascending=[False, True])
        np.testing.assert_array_equal(idx, [3, 2, 1, 0])

    def test_sort_standard(self):
        """sort_standard sorts by groups, begs, ends."""
        e = EventsData(
            begs=[2, 0, 1, 0],
            ends=[4, 2, 3, 1],
            groups=['B', 'A', 'A', 'B']
        )
        result = e.sort_standard()
        np.testing.assert_array_equal(result.groups, ['A', 'A', 'B', 'B'])
        np.testing.assert_array_equal(result.begs, [0, 1, 0, 2])
        np.testing.assert_array_equal(result.ends, [2, 3, 1, 4])

    def test_sort_ascending_list_wrong_length(self):
        """Raise error if ascending list length doesn't match by."""
        e = EventsData(begs=[1, 2], ends=[3, 4])
        with self.assertRaises(ValueError):
            e.sort(by=['begs', 'ends'], ascending=[True])


class TestFindSame(unittest.TestCase):
    """Tests for EventsData.find_same()."""

    def test_find_same_basic(self):
        """Identify duplicate beg/end pairs with different keep modes."""
        e = EventsData(begs=[0, 5, 0, 5], ends=[3, 8, 3, 8])
        np.testing.assert_array_equal(e.find_same(), [False, False, True, True])
        np.testing.assert_array_equal(
            e.find_same(keep='none'), [True, True, True, True])
        np.testing.assert_array_equal(
            e.find_same(keep='last'), [True, True, False, False])

    def test_find_same_no_duplicates(self):
        """No duplicates returns all False."""
        e = EventsData(begs=[0, 1, 2], ends=[3, 4, 5])
        np.testing.assert_array_equal(e.find_same(), [False, False, False])

    def test_find_same_grouped(self):
        """Same beg/end in different groups are not considered same."""
        e = EventsData(begs=[0, 0, 0, 0], ends=[3, 3, 3, 3], groups=['A', 'A', 'B', 'B'])
        np.testing.assert_array_equal(e.find_same(), [False, True, False, True])


class TestFindInside(unittest.TestCase):
    """Tests for EventsData.find_inside()."""

    def test_find_inside_basic(self):
        """Identify ranges strictly inside another."""
        e = EventsData(begs=[0, 2, 5, 1], ends=[10, 8, 7, 3])
        np.testing.assert_array_equal(e.find_inside(), [False, True, True, True])

    def test_find_inside_no_containment(self):
        """Non-overlapping ranges have no inside."""
        e = EventsData(begs=[0, 5, 10], ends=[4, 9, 14])
        np.testing.assert_array_equal(e.find_inside(), [False, False, False])

    def test_find_inside_enforce_edges(self):
        """enforce_edges detects shared-boundary containment."""
        # Shared beg: [0,5] inside [0,10] only with enforce_edges
        e = EventsData(begs=[0, 0], ends=[10, 5])
        np.testing.assert_array_equal(e.find_inside(enforce_edges=False), [False, False])
        np.testing.assert_array_equal(e.find_inside(enforce_edges=True), [False, True])
        # Shared end: [3,10] inside [0,10] only with enforce_edges
        e = EventsData(begs=[0, 3], ends=[10, 10])
        np.testing.assert_array_equal(e.find_inside(enforce_edges=False), [False, False])
        np.testing.assert_array_equal(e.find_inside(enforce_edges=True), [False, True])

    def test_find_inside_grouped(self):
        """Only compares within same group."""
        e = EventsData(
            begs=[0, 2, 0, 2], ends=[10, 8, 10, 8],
            groups=['A', 'A', 'B', 'B']
        )
        np.testing.assert_array_equal(e.find_inside(), [False, True, False, True])

    def test_find_inside_identical_ranges(self):
        """Identical ranges are NOT inside each other."""
        e = EventsData(begs=[0, 0], ends=[10, 10])
        np.testing.assert_array_equal(e.find_inside(enforce_edges=False), [False, False])
        np.testing.assert_array_equal(e.find_inside(enforce_edges=True), [False, False])


class TestSeparate(unittest.TestCase):
    """Tests for EventsData.separate()."""

    def test_separate_basic_overlap(self):
        """Adjacent overlapping events are split at center midpoint."""
        e = EventsData(begs=[0, 5], ends=[8, 12])
        # Centers: 4, 8.5 -> midpoint = 6.25
        r = e.separate()
        self.assertEqual(r.num_events, 2)
        self.assertAlmostEqual(r.ends[0], 6.25)
        self.assertAlmostEqual(r.begs[1], 6.25)

    def test_separate_no_overlap(self):
        """Non-overlapping events are unchanged."""
        e = EventsData(begs=[0, 5, 10], ends=[4, 9, 14])
        r = e.separate()
        np.testing.assert_array_equal(r.begs, [0, 5, 10])
        np.testing.assert_array_equal(r.ends, [4, 9, 14])

    def test_separate_eclipsed(self):
        """Eclipsed (fully contained) events are zeroed out."""
        e = EventsData(begs=[0, 2], ends=[10, 5])
        r = e.separate(drop_short=False)
        # [2,5] is inside [0,10], gets zeroed to its center (3.5)
        self.assertEqual(r.num_events, 2)
        self.assertAlmostEqual(r.begs[0], 0)
        self.assertAlmostEqual(r.ends[0], 10)
        self.assertAlmostEqual(r.begs[1], r.ends[1])  # zero-length

    def test_separate_eclipsed_drop_short(self):
        """Eclipsed events are dropped when drop_short=True."""
        e = EventsData(begs=[0, 2], ends=[10, 5])
        r = e.separate(drop_short=True)
        self.assertEqual(r.num_events, 1)
        np.testing.assert_array_equal(r.begs, [0])
        np.testing.assert_array_equal(r.ends, [10])

    def test_separate_identical_events(self):
        """Identical events: first kept, duplicates zeroed."""
        e = EventsData(begs=[0, 0, 5], ends=[3, 3, 8])
        r = e.separate(drop_short=True)
        self.assertEqual(r.num_events, 2)

    def test_separate_method_center(self):
        """Method 'center' splits at center midpoint."""
        e = EventsData(begs=[0, 5], ends=[8, 12])
        r = e.separate(method='center')
        mid = (4.0 + 8.5) / 2  # midpoint of centers
        self.assertAlmostEqual(r.ends[0], mid)
        self.assertAlmostEqual(r.begs[1], mid)

    def test_separate_method_left(self):
        """Method 'left' gives overlap to left event."""
        e = EventsData(begs=[0, 5], ends=[8, 12])
        r = e.separate(method='left')
        self.assertAlmostEqual(r.ends[0], 8.0)
        self.assertAlmostEqual(r.begs[1], 8.0)

    def test_separate_method_right(self):
        """Method 'right' gives overlap to right event."""
        e = EventsData(begs=[0, 5], ends=[8, 12])
        r = e.separate(method='right')
        self.assertAlmostEqual(r.ends[0], 5.0)
        self.assertAlmostEqual(r.begs[1], 5.0)

    def test_separate_grouped(self):
        """Groups are handled independently."""
        e = EventsData(
            begs=[0, 5, 0, 5], ends=[8, 12, 8, 12],
            groups=['A', 'A', 'B', 'B']
        )
        r = e.separate()
        # Both groups get same treatment
        self.assertAlmostEqual(r.ends[0], r.ends[2])
        self.assertAlmostEqual(r.begs[1], r.begs[3])

    def test_separate_single_event(self):
        """Single event returns unchanged."""
        e = EventsData(begs=[0], ends=[10])
        r = e.separate()
        np.testing.assert_array_equal(r.begs, [0])
        np.testing.assert_array_equal(r.ends, [10])

    def test_separate_invalid_method(self):
        """Invalid method raises ValueError."""
        e = EventsData(begs=[0, 5], ends=[8, 12])
        with self.assertRaises(ValueError):
            e.separate(method='invalid')


if __name__ == '__main__':
    unittest.main()
