import unittest
from unittest import TestCase

import pandas as pd
import linref
from linref import EventsCollection, EventsGroup, EventsMerge


class TestCollection(TestCase):

    def test_init_basic(self):
        ec = EventsCollection(df_basic, keys=['RID'], beg='BMP', end='EMP')
        self.assertIsInstance(ec, EventsCollection)

    def test_init_standard(self):
        ec = EventsCollection.from_standard(df_basic)
        self.assertIsInstance(ec, EventsCollection)

    def test_init_unsorted(self):
        ec_unsorted = EventsCollection.from_standard(df_unsorted)
        self.assertFalse(df_basic.equals(ec_unsorted.df))


class TestIntersecting(TestCase):

    def test_intersecting(self):
        """
        Test basic use of intersecting on EventsGroup
        """
        ec = EventsCollection.from_standard(df_basic)
        ec_intersecting = ec['A'].intersecting(0.55, 1.05)
        self.assertTrue(ec_intersecting.equals(df_basic.iloc[1:4]))


class TestDissolve(TestCase):

    def test_dissolve(self):
        """
        Test basic use of dissolve on EventsCollection
        """
        ec = EventsCollection.from_standard(df_basic)
        df_res = ec.dissolve(attr=['A'], aggs=['B'], agg_func=lambda l: [i.upper() for i in l]).df
        self.assertTrue(df_res.equals(df_dissolve))


# Define standard unit test variables
df_basic = pd.DataFrame(
    data={
        'RID': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'BMP': [0.0, 0.5, 0.6, 0.9, 1.2, 1.6, 0.0, 1.2, 1.5, 1.0, 2.2, 2.9],
        'EMP': [0.5, 0.6, 0.9, 1.2, 1.6, 2.0, 1.2, 1.5, 1.8, 2.2, 2.9, 3.3],
        'A':   ['a', 'a', 'b', 'b', 'b', 'a', 'c', 'c', 'c', 'b', 'a', 'a'],
        'B':   ['b', 'b', 'b', 'b', 'c', 'b', 'b', 'b', 'b', 'c', 'b', 'a'],
    }
)
df_unsorted = df_basic.sample(frac=1, random_state=42)
df_dissolve = pd.DataFrame(
    data={
        'RID': {0: 'A', 2: 'A', 1: 'A', 3: 'B', 5: 'C', 4: 'C'},
        'BMP': {0: 0.0, 2: 0.6, 1: 1.6, 3: 0.0, 5: 1.0, 4: 2.2},
        'EMP': {0: 0.6, 2: 1.6, 1: 2.0, 3: 1.8, 5: 2.2, 4: 3.3},
        'A': {0: 'a', 2: 'b', 1: 'a', 3: 'c', 5: 'b', 4: 'a'},
        'B_agg': {0: ['B', 'B'], 2: ['B', 'B', 'C'], 1: ['B'], 3: ['B', 'B', 'B'], 5: ['C'], 4: ['B', 'A']}
    }
)
ec_basic = EventsCollection(df_basic, keys=['RID'], beg='BMP', end='EMP')

if __name__ == '__main__':
    unittest.main()