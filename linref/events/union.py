"""
===============================================================================

Module featuring classes and functionality for unifying events collections.


Classes
-------
EventsUnion


Dependencies
------------
pandas, numpy, copy, warnings, functools


Development
-----------
Developed by:
Tariq Shihadah, tariq.shihadah@gmail.com

Created:
4/13/2022

Modified:
4/13/2022

===============================================================================
"""


################
# DEPENDENCIES #
################

import pandas as pd
import numpy as np
import copy, warnings
from functools import wraps
from rangel import RangeCollection


class EventsUnion(object):
    """
    Parameters
    ----------
    objs : list-like of EventsCollection instances
        A selection of EventsCollection object instances to be combined 
        into a single instance based on the input parameters.
    **kwargs
        Keyword arguments to be passed to the initialization function for 
        the new EventsCollection instance.
    """

    def __init__(self, objs, **kwargs):
        self.objs = objs

    @property
    def objs(self):
        return self._objs

    @objs.setter
    def objs(self, objs):
        # Validate input object types
        if check_compatibility(objs):
            self._objs = objs

    @property
    def group_keys_unique(self):
        return list(set(
            key for obj in self.objs for key in obj.group_keys_unique))

    def get_groups(self, keys, empty=True):
        """
        Retrieve unique groups of events from each related collection based on 
        provided key values.

        Parameters
        ----------
        keys : key value, tuple of key values, or list of the same
            If only one key column is defined within the collections, a single 
            column value may be provided. Otherwise, a tuple of column values 
            must be provided in the same order as they appear in self.keys. To 
            get multiple groups, a list of key values or tuples may be 
            provided.
        empty : bool, default True
            Whether to allow for empty events groups to be returned when the 
            provided keys are valid but are not associated with any actual 
            events. If False, these cases will return a KeyError.
        """
        # Retrieve groups from all collections
        groups = []
        for obj in self._objs:
            group = obj.get_group(keys, empty=empty)
            groups.append(group)
        return groups

    def union(self, fill_gaps=False):
        """
        Parameters
        ----------
        fill_gaps : bool, default False
            Whether to fill gaps in merged collection with empty events. These 
            events would not be associated with any parent collection.
        """
        # Iterate over all unique group keys
        records = []
        for keys in self.group_keys_unique:
            # Iterate over groups
            groups = self.get_groups(keys, empty=True)
            ranges = []
            sources = []
            for i, group in enumerate(groups):
                # Create range data for group
                begs = group.df[group.beg].values
                ends = group.df[group.end].values
                rc = RangeCollection(
                    begs=begs, ends=ends, sort=False, copy=False)
                ranges.append(rc)
                # Prepare source data
                source = group.df[self._objs[i].others]
                source = \
                    source.append(pd.Series(dtype=float), ignore_index=True)
                sources.append(source)
            # Union ranges
            rc, indices = RangeCollection.union(
                ranges, fill_gaps=fill_gaps, return_index=True, null_index=-1)
            # Prepare dataframes
            basedata = {col: [val] * rc.num_ranges for col, val \
                in zip(self.objs[0].keys, keys)}
            dfs = [pd.DataFrame({**basedata,
                self.objs[0].beg: rc.begs,
                self.objs[0].end: rc.ends,
            })]
            dfs += [source.iloc[index].reset_index(drop=True) \
                for index, source in zip(indices, sources)]
            records.append(pd.concat(dfs, axis=1))
        return pd.concat(records)


#####################
# LATE DEPENDENCIES #
#####################

from linref.events.collection import EventsCollection, check_compatibility