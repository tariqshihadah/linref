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

    def union(self, fill_gaps=False, suffix=True):
        """
        Combine multiple EventsCollection instances into a single instance, 
        creating least common intervals among all collections and maintaining 
        all event attributes. The resulting combined events will be used to 
        create and return an EventsCollection modeled after the first indexed 
        collection in self.objs.

        Parameters
        ----------
        fill_gaps : bool, default False
            Whether to fill gaps in the merged collection with empty events. 
            These events would not be associated with any parent collection and 
            would not be populated with any events attributes.
        suffix : bool, default True
            Whether to address repeating column labels by appending a suffix 
            to all repeating labels indicating the order in which they appear. 
            E.g., repeated column labels 'info' and 'info' would be converted 
            into 'info_1' and 'info_2'. If False, no modifications will be made 
            to column labels which may produce errors when instantiating or 
            utilizing the resulting EventsCollection.
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
        # Concatenate records
        df = pd.concat(records)
        # Address repeating column labels
        if suffix:
            cols = list(df)
            counts = [cols.count(col) for col in cols]
            suffixes = ['_' + str(cols[:i+1].count(col)) \
                for i, col in enumerate(cols)]
            labels = [col + (suffix if count > 1 else '') \
                for col, suffix, count in zip(cols, suffixes, counts)]
            df.columns = labels
        # Create events collection based on objs[0]
        ec = self.objs[0].from_similar(df, geom=None)
        return ec


#####################
# LATE DEPENDENCIES #
#####################

from linref.events.collection import EventsCollection, check_compatibility