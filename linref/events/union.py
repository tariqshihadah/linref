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
    def num_objs(self):
        return len(self.objs)

    @property
    def group_keys_unique(self):
        return list(set(
            key for obj in self.objs for key in obj.group_keys_unique))
    
    @property
    def num_keys(self):
        return self.objs[0].num_keys

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
        groups = [obj.get_group(keys, empty=empty) for obj in self._objs]
        return groups

    def union(
        self, 
        fill_gaps=False, 
        get_index=True, 
        merge=False, 
        suffixes=None, 
        **kwargs
        ):
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
        get_index : bool, default True
            Whether to produce columns relating each new record to the index of 
            the originating record in the input events dataframes. When this is 
            not necessary, setting to False may produce significant time 
            savings.
        merge : bool, default False
            Whether to merge columns from each original dataframe to the newly 
            created resegmented events collection dataframe. If not done during 
            the union, it can be done later by merging on the new 'index_i' 
            columns which correlate with the indices of the original 
            dataframes. To perform this merge manually, the get_index parameter 
            should be True.
        suffixes : list-like, default ['_0', ..., '_n']
            Sequence of length equal to the number of events collections being 
            unified, where each element is a string indicating the suffix to 
            add to overlapping column names in each corresponding events 
            dataframe. All entries must be unique.
        """
        # Validate suffixes
        if suffixes is None:
            suffixes = [f'_{i}' for i in range(self.num_objs)]
        else:
            try:
                assert len(suffixes) == self.num_objs
                assert len(set(suffixes)) == self.num_objs
                assert all(isinstance(suffix, str) for suffix in suffixes)
            except:
                raise ValueError(
                    "Input suffixes must be list-like of unique strings with "
                    "a length equal to the number of events collections being "
                    f"unified ({self.num_objs:,.0f}).")
        # Initialize new linear referencing data columns
        keys = []
        begs = []
        ends = []
        indices = []
        # Iterate over all unique group keys across all collections
        # For collections that do not contain a given group key, the resulting 
        # data will be left as null
        for group_key in self.group_keys_unique:
            # Get each group associated with the selected key across all 
            # collections being analyzed
            groups = self.get_groups(group_key, empty=True)
            # Retrieve the range data associated with each group being unified
            ranges = [group.rng for group in groups]
            # Union ranges
            if get_index:
                rc, index = RangeCollection.union(
                    ranges, fill_gaps=fill_gaps, return_index=True,
                    null_index=-1)
                # Reshape index arrays
                arrs = []
                for i, arr_i in enumerate(index):
                    try:
                        arr_i = np.where(
                            arr_i!=-1, groups[i].df.index.values[arr_i], np.nan)
                    except IndexError:
                        pass
                    arrs.append(arr_i)
                # Concatenate selected indices
                index = np.array(arrs).T
                indices.append(index)
            else:
                rc = RangeCollection.union(
                    ranges, fill_gaps=fill_gaps, return_index=False,
                    null_index=-1)
            # Log unified range results
            keys.append(np.tile(group_key, (rc.num_ranges, 1)))
            begs.append(rc.begs)
            ends.append(rc.ends)
            
        # Prepare resulting unified dataframe
        keys = np.concatenate(keys, axis=0)
        begs = np.concatenate(begs)
        ends = np.concatenate(ends)
        indices = np.concatenate(indices, axis=0)
        if get_index:
            indices[indices==-1] = np.nan
            data = pd.DataFrame({
                **{col: arr for col, arr in zip(self.objs[0].keys, keys.T)},
                **{self.objs[0].beg: begs, self.objs[0].end: ends},
                **{f'index_{i}': arr for i, arr in enumerate(indices.T)}
            })
        else:
            data = pd.DataFrame({
                **{col: arr for col, arr in zip(self.objs[0].keys, keys.T)},
                **{self.objs[0].beg: begs, self.objs[0].end: ends},
            })
        
        # Merge resegmented data with original dataframe columns
        if merge and get_index:
            for i, obj in enumerate(self.objs):
                suffixes_i = (None, suffixes[i])
                data = data.merge(
                    obj.df.drop(columns=self.objs[0].targets, errors='ignore'),
                    how='left', left_on=f'index_{i}', right_index=True, 
                    suffixes=suffixes_i, **kwargs)
            
        # Convert to events collection in the model of the first collection
        ec = self.objs[0].from_similar(data, geom=None)
        return ec


#####################
# LATE DEPENDENCIES #
#####################

from linref.events.collection import EventsCollection, check_compatibility