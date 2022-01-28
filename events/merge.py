"""
===============================================================================

Module featuring classes and functionality for merging events collections and 
summarizing/retrieving information from these merges. For ease of use, features 
in this module should be accessed through collection-level merging methods such 
as EventsCollection.merge instead of abstractly through the classes themselves.


Classes
-------
EventsMerge, EventsMergeAttribute, EventsMergeTrace


Dependencies
------------
pandas, numpy, copy, warnings, functools


Development
-----------
Developed by:
Tariq Shihadah, tariq.shihadah@gmail.com

Created:
10/1/2021

Modified:
10/1/2021

===============================================================================
"""


################
# DEPENDENCIES #
################

import pandas as pd
import numpy as np
import copy, warnings
from functools import wraps


class EventsMerge(object):
    
    def __init__(self, left, right):
        # Log parameters
        self.left = left
        self.right = right
        
    def __getitem__(self, column):
        return EventsMergeAttribute(self, column)

    def __repr__(self):
        text = (
            f"left:  {self.left}\n"
            f"right: {self.right}\n"
            f"(traces {'built' if hasattr(self, '_traces') else 'not built'})"
        )
        return text
        
    @property
    def left(self):
        return self._left
    
    @left.setter
    def left(self, obj):
        self._validate_target(obj)
        self._left = obj
        
    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self, obj):
        # Validate and set
        self._validate_target(obj, left=False)
        self._right = obj
        
    @property
    def traces(self):
        return self._traces
    
    @traces.setter
    def traces(self, obj):
        if not isinstance(obj, list):
            raise TypeError("Traces must be a list of EventsMerge_trace "
                            "objects.")
        self._traces = obj
        
    @property
    def num_keys(self):
        return self.left.num_keys
        
    @property
    def keys(self):
        return self.left.keys
    
    @property
    def columns(self):
        return self.right.columns
        
    def _validate_target(self, obj, left=True):
        # Ensure left is set first if the target is the right
        if not (left) and not (hasattr(self, '_left')):
            raise AttributeError("The left target must be set before the right "
                                 "target.")
        # Ensure type
        if not isinstance(obj, EventsCollection):
            raise TypeError("EventsMerge targets must be EventsCollections.")
        # Ensure matching keys
        try:
            assert obj.num_keys == self.num_keys
        except AttributeError:
            pass
        except AssertionError:
            raise ValueError("Input EventsMerge target must have the same "
                             "number of keys as the existing left target "
                             f"({self.num_keys}).")
            
    def copy(self, deep=False):
        """
        Create an exact copy of the events class instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
    
    def build(self, inplace=False):
        """
        Perform intersects and overlays to produce EventsMergeTrace objects 
        for aggregation.
        """
        def _build_row(key, beg, end):
            try:
                # Retrieve corresponding events group and intersect/overlay
                eg = self.right.get_group(key, empty=False)
                mask = eg.intersecting(beg, end, mask=True)
                weights = eg.overlay(beg, end, normalize=False, arr=True)
                return EventsMergeTrace(key, beg, end, mask, weights, 
                                        success=True)
            except KeyError as e:
                return EventsMergeTrace(success=False)
        
        # In place?
        em = self if inplace else self.copy()

        # Build all traces
        em.traces = [_build_row(key, beg, end) for key, beg, end in \
                     zip(self.left.group_keys, self.left.begs, self.left.ends)]
        return em if not inplace else None

    def cut(self, **kwargs):
        """
        Cut intersecting event routes at the intersecting begin and end 
        locations, returning the resulting route's geometry or the route itself 
        if requested.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        return_mls : bool, default True
            Whether to return the MultiLineString associated with each cut 
            MLRRoute instead of the route itself.
        """
        # Get attribute for routes if available
        try:
            ema = self[self.right.route]
        except:
            raise ValueError("Right collection does not contain a valid "
                "routes column label.")
        # Perform cut
        return ema.cut(**kwargs)

class EventsMergeAttribute(object):
    
    def __init__(self, parent, column):
        self.parent = parent
        self.column = column
        
    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self, obj):
        if not isinstance(obj, EventsMerge):
            raise TypeError("Input parent must be EventsMerge type.")
        self._parent = obj
        
    @property
    def column(self):
        return self._column
    
    @column.setter
    def column(self, label):
        if not label in self.parent.columns:
            raise ValueError("Invalid column label for merged events. Must be "
                             "present in EventsMerge.columns.")
        self._column = label
        self._loc = self.parent.right.columns.index(label)
        
    @property
    def traces(self):
        try:
            return self.parent.traces
        except AttributeError:
            self.parent.build(inplace=True)
            return self.parent.traces
    
    @property
    def loc(self):
        return self._loc

    def _to_series(self, arr):
        sr = pd.Series(
            data=arr,
            index=self.parent.left.df.index,
            name=self.column
        )
        return sr

    def _agg(self, func, empty=None, **kwargs):
        """
        Generic attribute aggregator.
        """
        # Validate fill value
        empty = np.nan if empty is None else empty
        # Iterate over events to create aggregated result
        res = []
        gen = zip(self.parent.left.group_keys, self.traces)
        for key, trace in gen:
            try:
                arr = self.parent.right.get_group(key, empty=False) \
                        .df.values[:, self.loc]
                res_i = func(arr, trace, **kwargs)
            except (IndexError, KeyError) as e:
                res_i = empty
            res.append(res_i)
        return res

    def agg(self, func, empty=None, **kwargs):
        """
        Return all values from intersecting events in an array, aggregated by 
        a provided aggregation function. The function will be passed an 
        numpy array of all values which intersect each event, in the order 
        that they appear in the target events dataframe.

        Parameters
        ----------
        func : callable
            Callable function which will be passed an array of intersecting 
            events attribute values.
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            res = func(arr[trace.mask])
            return res
        return self._to_series(self._agg(_func, empty=empty))

    def all(self, empty=None, **kwargs):
        """
        Return all values from intersecting events in an array.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            res = arr[trace.mask]
            return res
        return self._to_series(self._agg(_func, empty=empty))

    def cut(self, empty=None, return_mls=True):
        """
        Cut intersecting event routes at the intersecting begin and end 
        locations, returning the resulting route's geometry or the route itself 
        if requested.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        return_mls : bool, default True
            Whether to return the MultiLineString associated with each cut 
            MLRRoute instead of the route itself.
        """
        def _func(arr, trace, **kwargs):
            # Choose the first intersecting event and cut the route
            route = arr[trace.mask][0]
            try:
                res = route.cut(trace.beg, trace.end)
            except AttributeError:
                raise TypeError("EventsMergeAttribute must contain MLSRoute "
                    "objects to be cut.")
            return res.mls if return_mls else res
        return self._to_series(self._agg(_func, empty=empty))
    
    def first(self, empty=None):
        """
        Return the first event value according to the order of the provided 
        collection's events dataframe.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose the first intersecting event
            res = arr[trace.mask][0]
            return res
        return self._to_series(self._agg(_func, empty=empty))
    
    def last(self, empty=None):
        """
        Return the last event value according to the order of the provided 
        collection's events dataframe.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose the last intersecting event
            res = arr[trace.mask][-1]
            return res
        return self._to_series(self._agg(_func, empty=empty))

    def value_counts(self, empty=None):
        """
        Return a dictionary of all unique intersecting event values and their 
        occurence counts.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            res = {val: count for val, count in \
                   zip(*np.unique(arr[trace.mask], return_counts=True))}
            return res
        return self._to_series(self._agg(_func, empty=empty))
    
    def most(self, empty=None, dropna=False):
        """
        Return the event value associated with the greatest total overlay 
        length.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        dropna : boolean, default False
            Whether to drop np.nan values in intersecting events before 
            aggregating.
        """
        def _func(arr, trace, **kwargs):
            # Drop nan if requested
            if dropna:
                nanmask = ~np.isnan(arr.astype(float))
                arr = arr[nanmask]
                weights = trace.weights[nanmask]
            else:
                weights = trace.weights
            # Aggregate and add to result
            res = get_most(arr, weights)
            return res
        return self._to_series(self._agg(_func, empty=empty))
        
    def mode(self, empty=None):
        """
        Return the most frequent unique event value.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            res = get_mode(arr[trace.mask])
            return res
        return self._to_series(self._agg(_func, empty=empty))

    def sum(self, empty=None):
        """
        Return the sum of all event values.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            res = sum(arr[trace.mask])
            return res
        return self._to_series(self._agg(_func, empty=empty))

    def mean(self, empty=None, weighted=True, dropna=False):
        """
        Return an overlay length-weighted average of all event values. An 
        unweighted straight average can also be computed if weighted=True.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        weighted : boolean, default True
            Whether the computed average should be weighted. If False, an
            un-weighted average will be computed, giving all intersecting 
            values an equal weight.
        dropna : boolean, default False
            Whether to drop np.nan values before aggregating.
        """
        def _func(arr, trace, **kwargs):
            if weighted:
                # Drop nan if requested
                if dropna:
                    nanmask = ~np.isnan(arr.astype(float))
                    arr = arr[nanmask]
                    weights = trace.weights[nanmask]
                else:
                    weights = trace.weights
                numer = np.multiply(arr, weights).sum(axis=0)
                denom = weights.sum()
                res = np.divide(numer, denom) if denom > 0 else np.nan
            else:
                # Drop nan if requested
                arr = arr[trace.mask]
                if dropna:
                    nanmask = ~np.isnan(arr.astype(float))
                    arr = arr[nanmask]
                res = arr.mean()
            return res
        return self._to_series(self._agg(_func, empty=empty))


class EventsMergeTrace(object):
    
    def __init__(self, key=None, beg=None, end=None, mask=None, weights=None, 
                 success=True):
        self.key = key if not key is None else np.nan
        self.beg = beg if not beg is None else np.nan
        self.end = end if not end is None else np.nan
        self.mask = mask if not mask is None else np.nan
        self.weights = weights if not weights is None else np.nan
        self.success = success


###########
# HELPERS #
###########

def get_most(arr, weights):
    """
    Select the item from the input array which is associated with the highest 
    total weight from the weights array. Scores are computed by summing the 
    weights for each unique array value.
    
    Parameters
    ----------
    arr, weights : array-like
        Arrays of equal length of target values and weights associated with 
        each value.
    """
    # Enforce numpy arrays
    arr = np.asarray(arr)
    weights = np.asarray(weights)
    # Group and split sorted target array
    sorter = np.argsort(arr)
    unique, splitter = np.unique(arr[sorter], return_index=True)
    splitter = splitter[1:]
    # Split weights and aggregate
    splits = np.split(weights[sorter], splitter)
    scores = [x.sum() for x in splits]
    # Return the highest scoring item
    return unique[np.argmax(scores)]

def get_mode(arr):
    """
    Select the item from the input array which appears most frequently.
    
    Parameters
    ----------
    arr : array-like
        Array with target values
    """
    # Enforce numpy array
    arr = np.asarray(arr)
    # Find most frequent unique value and return
    unique, counts = np.unique(arr, return_counts=True)
    return unique[np.argmax(counts)]


#####################
# LATE DEPENDENCIES #
#####################

from linref.events.collection import EventsCollection
