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
pandas, numpy, rangel, copy, warnings, functools


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
from rangel import RangeCollection


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
        # Multiple labels provided
        if isinstance(label, (list, tuple, np.ndarray)):
            if not set(label).issubset(self.parent.columns):
                raise ValueError("All column labels must be present in "
                                 "EventsMerge.columns.")
            # Process input
            self._ndim = 2
            self._ncols = len(label)
            self._loc = [self.parent.right.columns.index(i) for i in label]
            self._column = list(label)
        # Single label provided
        else:
            if not label in self.parent.columns:
                raise ValueError("Invalid column label for merged events. Must "
                                 "be present in EventsMerge.columns.")
            # Process input
            self._ndim = 1
            self._ncols = 1
            self._loc = [self.parent.right.columns.index(label)]
            self._column = label
        
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

    @property
    def ncols(self):
        return self._ncols

    @property
    def ndim(self):
        return self._ndim

    def _to_pandas(self, index, data, as_series=False, squeeze=True):
        # Create pandas object
        if as_series:
            obj = pd.Series(
                data=data,
                index=index,
                name=self.column
            )
        else:
            obj = pd.DataFrame(
                data=data,
                index=index,
                columns=self.column if self._ndim != 1 else [self.column]
            )
            if self._ndim == 1 and squeeze:
                obj = obj.iloc[:, 0]
        return obj

    def _get_empty(self, empty=None):
        return np.nan if empty is None else empty

    def _agg(self, func, empty=None, as_array=True, **kwargs):
        """
        Generic attribute aggregator.

        Aggregation functions will be passed a single EventsMergeTrace instance 
        for each unique EventsGroup in the left EventsCollection, along with an 
        array of the values in the right EventsGroup indexed by the selected 
        columns.
        """
        # Validate fill value
        empty = self._get_empty(empty)
        # Iterate over events to create aggregated result
        index = []
        res = []
        for trace in self.traces:
            # Where actual traces are created, perform aggregation function
            if trace.success:
                arr = trace.group_right.df.values[:, self.loc]
                res_i = func(arr, trace, **kwargs)
            # Otherwise create a full empty array
            else:
                res_i = np.full(
                    (trace.group_left.df.shape[0], self._ncols), empty)
            # Log results from trace
            index.extend(trace.group_left.df.index)
            res.append(res_i)
        # Combine results for each left group into a single array that is  
        # compatible with the left collection
        if as_array:
            return np.array(index), np.concatenate(res)
        else:
            # Flatten data (relevant for geometry aggregators)
            return index, [i for sub in res for i in sub]

    def all(self, empty=None, **kwargs):
        """
        Return all values from intersecting events in a list.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            res = np.empty((trace.mask.shape[0], self._ncols), dtype=object)
            res[:] = [list(arr[mask_i, :].T) for mask_i in trace.mask]
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def any(self, empty=None, **kwargs):
        """
        Indicate whether each record intersects with at least one event.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Find any intersecting events, apply to all columns
            res = np.tile(trace.mask.any(axis=1), (self._ncols,1)).T
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def unique(self, empty=None, **kwargs):
        """
        Return all unique values from intersecting events in a tuple.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            res = np.empty((trace.mask.shape[0], self._ncols), dtype=object)
            res[:] = [[list(set(sub)) for sub in arr[mask_i, :].T] \
                for mask_i in trace.mask]
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def first(self, empty=None):
        """
        Return the first event value according to the order of the provided 
        collection's events dataframe.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Identify the first potential match
            index = np.argmax(trace.mask, axis=1)
            # Identify where matches are made
            valid = trace.mask.any(axis=1).reshape(-1,1)
            # Fill empty value where no match is made
            res = np.where(valid, arr[index], self._get_empty(empty))
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))
    
    def last(self, empty=None):
        """
        Return the last event value according to the order of the provided 
        collection's events dataframe.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Identify the first potential match
            index = np.argmax(np.flip(trace.mask, axis=1), axis=1)
            # Identify where matches are made
            valid = trace.mask.any(axis=1).reshape(-1,1)
            # Fill empty value where no match is made
            res = np.where(valid, arr[index], self._get_empty(empty))
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def count(self, empty=None):
        """
        Return the count of all intersected event values.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Sum count of all intersecting events, apply to all columns
            res = np.tile(trace.mask.sum(axis=1), (self._ncols,1)).T
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def sum(self, empty=None, nansum=False):
        """
        Return the sum of all intersected event values.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Choose all intersecting events
            if nansum:
                res = [np.nansum(arr[mask_i, :], axis=0).T for \
                    mask_i in trace.mask]
            else:
                res = [np.sum(arr[mask_i, :], axis=0).T for \
                    mask_i in trace.mask]
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def sumproduct(self, empty=None, normalized=False, dropna=False):
        """
        Return the sum of all event values multiplied by the weights of the 
        intersecting events. If normalized=False, the event values will be 
        multiplied by the actual overlapping length (e.g., multiplying a per-
        mile value by the miles of overlap). If normalized=True, the event 
        values will be multiplied by the normalized overlapping length (e.g., 
        multiplying a total value of an overlapped event by the proportion of 
        the event which is overlapped).

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        normalized : boolean, default False
            Whether the weights of the intersecting events being multiplied 
            with the event values should be normalized by the total length of 
            the events being intersected.
        dropna : boolean, default False
            Whether to drop np.nan values before aggregating.
        """
        def _func(arr, trace, **kwargs):
            # Prepare weights data congruent with array data
            weights = trace.weights
            weights = np.tile(np.expand_dims(weights, 2), (1,1,self._ncols))
            if normalized:
                weights /= trace.group_right.lengths
            # Drop nan if requested, zeroing weights where nan values occur
            if dropna:
                weights = np.where(np.isnan(arr.astype(float)), 0, weights)
           # Compute sums
            res = np.multiply(
                arr.reshape(1,-1,self._ncols), weights).sum(axis=1)
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def mean(self, empty=None, weighted=True, dropna=False):
        """
        Return an overlay length-weighted average of all event values. An 
        unweighted simple average can also be computed if weighted=True.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
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
            # Determine event weights
            weights = trace.weights if weighted else trace.mask
            weights = np.tile(np.expand_dims(weights, 2), (1,1,self._ncols))
            # Drop nan if requested, zeroing weights where nan values occur
            if dropna:
                weights = np.where(np.isnan(arr.astype(float)), 0, weights)
            # Compute means
            numer = np.multiply(
                arr.reshape(1,-1,self._ncols), weights).sum(axis=1)
            denom = weights.sum(axis=1)
            denom = np.where(denom==0, np.nan, denom)
            res = np.divide(numer, denom)
            return res
        return self._to_pandas(*self._agg(_func, empty=empty))

    def most(self, empty=None, dropna=True):
        """
        Return the event value associated with the greatest total overlay 
        length, ignoring missing values by default.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        dropna : boolean, default False
            Whether to drop np.nan values in intersecting events before 
            aggregating.
        """
        def _func(arr, trace, **kwargs):
            # Iterate over 2nd dimension
            res = []
            weights = trace.weights
            for arr_i in arr.T:
                # Drop nan if requested
                if dropna:
                    nanmask = ~pd.isna(arr_i)
                    if nanmask.sum() == 0:
                        res.append(
                            np.full(weights.shape[0], self._get_empty(empty)))
                        continue
                    else:
                        arr_i = arr_i[nanmask]
                        weights_i = weights[:,nanmask]
                else:
                    weights_i = weights
                # Aggregate and add to result
                res.append(get_most(arr_i, weights_i))
            return np.array(res).T
        return self._to_pandas(*self._agg(_func, empty=empty))
        
    def mode(self, empty=None):
        """
        Return the most frequent unique event value.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        def _func(arr, trace, **kwargs):
            # Iterate over all event masks
            res = []
            for mask_i in trace.mask:
                # Iterate over columns
                res_i = [get_mode(arr_i) for arr_i in arr[mask_i, :].T]
                res.append(res_i)
            return np.array(res)
        return self._to_pandas(*self._agg(_func, empty=empty))

    def value_counts(self, expand=True, dropna=True):
        """
        Return a dataframe of all unique intersecting event values and their 
        occurence counts.

        Parameters
        ----------
        expand : bool, default True
            Whether to automatically expand the value counts data to a 
            dataframe when a single column is being analyzed.
        """
        def _func(arr, trace, **kwargs):
            # Find null values in array
            if dropna:
                nanmask = ~pd.isna(arr.T)
            # Iterate over all event masks
            res = []
            for mask_i in trace.mask:
                # Iterate over columns
                res_i = []
                for j in range(arr.shape[1]):
                    # Remove NaN
                    if dropna:
                        arr_i = arr[mask_i & nanmask[j]]
                    else:
                        arr_i = arr[mask_i]
                    # Get counts of unique values
                    unique, counts = np.unique(
                        arr_i, return_counts=True, equal_nan=True)
                    # Summarize in dictionary
                    res_i.append(
                        {val: count for val, count in zip(unique, counts)})
                res.append(res_i)
            return np.array(res)
        # Generate data and expand if requested
        index, data = self._agg(_func, empty={})
        if self._ndim == 1:
            if expand:
                return pd.DataFrame(data.flatten().tolist(), index=index)
            else:
                return self._to_pandas(index, data)
        else:
            return self._to_pandas(index, data)
    
    def cut(self, empty=None, return_mls=True):
        """
        Cut intersecting event routes at the intersecting begin and end 
        locations, returning the resulting route's geometry or the route itself 
        if requested.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        return_mls : bool, default True
            Whether to return the MultiLineString associated with each cut 
            MLRRoute instead of the route itself.
        """
        if self._ncols != 1:
            raise ValueError("EventsMergeAttribute must represent a single "
                             "column to perform cut.")
        def _func(arr, trace, **kwargs):
            # Iterate over all event masks
            indices = np.argmax(trace.mask, axis=1)
            gen = zip(indices, trace.group_left.begs, trace.group_left.ends)
            res = []
            for i, beg, end in gen:
                # Choose the first intersecting event and cut the route
                route = arr[i, 0]
                try:
                    res_i = route.cut(beg, end)
                    res_i = res_i.mls if return_mls else res_i
                    res.append(res_i)
                except AttributeError:
                    raise TypeError(
                        "EventsMergeAttribute must represent a single column "
                        "and must contain MLSRoute objects to be cut.")
            return res
        return self._to_pandas(*self._agg(
            _func, empty=empty, as_array=False), as_series=True)
    
    def interpolate(self, snap=None, point='begs', empty=None, **kwargs):
        """
        Interpolate along intersecting event routes at the intersecting 
        location (or begin point for linear events), returning the resulting 
        interpolated point geometry.

        Parameters
        ----------
        snap : {None, 'near', 'left', 'right'}, default None
            If the event location does not fall within any geometry, snap to 
            the nearest match based on distance, choosing the closest location 
            to the left, right, or the nearest side ('near'). If None, a value 
            error will be raised when no intersecting ranges are found.
        point : {'begs', 'ends', 'centers'}, default 'begs'
            Where on the intersecting events the point should be made, at the 
            begin, end, or center point of the range.
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        if self._ncols != 1:
            raise ValueError("EventsMergeAttribute must represent a single "
                             "column to perform interpolate.")
        def _func(arr, trace, **kwargs):
            # Iterate over all event masks
            indices = np.argmax(trace.mask, axis=1)
            gen = zip(indices, getattr(trace.group_left, point))
            res = []
            for i, loc in gen:
                # Choose the first intersecting event and interpolate the point
                route = arr[i, 0]
                try:
                    res_i = route.interpolate(loc, snap=snap, **kwargs)
                    res.append(res_i)
                except AttributeError:
                    raise TypeError(
                        "EventsMergeAttribute must represent a single "
                        "column and must contain MLSRoute objects to be "
                        "interpolated.")
            return res
        return self._to_pandas(*self._agg(
            _func, empty=empty, as_array=False), as_series=True)
     

class EventsMergeTrace(object):
    """
    Object class for managing data on the relationship between two events 
    collections that have been merged using the EventsMerge system. Traces 
    contain a few main elements:

    group_left, group_right : pointers to the left and right events groups that 
        are related. During aggregation, information in the right group will be 
        aggregated and formed to the dataframe underlying the left group.
    key : the unique key associated with both events groups that produces their 
        relationship.
    mask : a boolean array of shape (group_left.df.shape[0], 
        group_right.df.shape[0]), i.e., a number of rows equal to the number of 
        rows in the left events group and a number of columns equal to the 
        number of rows in the right events group. This mask defines all 
        instances where the left and right groups intersect based on their 
        defined ranges and closed parameters.
    weights : a numeric array of shape (group_left.df.shape[0], 
        group_right.df.shape[0]), i.e., a number of rows equal to the number of 
        rows in the left events group and a number of columns equal to the 
        number of rows in the right events group. This array defines the actual 
        numeric length that is overlapped between the individual events in the 
        left and right events groups.
    success : a boolean indicator of whether or not a valid relationship has 
        been discovered between the left group and any right group. When False, 
        no right group will be indicated.
    """
    
    def __init__(self, group_left=None, group_right=None, key=None, mask=None, 
            weights=None, success=True):
        self.group_left = group_left
        self.group_right = group_right
        self.key = key
        self.mask = mask
        self.weights = weights
        self.success = success


class EventsMerge(object):
    """
    High-level object class for managing merges between two events collections 
    and summarizing/retrieving information from these merges. Generated through 
    collection-level merging methods such as EventsCollection.merge.
    """
    
    def __init__(self, left, right):
        # Log parameters
        self.left = left
        self.right = right
        
    def __getitem__(self, column) -> EventsMergeAttribute:
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
        try:
            return self._traces
        except AttributeError:
            self.build(inplace=True)
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
            raise ValueError(
                "Input EventsMerge target must have the same number of keys "
                f"as the existing left target ({self.num_keys}).")
            
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

    def build(self, inplace=True):
        """
        Perform intersects and overlays to produce EventsMergeTrace objects 
        for aggregation.
        """
        # Iterate over EventsGroups in the left collection
        traces = []
        for key, group_left in self.left.iter_groups():
            # Get the associated EventsGroup from the right collection
            try:
                group_right = self.right.get_group(key, empty=False)
            except KeyError:
                # No matching group on the right? Log empty trace
                traces.append(
                    EventsMergeTrace(
                        group_left=group_left, key=key, success=False))
                continue
            # Perform intersection
            mask = group_left.intersecting(
                other=group_right, squeeze=False, get_mask=True)
            weights = group_left.overlay(
                other=group_right, normalize=False, squeeze=False)
            # Aggregate results
            traces.append(
                EventsMergeTrace(
                    group_left=group_left, group_right=group_right, 
                    key=key, mask=mask, weights=weights, success=True))
        # Log traces
        self.traces = traces
    
    def distribute(self, column=None, **kwargs):
        """
        Intersect and distribute events over the range collection, scaling 
        their values relative to their indexed distance from their intersecting 
        range location.
        
        Parameters
        ----------
        column : pandas column label or list of same, optional
            The events dataframe column(s) containing the values associated 
            with each event being analyzed. If not provided, all values will 
            default to be 1.
        blur_size : int, default 0
            The number of pixels to blur events across based on the blur style.
        blur_style : str or callable, default 'linear'
            The scaling function to be called at each blurring step to scale 
            original values. If a callable is provided, it must accept a single 
            integer input for the zero-indexed pixel number, returning a single 
            float scaling value. Predefined blurring functions can be called 
            using the following labels:
            
            Options
            -------
            linear : linearly scale down values from the original value to zero 
                at the first index outside the blurred pixel range
            norm_static : Tk
            norm_scale : Tk
            none : do not scale down original values

        length_normalize : bool, default True
            Normalize the intersection scores by the length of the range to 
            account for differing range lengths.
                
        Created:  2022-10-04
        """
        # Validate column choice
        squeeze = True
        if not column is None:
            # Enforce list type
            if not isinstance(column, list):
                column = [column]
            else:
                squeeze = False
            # Ensure valid labels
            mismatched = set(column) - set(self.right.df.columns)
            if len(mismatched) > 0:
                raise ValueError(
                    f"Input column labels {mismatched} are not present in "
                    "the right dataframe.")

        # Iterate over EventsGroups in the left collection
        index = []
        res = []
        for key, group_left in self.left.iter_groups():
            # Get the associated EventsGroup from the right collection
            group_right = self.right.get_group(key, empty=True)
            # Get event values
            if not column is None:
                if isinstance(column, list):
                    values = group_right.df[column].values
                else:
                    values = group_right.df[[column]].values
            else:
                values = None
            # Perform distribution
            res_i = group_left.rng.distribute(
                group_right.rng, values=None, **kwargs)
            # Aggregate results over all column values
            if not column is None:
                res_i = (np.expand_dims(res_i, 2) * values).sum(axis=1)
            else:
                res_i = res_i.sum(axis=1)
            # Log results
            index.append(group_left.df.index)
            res.append(res_i)

        # Synthesize into pandas object
        index = np.concatenate(index)
        data = np.concatenate(res)
        # No column requested, prepare generic results
        if column is None:
            obj = pd.Series(
                index=index, data=data, name=None, dtype=float)
        # Column(s) requested, prepare complete results
        else:
            obj = pd.DataFrame(
                index=index, data=data, columns=column, dtype=float)
            # Squeeze dataframe if a single column label is provided
            if squeeze:
                obj = obj.squeeze()
        return obj

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

    def interpolate(self, **kwargs):
        """
        Interpolate along intersecting event routes at the intersecting 
        location (or begin point for linear events), returning the resulting 
        interpolated point geometry.

        Parameters
        ----------
        snap : {None, 'near', 'left', 'right'}, default None
            If the event location does not fall within any geometry, snap to 
            the nearest match based on distance, choosing the closest location 
            to the left, right, or the nearest side ('near'). If None, a value 
            error will be raised when no intersecting ranges are found.
        point : {'begs', 'ends', 'centers'}, default 'begs'
            Where on the intersecting events the point should be made, at the 
            begin, end, or center point of the range.
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there are no intersecting events and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        # Get attribute for routes if available
        try:
            ema = self[self.right.route]
        except:
            raise ValueError("Right collection does not contain a valid "
                "routes column label.")
        # Perform cut
        return ema.interpolate(**kwargs)

    def count(self, **kwargs):
        """
        Count the number of intersecting events.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        # Get the first key attribute
        ema = self[self.right.keys[0]]
        # Perform aggregation
        return ema.count(**kwargs)

    def any(self, **kwargs):
        """
        Indicate whether each record intersects with at least one event.

        Parameters
        ----------
        empty : scalar, string, or other pd.Series-compatible value, optional
            Value to use to fill when there is no matching events group and 
            aggregation cannot be performed. If None, values will be filled 
            with np.nan.
        """
        # Get the first key attribute
        ema = self[self.right.keys[0]]
        # Perform aggregation
        return ema.any(**kwargs)


###########
# HELPERS #
###########

def get_most(arr, weights):
    """
    Select the item from the input 1D array which is associated with the 
    highest total weight from each row in the 2D weights array. Scores are 
    computed by summing the weights for each unique array value for each row of 
    weights. When multiple values are tied, the first item in sorted order will 
    be selected.
    """
    # Group and split sorted target array
    sorter = np.argsort(arr)
    unique, splitter = np.unique(arr[sorter], return_index=True)
    splitter = splitter[1:]
    # Split weights and aggregate to select highest scoring values
    splits = np.split(weights[:, sorter], splitter, axis=1)
    select = unique[np.argmax([x.sum(axis=1) for x in splits], axis=0)]
    return select
    
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
