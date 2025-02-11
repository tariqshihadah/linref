from __future__ import annotations
import numpy as np
import pandas as pd
from linref.events import base
from scipy import sparse as sp


def _require_agg_data(func) -> callable:
    """
    Decorator for requiring aggregation data input.
    """
    def wrapper(*args, **kwargs):
        data = kwargs.pop('data', None)
        if data is None:
            raise ValueError(
                "Input aggregation data must be provided for the given "
                "aggregator.")
        return func(*args, data=data, **kwargs)
    return wrapper

def _validate_agg_axis_wrapper(func) -> callable:
    """
    Decorator for validating aggregation axis input.
    """
    def wrapper(*args, **kwargs):
        axis = kwargs.get('axis', 1)
        if axis not in {0, 1}:
            raise ValueError("Invalid axis provided. Must be 0 or 1.")
        return func(*args, **kwargs)
    return wrapper

def _validate_agg_2d_data_wrapper(func) -> callable:
    """
    Decorator for validating aggregation data input.
    """
    def wrapper(*args, **kwargs):
        axis = kwargs.get('axis', 1)
        data = kwargs.pop('data', None)
        # Check axis
        if axis not in {0, 1}:
            raise ValueError("Invalid axis provided. Must be 0 or 1.")
        # Check shape and dimensionality
        if data is None:
            data = np.ones((args[0].shape[axis], 1))
            kwargs['squeeze'] = True
        else:
            if isinstance(data, pd.DataFrame):
                data = data.values
            elif isinstance(data, pd.Series):
                data = data.values.reshape(-1, 1)
            elif not isinstance(data, np.ndarray):
                raise TypeError(
                    "Input aggregation data must be a numpy array or pandas "
                    "Series or DataFrame.")
            if data.ndim not in {1, 2}:
                raise ValueError(
                    "Input aggregation data must be a 1D or 2D numpy array.")
            elif data.ndim == 1:
                data = data.reshape(-1, 1)
                kwargs['squeeze'] = True
            if axis == 0 and data.shape[0] != args[0].left.num_events:
                raise ValueError(
                    "When axis=0, the input aggregation data's first dimension "
                    "must be equal to the number of events in the left "
                    "collection.")
            if axis == 1 and data.shape[0] != args[0].right.num_events:
                raise ValueError(
                    "When axis=1, the input aggregation data's first dimension "
                    "must be equal to the number of events in the right "
                    "collection.")
        return func(*args, data=data, **kwargs)
    return wrapper

def _validate_agg_1d_data_wrapper(func) -> callable:
    """
    Decorator for validating aggregation data input.
    """
    def wrapper(*args, **kwargs):
        axis = kwargs.get('axis', 1)
        data = kwargs.pop('data', None)
        # Check axis
        if axis not in {0, 1}:
            raise ValueError("Invalid axis provided. Must be 0 or 1.")
        # Check shape and dimensionality
        if data is None:
            data = np.ones((args[0].shape[axis],))
        else:
            if isinstance(data, pd.Series):
                data = data.values
            elif not isinstance(data, np.ndarray):
                raise TypeError(
                    "Input aggregation data must be a numpy array or pandas "
                    "Series.")
            if data.ndim != 1:
                raise ValueError(
                    "Input aggregation data must be a 1D numpy array.")
            if axis == 0 and data.shape[0] != args[0].left.num_events:
                raise ValueError(
                    "When axis=0, the input aggregation data's first dimension "
                    "must be equal to the number of events in the left "
                    "collection.")
            if axis == 1 and data.shape[0] != args[0].right.num_events:
                raise ValueError(
                    "When axis=1, the input aggregation data's first dimension "
                    "must be equal to the number of events in the right "
                    "collection.")
        return func(*args, data=data, **kwargs)
    return wrapper

def _squeeze_output_wrapper(func) -> callable:
    """
    Decorator for squeezing output arrays.
    """
    def wrapper(*args, **kwargs):
        squeeze = kwargs.get('squeeze', True)
        arr = func(*args, **kwargs)
        if squeeze:
            return np.squeeze(arr)
        return arr
    return wrapper


class EventsRelation(object):

    _valid_relate_agg_methods = {'overlay', 'intersect'}

    def __init__(self, left, right, cache=True) -> None:
        self.cache = cache
        self._left = left
        self._right = right
        self._validate_events()
        self.reset_cache()

    @property
    def left(self) -> base.EventsData:
        """
        The left events data object.
        """
        return self._left
    
    @left.setter
    def left(self, value) -> None:
        self._left = value
        self._validate_events()
        self.reset_cache()

    @property
    def right(self) -> base.EventsData:
        """
        The right events data object.
        """
        return self._right
    
    @right.setter
    def right(self, value) -> None:
        self._right = value
        self._validate_events()
        self.reset_cache()

    @property
    def cache(self) -> bool:
        return self._cache
    
    @cache.setter
    def cache(self, value) -> None:
        if not isinstance(value, bool):
            raise TypeError("The 'cache' parameter must be a boolean.")
        self._cache = value
        if not value:
            self.reset_cache()

    @property
    def shape(self) -> tuple:
        return self.left.num_events, self.right.num_events
    
    @property
    def overlay_data(self) -> sp.csr_matrix | None:
        return self._overlay_data
        
    @property
    def intersect_data(self) -> sp.csr_matrix | None:
        return self._intersect_data
    
    def _validate_events(self) -> None:
        """
        Validate the input events.
        """
        if not isinstance(self.left, base.EventsData) or not isinstance(self.right, base.EventsData):
            raise TypeError("Input objects must be EventsData class instances.")
        if self.left.is_grouped != self.right.is_grouped:
            raise ValueError("Input objects must have the same grouping status.")
        
    def _get_method_data(self, method) -> sp.csr_matrix | None:
        if method == 'overlay':
            return self._get_overlay_data()
        elif method == 'intersect':
            return self._get_intersect_data()
        else:
            raise ValueError(
                f"Invalid method provided ({method}). Must be one of "
                f"{self._valid_relate_agg_methods}.")
        
    def _get_intersect_data(self, **kwargs) -> sp.csr_matrix | None:
        if self.intersect_data is None:
            return self.intersect(**kwargs)
        return self.intersect_data
    
    def _get_overlay_data(self, **kwargs) -> sp.csr_matrix | None:
        if self.overlay_data is None:
            return self.overlay(**kwargs)
        return self.overlay_data

    def reset_cache(self) -> None:
        self._overlay_data = None
        self._overlay_kwargs = None
        self._intersect_data = None
        self._intersect_kwargs = None
        
    def overlay(self, normalize=False, norm_by='right', chunksize=1000, grouped=True) -> sp.csr_matrix:
        """
        Compute the overlay of the left and right events.
        
        Parameters
        ----------
        normalize : bool, default False
            Whether overlapping lengths should be normalized to give a 
            proportional result with a float value between 0 and 1.
        norm_by : str, default 'right'
            How overlapping lengths should be normalized. Only applied if
            `normalize` is True.
            - 'right' : Normalize by the length of the right events.
            - 'left' : Normalize by the length of the left events.
        chunksize : int or None, default 1000
            The maximum number of events to process in a single chunk.
            Input chunksize will affect the memory usage and performance of
            the function. This does not affect actual results, only 
            computation.
        grouped : bool, default True
            Whether to process the overlay operation for each group separately.
            This will affect the memory usage and performance of the function. 
            This does not affect actual results, only computation.

        Returns
        -------
        arr : scipy.sparse.csr_matrix
            The sparse matrix of overlay results. The shape of the matrix will
            be (m, n), where m is the number of events in the left dataframe
            and n is the number of events in the right dataframe.
        """

        # Perform overlay
        arr = overlay(
            self.left,
            self.right,
            normalize=normalize,
            norm_by=norm_by,
            chunksize=chunksize,
            grouped=grouped
        )
        # Cache results
        if self._cache:
            self._overlay_data = arr
            self._overlay_kwargs = {
                'normalize': normalize,
                'norm_by': norm_by,
                'chunksize': chunksize,
                'grouped': grouped
            }
        return arr
    
    def intersect(self, enforce_edges=True, chunksize=1000, grouped=True) -> sp.csr_matrix:
        """
        Compute the intersection of the left and right events.

        Parameters
        ----------
        enforce_edges : bool, default True
            Whether to enforce edge cases when computing intersections.
            If True, edge cases will be tested for intersections. Ignored 
            for point to point intersections.
        chunksize : int or None, default 1000
            The maximum number of events to process in a single chunk.
            Input chunksize will affect the memory usage and performance of
            the function. This does not affect actual results, only 
            computation.
        grouped : bool, default True
            Whether to process the overlay operation for each group separately.
            This will affect the memory usage and performance of the function. 
            This does not affect actual results, only computation.

        Returns
        -------
        arr : scipy.sparse.csr_matrix
            The sparse matrix of intersection results. The shape of the matrix 
            will be (m, n), where m is the number of events in the left dataframe
            and n is the number of events in the right dataframe.
        """
        # Perform intersect
        if self.left.is_point and self.right.is_point:
            arr = intersect_point_point(
                self.left,
                self.right,
                chunksize=chunksize,
                grouped=grouped
            )
        elif self.left.is_point and self.right.is_linear:
            arr = intersect_point_linear(
                self.left,
                self.right,
                enforce_edges=enforce_edges,
                chunksize=chunksize,
                grouped=grouped
            )
        elif self.left.is_linear and self.right.is_point:
            arr = intersect_point_linear(
                self.right,
                self.left,
                enforce_edges=enforce_edges,
                chunksize=chunksize,
                grouped=grouped
            )
        elif self.left.is_linear and self.right.is_linear:
            arr = intersect_linear_linear(
                self.left,
                self.right,
                enforce_edges=enforce_edges,
                chunksize=chunksize,
                grouped=grouped
            )
        else:
            raise ValueError("Invalid event types for intersection operation.")
        # Cache results
        if self._cache:
            self._intersect_data = arr
            self._intersect_kwargs = {
                'enforce_edges': enforce_edges,
                'chunksize': chunksize,
                'grouped': grouped
            }
        return arr
    
    # ----------------------------------------------------------------------- #
    # Aggregation methods
    # ----------------------------------------------------------------------- #

    @_validate_agg_axis_wrapper
    def count(self, axis=1, **kwargs) -> np.ndarray:
        """
        Count the number of intersections or overlays along the specified axis.

        Parameters
        ----------
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        **kwargs
            Additional keyword arguments to pass to the intersection or overlay
            methods if they have not been previously computed and cached.

        Returns
        -------
        arr : numpy.ndarray
            The aggregated data array. The shape of the array will be (m,), 
            where m is the number of events in the right dataframe if axis=0
            or the number of events in the left dataframe if axis=1.
        """
        # Check for cached data
        arr = self._get_overlay_data(**kwargs)

        # Perform aggregation
        return arr.sum(axis=axis)
    
    @_require_agg_data
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def list(self, data=None, axis=1, squeeze=True, **kwargs) -> np.ndarray:
        """
        Aggregate all input data values along the specified axis of the events
        relation against the other axis, returning a list of all values for
        intersecting events.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) or (n, x) where n is the number of events
            in the left dataframe if axis=0 or the number of events in the right
            dataframe if axis=1. This will result in an output shape of (m,) or 
            (m, x), respectively, where m is the number of events in the 
            opposite dataframe.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        squeeze : bool, default True
            Whether to squeeze the output array to a 1D array if possible.

        Returns
        -------
        arr : numpy.ndarray
            The aggregated data array. The shape of the array will be (m,) 
            or (m, x), where m is the number of events in the right dataframe
            if axis=0 or the number of events in the left dataframe if axis=1,
            and x is the number of columns in the input data array. If the 
            squeeze parameter is True, the output array will be squeezed to a
            1D array if possible.
        """
        # Check for cached data
        arr = self._get_intersect_data(**kwargs)
        arr = arr if axis == 1 else arr.T

        # Iterate over sparse rows
        output = []
        for row in arr:
            # Get data values
            values = data[row.indices]
            # Log values
            output.append(values.T.tolist())
        
        # Convert to numpy array of lists
        output_array = np.empty((len(output), data.shape[1]), dtype=object)
        output_array[:] = output
        return output_array
    
    def set(self, data=None, axis=1, squeeze=True, **kwargs) -> np.ndarray:
        """
        Aggregate all input data values along the specified axis of the events
        relation against the other axis, returning a set of all values for
        intersecting events.
        
        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) where n is the number of events in the left 
            dataframe if axis=0 or the number of events in the right dataframe if 
            axis=1. This will result in an output shape of (m,), where m is the 
            number of events in the opposite dataframe.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        squeeze : bool, default True
            Whether to squeeze the output array to a 1D array if possible.

        Returns
        -------
        arr : numpy.ndarray
            The aggregated data array. The shape of the array will be (m,) 
            or (m, x), where m is the number of events in the right dataframe
            if axis=0 or the number of events in the left dataframe if axis=1,
            and x is the number of columns in the input data array. If the 
            squeeze parameter is True, the output array will be squeezed to a
            1D array if possible.
        """
        # Pull list outputs and apply set operation
        output_array = self.list(data=data, axis=axis, squeeze=False, **kwargs)
        output_array = np.vectorize(set)(output_array)
        return output_array
    
    @_require_agg_data
    @_validate_agg_1d_data_wrapper
    def value_counts(self, data=None, axis=1, **kwargs) -> pd.DataFrame:
        """
        Aggregate all input data values along the specified axis of the events
        relation against the other axis, returning pandas DataFrame with
        a column for each unique value in the provided data array containing 
        counts of each value for intersecting events. The index of the DataFrame
        will be the index of the events in the opposite dataframe.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) where n is the number of events in the left 
            dataframe if axis=0 or the number of events in the right dataframe if 
            axis=1. This will result in an output shape of (m,), where m is the 
            number of events in the opposite dataframe.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame containing the value counts of the input data for 
            intersecting events. The index of the DataFrame will be the index 
            of the events in the opposite dataframe.
        """
        # Check for cached data
        arr = self._get_intersect_data(**kwargs)
        arr = arr if axis == 1 else arr.T

        # Iterate over sparse rows
        output = []
        for row in arr:
            # Get data values
            values = data[row.indices]
            # Log values
            output.append(dict(zip(*np.unique(values, return_counts=True))))
        
        # Convert to numpy array of lists
        index = self.left.index if axis == 1 else self.right.index
        output = pd.DataFrame(output, index=index).fillna(0)
        return output
    
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def sum(self, data=None, method='overlay', axis=1, squeeze=True, **kwargs) -> np.ndarray:
        """
        Sum the input data along the specified axis of the events relationship,
        multiplying by the overlay or intersection data and summing the result.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) or (n, x) where n is the number of events
            in the left dataframe if axis=0 or the number of events in the right
            dataframe if axis=1. This will result in an output shape of (m,) or 
            (m, x), respectively, where m is the number of events in the 
            opposite dataset. If None, the events relationship data will be
            summed directly.
        method : {'intersect', 'overlay'}, default 'overlay'
            The method to use for the events relationship data being 
            multiplied.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        squeeze : bool, default True
            Whether to squeeze the output array to a 1D array if possible.
        **kwargs
            Additional keyword arguments to pass to the intersection or overlay
            methods if they have not been previously computed and cached.

        Returns
        -------
        arr : numpy.ndarray
            The aggregated data array. The shape of the array will be (m,) 
            or (m, x), where m is the number of events in the right dataframe
            if axis=0 or the number of events in the left dataframe if axis=1,
            and x is the number of columns in the input data array. If the 
            squeeze parameter is True, the output array will be squeezed to a
            1D array if possible.
        """
        # Check for cached data
        arr = self._get_method_data(method)
        
        # Perform aggregation
        aggregated = []
        for column in data.T:
            # Reshape column for broadcasting
            column = column.reshape(-1, 1) if axis == 0 else column.reshape(1, -1)
            aggregated.append(arr.multiply(column).sum(axis=axis))
        
        # Concatenate results
        return np.vstack(aggregated).T
    
    @_require_agg_data
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def mean(self, data=None, method='overlay', axis=1, squeeze=True, **kwargs) -> np.ndarray:
        """
        Compute the mean of the input data along the specified axis of the events 
        relationship, multiplying by the overlay or intersection data and summing 
        the result.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) or (n, x) where n is the number of events
            in the left dataframe if axis=0 or the number of events in the right
            dataframe if axis=1. This will result in an output shape of (m,) or 
            (m, x), respectively, where m is the number of events in the 
            opposite dataset.
        method : {'intersect', 'overlay'}, default 'overlay'
            The method to use for the events relationship data being 
            aggregated. If 'overlay', the overlay data will be used, producing
            a length-weighted mean. If 'intersect', the intersection data will
            be used, producing a count-weighted mean.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        squeeze : bool, default True
            Whether to squeeze the output array to a 1D array if possible.
        **kwargs
            Additional keyword arguments to pass to the intersection or overlay
            methods if they have not been previously computed and cached.

        Returns
        -------
        arr : numpy.ndarray
            The aggregated data array. The shape of the array will be (m,) 
            or (m, x), where m is the number of events in the right dataframe
            if axis=0 or the number of events in the left dataframe if axis=1,
            and x is the number of columns in the input data array. If the 
            squeeze parameter is True, the output array will be squeezed to a
            1D array if possible.
        """
        # Check for cached data
        arr = self._get_method_data(method)
        
        # Perform aggregation
        aggregated = []
        for column in data.T:
            # Reshape column for broadcasting
            column = column.reshape(-1, 1) if axis == 0 else column.reshape(1, -1)
            numerator = arr.multiply(column).sum(axis=axis)
            denominator = arr.sum(axis=axis)
            aggregated.append(np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator!=0
            ))
        
        # Concatenate results
        return np.vstack(aggregated).T

    @_require_agg_data
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def mode(self, data=None, method='overlay', axis=1, squeeze=True, **kwargs) -> np.ndarray:
        """
        Compute the mode of the input data along the specified axis of the events 
        relationship, multiplying by the overlay or intersection data and summing 
        the result.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) or (n, x) where n is the number of events
            in the left dataframe if axis=0 or the number of events in the right
            dataframe if axis=1. This will result in an output shape of (m,) or 
            (m, x), respectively, where m is the number of events in the 
            opposite dataset.
        method : {'intersect', 'overlay'}, default 'overlay'
            The method to use for the events relationship data being 
            aggregated. If 'overlay', the overlay data will be used, producing
            a length-weighted mode. If 'intersect', the intersection data will
            be used, producing a count-weighted mode.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        squeeze : bool, default True
            Whether to squeeze the output array to a 1D array if possible.
        **kwargs
            Additional keyword arguments to pass to the intersection or overlay
            methods if they have not been previously computed and cached.

        Returns
        -------
        arr : numpy.ndarray
            The aggregated data array. The shape of the array will be (m,) 
            or (m, x), where m is the number of events in the right dataframe
            if axis=0 or the number of events in the left dataframe if axis=1,
            and x is the number of columns in the input data array. If the 
            squeeze parameter is True, the output array will be squeezed to a
            1D array if possible.
        """
        # Check for cached data
        arr = self._get_method_data(method)
        
        # Perform aggregation
        aggregated = []
        for column in data.T:
            # Sort unique values and identify splits in the weights data
            sorter = np.argsort(column)
            unique, splitter = np.unique(column[sorter], return_index=True)
            splitter = np.append(splitter, len(column))
            # Sort weights data before splitting
            arr_sorted = arr.T[sorter] if axis == 1 else arr[sorter]
            # Compute weighted scores for each unique value
            scores = []
            for i, j in zip(splitter[:-1], splitter[1:]):
                scores.append(arr_sorted[i:j].sum(axis=0))
            # Find mode using the highest score for each record
            mode = unique[np.argmax(np.vstack(scores), axis=0)]
            aggregated.append(mode)

        # Concatenate results
        return np.vstack(aggregated).T

def _grouped_operation_wrapper(func) -> callable:
    """
    Decorator for wrapping functions that operate on grouped data.
    """
    def wrapper(left, right, *args, **kwargs):
        # Validate inputs
        if not isinstance(left, base.EventsData) or not isinstance(right, base.EventsData):
            raise TypeError("Input objects must be EventsData class instances.")
        if left.is_grouped != right.is_grouped:
            raise ValueError("Input objects must have the same grouping status.")

        # Check if grouped
        grouped = kwargs.pop('grouped', True)
        if not left.is_grouped or not grouped:
            # Return group-less operation
            return func(left, right, *args, **kwargs)
        else:
            # Iterate over groups
            left_index = []
            right_index = []
            res = []
            for group, left_group in left.reset_index().iter_groups(ungroup=True):
                # Get right group
                right_group = right.reset_index().select_group(
                    group, ungroup=True, ignore_missing=True, inplace=False)
                if right_group.num_events == 0:
                    continue
                # Compute operation on group
                group_res = func(left_group, right_group, *args, **kwargs)
                # Convert to sparse array and log
                res.append(group_res)
                left_index.append(left_group.index)
                right_index.append(right_group.index)
            # Concatenate results
            res = sp.block_diag(res, format='coo')
            left_index = np.concatenate(left_index, axis=0)
            right_index = np.concatenate(right_index, axis=0)
            # Reorder to match original indices
            left_select = np.argsort(left_index)
            right_select = np.argsort(right_index)
            res = res.tocsr()[left_select, :][:, right_select]
            return res
    return wrapper

def _chunked_operation_wrapper(func) -> callable:
    """
    Decorator for wrapping functions that operate on chunks of data.
    """
    def wrapper(left, right, *args, **kwargs):
        # Validate chunksize
        chunksize = kwargs.pop('chunksize', None)
        if chunksize is not None:
            if not isinstance(chunksize, int):
                raise TypeError("The 'chunksize' parameter must be an integer.")
            if chunksize < 1:
                raise ValueError("The 'chunksize' parameter must be greater than 0.")
        else:
            # If no chunksize provided, skip chunking altogether
            chunk = func(left, right, *args, **kwargs)
            res = sp.coo_array(chunk)
            return res
        
        # Iterate over chunks in both dimensions
        left_arrays = []
        for i in range(0, left.num_events, chunksize):
            right_arrays = []
            for j in range(0, right.num_events, chunksize):
                # Get chunk of events
                left_chunk = left.select_slice(slice(i, i + chunksize), inplace=False)
                right_chunk = right.select_slice(slice(j, j + chunksize), inplace=False)
                # Compute operation on chunk
                chunk = func(left_chunk, right_chunk, *args, **kwargs)
                # Convert to sparse array and log
                right_arrays.append(sp.coo_array(chunk))
            # Concatenate right arrays and log
            left_arrays.append(sp.hstack(right_arrays))
        # Concatenate left arrays and return
        res = sp.vstack(left_arrays)
        return res
    return wrapper

@_grouped_operation_wrapper
@_chunked_operation_wrapper
def overlay(left, right, normalize=True, norm_by='right', chunksize=None) -> sp.csr_matrix:
    """
    Compute the overlay of two collections of events.

    Parameters
    ----------
    left, right : EventsData
        Input EventsData instances to overlay.
    normalize : bool, default True
        Whether overlapping lengths should be normalized to give a 
        proportional result with a float value between 0 and 1.
    norm_by : str, default 'right'
        How overlapping lengths should be normalized. Only applied if
        `normalize` is True.
        - 'right' : Normalize by the length of the right events.
        - 'left' : Normalize by the length of the left events.
    chunksize : int or None, default None
        The maximum number of elements to process in a single chunk.
        Input chunksize will affect the memory usage and performance of
        the function.
    """
    _norm_by_options = {'right', 'left'}
    
    # Validate inputs
    if not isinstance(left, base.EventsData) or not isinstance(right, base.EventsData):
        raise TypeError("Input objects must be EventsData class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input collections must have the same grouping status.")
    if not left.is_linear or not right.is_linear:
        raise ValueError("Input events must be linear.")
    if not left.is_monotonic or not right.is_monotonic:
        raise ValueError("Input events must be monotonic.")

    # Compute overlap lengths
    lefts = left.ends.reshape(-1, 1) - right.begs.reshape(1, -1)
    rights = right.ends.reshape(1, -1) - left.begs.reshape(-1, 1)

    # Compare against event lengths
    overlap = np.minimum(lefts, rights)
    lengths = np.minimum(
        left.lengths.reshape(-1, 1),
        right.lengths.reshape(1, -1)
    )
    np.minimum(overlap, lengths, out=overlap)
    np.clip(overlap, 0, None, out=overlap)

    # Normalize if necessary
    if normalize:
        # Get denominator
        if norm_by == 'right':
            denom = right.lengths.reshape(1, -1)
        elif norm_by == 'left':
            denom = left.lengths.reshape(-1, 1)
        else:
            raise ValueError(
                f"Invalid 'norm_by' parameter value provided ({norm_by}). Must be one "
                f"of {_norm_by_options}.")
        # Normalize
        denom = np.where(denom==0, np.inf, denom)
        np.divide(overlap, denom, out=overlap)

    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        mask = left.groups.reshape(-1, 1) == right.groups.reshape(1, -1)
        np.multiply(overlap, mask, out=overlap)
    
    return overlap

@_grouped_operation_wrapper
@_chunked_operation_wrapper
def intersect_point_point(left, right, chunksize=None) -> sp.csr_matrix:
    """
    Identify intersections between two collections of point events.
    """
    # Validate inputs
    if not isinstance(left, base.EventsData) or not isinstance(right, base.EventsData):
        raise TypeError("Input objects must be EventsData class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input collections must have the same grouping status.")

    # Reshape arrays for broadcasting
    left_locs = left.locs.reshape(-1, 1)
    right_locs = right.locs.reshape(1, -1)
    
    # Test for intersection of locations
    res = np.equal(left_locs, right_locs)

    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        mask = left.groups.reshape(-1, 1) == right.groups.reshape(1, -1)
        np.logical_and(res, mask, out=res)
    
    return res

@_grouped_operation_wrapper
@_chunked_operation_wrapper
def intersect_point_linear(left, right, enforce_edges=True, chunksize=None) -> sp.csr_matrix:
    """
    Identify intersections between a collection of point events and a collection 
    of linear events.
    """
    # Validate inputs
    if not isinstance(left, base.EventsData) or not isinstance(right, base.EventsData):
        raise TypeError("Input objects must be EventsData class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input objects must have the same grouping status.")

    # Reshape arrays for broadcasting
    left_locs = left.locs.reshape(-1, 1)
    right_begs = right.begs.reshape(1, -1)
    right_ends = right.ends.reshape(1, -1)

    # Initialize result array
    res = np.zeros((left_locs.shape[0], right_begs.shape[1]), dtype=bool)

    # Test for intersection of locations
    right_closed_base = right.closed_base
    # - Test 1
    if right_closed_base in ['left', 'both']:
        np.greater_equal(left_locs, right_begs, out=res)
    else:
        np.greater(left_locs, right_begs, out=res)
    # - Test 2
    if right_closed_base in ['right', 'both']:
        np.less_equal(left_locs, right_ends, out=res, where=res)
    else:
        np.less(left_locs, right_ends, out=res, where=res)

    # Test for modified edges
    if right.closed_mod and enforce_edges:
        # Get mask of modified edges to overwrite unmodified edges
        mask = right.modified_edges.reshape(1, -1)
        if right_closed_base == 'left':
            np.equal(left_locs, right_ends, out=res, where=mask & ~res)
        elif right_closed_base == 'right':
            np.equal(left_locs, right_begs, out=res, where=mask & ~res)
    
    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        mask = left.groups.reshape(-1, 1) == right.groups.reshape(1, -1)
        np.logical_and(res, mask, out=res)
    
    return res

@_grouped_operation_wrapper
@_chunked_operation_wrapper
def intersect_linear_linear(left, right, enforce_edges=True, chunksize=None) -> sp.csr_matrix:
    """
    Identify intersections between two collections of linear events.
    """
    # Validate inputs
    if not isinstance(left, base.EventsData) or not isinstance(right, base.EventsData):
        raise TypeError("Input objects must be EventsData class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input objects must have the same grouping status.")

    # Reshape arrays for broadcasting
    left_begs = left.begs.reshape(-1, 1)
    left_ends = left.ends.reshape(-1, 1)
    right_begs = right.begs.reshape(1, -1)
    right_ends = right.ends.reshape(1, -1)

    # Initialize result array
    res = np.zeros((left_begs.shape[0], right_begs.shape[1]), dtype=bool)
    step = np.zeros((left_begs.shape[0], right_begs.shape[1]), dtype=bool)

    # Initialize result array with linear intersections
    np.greater(left_ends, right_begs, out=res)
    np.less(left_begs, right_ends, out=step)
    res &= step

    # Test edges if necessary
    if enforce_edges:
        # Identify if edge cases need testing
        test_edges = not (
            ((left.closed == 'neither') or (right.closed == 'neither')) or \
            ((left.closed == 'left') and (right.closed == 'left')) or \
            ((left.closed == 'right') and (right.closed == 'right'))
        )
        if test_edges:
            # Identify which edge cases need testing
            test_begs_ends = (left.closed != 'right') and (right.closed != 'left')
            test_ends_begs = (left.closed != 'left') and (right.closed != 'right')

            # Identify modified edges if needed
            if left.closed_mod:
                left_mod = left.modified_edges.reshape(-1, 1)
            if right.closed_mod:
                right_mod = right.modified_edges.reshape(1, -1)

            # - Test 1: left_begs == right_ends
            mask = np.invert(res)
            if test_begs_ends:
                # Create mask for where edge cases are relevant
                if left.closed == 'right_mod':
                    np.logical_and(mask, left_mod, out=mask)
                if right.closed == 'left_mod':
                    np.logical_and(mask, right_mod, out=mask)
                # Apply test
                np.equal(left_begs, right_ends, out=step)
                np.logical_and(step, mask, out=step)
                np.logical_or(res, step, out=res)

            # - Test 2: left_ends == right_begs
            np.invert(res, out=mask)
            if test_ends_begs:
                # Create mask for where edge cases are relevant
                if left.closed == 'left_mod':
                    np.logical_and(mask, left_mod, out=mask)
                if right.closed == 'right_mod':
                    np.logical_and(mask, right_mod, out=mask)
                # Apply test
                np.equal(left_ends, right_begs, out=step)
                np.logical_and(step, mask, out=step)
                np.logical_or(res, step, out=res)

    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        mask = left.groups.reshape(-1, 1) == right.groups.reshape(1, -1)
        np.logical_and(res, mask, out=res)
    
    return res
