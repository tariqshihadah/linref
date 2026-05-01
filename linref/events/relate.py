from __future__ import annotations
import numpy as np
import pandas as pd
from linref.events import base, geometry
from linref.events.profile import resolve_profile
from linref.utility import utility
from linref.errors import LRSConfigurationError, LRSCompatibilityError
from scipy import sparse as sp
from scipy.stats import norm
import copy


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

def _get_selector_data_wrapper(func) -> callable:
    """
    Decorator for getting selector data.
    """
    def wrapper(*args, **kwargs):
        axis = kwargs.get('axis', 1)
        data = kwargs.pop('data', None)
        self = args[0]
        # Attempt to get data if selector is set
        if data is None:
            if self._selector is not None:
                data = self._get_selected_data(axis)
        return func(*args, data=data, **kwargs)
    return wrapper

def _get_linestring_m_data_wrapper(func) -> callable:
    """
    Decorator for getting linestring M data.
    """
    def wrapper(*args, **kwargs):
        axis = kwargs.get('axis', 1)
        data = kwargs.pop('data', None)
        self = args[0]
        # Attempt to get data if selector is set
        if data is None:
            if self._selector is None:
                try:
                    data = \
                        self.left_df .lr.geoms_m if axis == 0 else \
                        self.right_df.lr.geoms_m
                    assert data is not None
                except (AttributeError, AssertionError):
                    raise ValueError(
                        "No linestring M data found in the selected dataframe. "
                        f"Ensure that the {'left' if axis == 0 else 'right'} "
                        "dataframe has linestring M geometry data and that it "
                        "is set within the dataframe's LRS."
                    )
            else:
                data = self._get_selected_data(axis)
        else:
            data = np.asarray(data)
        # Validate data type
        if isinstance(data[0], geometry.LineString):
            raise TypeError(
                "Input aggregation data does not contain M values. Use "
                "the `add_geom_m` method to add M values to LineString "
                "geometries."
            )
        elif not isinstance(data[0], geometry.LineStringM):
            raise TypeError(
                "Input aggregation data must be a list or array of "
                "LineStringM geometry objects."
            )
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
            else:
                try:
                    data = np.asarray(data)
                except AttributeError:
                    pass
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    "Input aggregation data must be a numpy array or pandas "
                    f"Series. Given type: {type(data)}.")
            if data.ndim != 1:
                raise ValueError(
                    "Input aggregation data must be a 1D numpy array. Given "
                    f"array with {data.ndim} dimensions.")
            if axis == 0 and data.shape[0] != args[0].left.num_events:
                raise ValueError(
                    "When axis=0, the input aggregation data's first dimension "
                    "must be equal to the number of events in the left "
                    "collection. Given shape: {data.shape}.")
            if axis == 1 and data.shape[0] != args[0].right.num_events:
                raise ValueError(
                    "When axis=1, the input aggregation data's first dimension "
                    "must be equal to the number of events in the right "
                    "collection. Given shape: {data.shape}.")
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

    _valid_relate_agg_methods = {'equal_groups', 'overlay', 'intersect'}

    def __init__(
            self,
            left: base.EventsData,
            right: base.EventsData,
            left_df: pd.DataFrame | None = None,
            right_df: pd.DataFrame | None = None,
            cache: bool = True
        ) -> None:
        # Log input data
        self.cache = cache
        self._left = left # Set directly for order of operations validation
        self._right = right # Set directly for order of operations validation
        self._left_df = left_df
        self._right_df = right_df
        # Initialize and validate
        self._validate_events()
        self.reset_cache()
        self._set_selector(None, inplace=True)

    def __getitem__(self, key) -> EventsRelation:
        """
        Return a new EventsRelation object with the specified selector.
        """
        relate = self.copy()
        relate._set_selector(key, inplace=True)
        return relate

    @property
    def left(self) -> base.EventsData:
        """
        The left events data object.
        """
        return self._left
    
    @left.setter
    def left(self, obj) -> None:
        self._left = obj
        self._validate_events()
        self.reset_cache()

    @property
    def right(self) -> base.EventsData:
        """
        The right events data object.
        """
        return self._right
    
    @right.setter
    def right(self, obj) -> None:
        self._right = obj
        self._validate_events()
        self.reset_cache()

    @property
    def left_df(self) -> pd.DataFrame | None:
        """
        The dataframe associated with the left events data.
        """
        return self._left_df
    
    @left_df.setter
    def left_df(self, obj) -> None:
        self._left_df = obj
        self._validate_events()

    @property
    def right_df(self) -> pd.DataFrame | None:
        """
        The dataframe associated with the right events data.
        """
        return self._right_df
    
    @right_df.setter
    def right_df(self, obj) -> None:
        self._right_df = obj
        self._validate_events()

    @property
    def cache(self) -> bool:
        """Whether computed relationship data is cached for reuse."""
        return self._cache
    
    @cache.setter
    def cache(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("The 'cache' parameter must be a boolean.")
        self._cache = value
        if not value:
            self.reset_cache()

    @property
    def shape(self) -> tuple:
        """Shape of the relationship matrix as (left_events, right_events)."""
        return self.left.num_events, self.right.num_events
    
    @property
    def equal_groups_data(self) -> sp.csr_matrix | None:
        """Cached equal-groups boolean matrix, or None if not yet computed."""
        return self._equal_groups_data
    
    @property
    def overlay_data(self) -> sp.csr_matrix | None:
        """Cached overlay matrix, or None if not yet computed."""
        return self._overlay_data
        
    @property
    def intersect_data(self) -> sp.csr_matrix | None:
        """Cached intersection matrix, or None if not yet computed."""
        return self._intersect_data
    
    @property
    def T(self) -> EventsRelation:
        """
        Return a new EventsRelation object with the left and right events,
        dataframes, and cached data transposed.
        """
        obj = self.copy()
        # Swap left and right events
        obj._left, obj._right = obj._right, obj._left
        # Swap left and right dataframes
        obj._left_df, obj._right_df = obj._right_df, obj._left_df
        # Reset cache and selector
        obj.reset_cache()
        obj.reset_selector()
        return obj

    def cache_memory_usage(self, unit: str = 'MB') -> float:
        """
        Get the memory usage of the cached data.

        Parameters
        ----------
        unit : {'bytes', 'KB', 'MB', 'GB'}, default 'MB'
            The unit to return the memory usage in. Must be one of:
            - 'bytes' : Bytes
            - 'KB' : Kilobytes
            - 'MB' : Megabytes
            - 'GB' : Gigabytes
        """
        # If no data cached, return 0
        if not self._cache:
            return 0.0
        # Calculate memory usage
        bytes = 0
        if self._equal_groups_data is not None:
            bytes += \
                self._equal_groups_data.data    .nbytes + \
                self._equal_groups_data.indptr  .nbytes + \
                self._equal_groups_data.indices .nbytes
        if self._overlay_data is not None:
            bytes += \
                self._overlay_data.data    .nbytes + \
                self._overlay_data.indptr  .nbytes + \
                self._overlay_data.indices .nbytes
        if self._intersect_data is not None:
            bytes += \
                self._intersect_data.data    .nbytes + \
                self._intersect_data.indptr  .nbytes + \
                self._intersect_data.indices .nbytes
        # Convert to requested unit
        if unit == 'bytes':
            return bytes
        elif unit == 'KB':
            return bytes / 1024
        elif unit == 'MB':
            return bytes / 1024**2
        elif unit == 'GB':
            return bytes / 1024**3
        else:
            raise ValueError(f"Invalid unit '{unit}'. Must be one of 'bytes', 'KB', 'MB', or 'GB'.")

    def _validate_events(self) -> None:
        """
        Validate the input events data.
        """
        # Validate events
        if not isinstance(self.left, base.EventsData) or not isinstance(self.right, base.EventsData):
            raise TypeError("Input objects must be EventsData class instances.")
        if self.left.is_grouped != self.right.is_grouped:
            raise ValueError("Input objects must have the same grouping status.")
        # Validate dataframes
        if self._left_df is not None:
            if not isinstance(self._left_df, pd.DataFrame):
                raise TypeError("The 'left_df' parameter must be a pandas DataFrame.")
            if self.left.num_events != self._left_df.shape[0]:
                raise ValueError(
                    f"The number of events in the left dataframe ({self._left_df.shape[0]}) must match the "
                    f"number of events in the left events data object ({self.left.num_events}).")
        if self._right_df is not None:
            if not isinstance(self._right_df, pd.DataFrame):
                raise TypeError("The 'right_df' parameter must be a pandas DataFrame.")
            if self.right.num_events != self._right_df.shape[0]:
                raise ValueError(
                    f"The number of events in the right dataframe ({self._right_df.shape[0]}) must match the "
                    f"number of events in the right events data object ({self.right.num_events}).")

    def _get_method_data(self, method, axis, **kwargs) -> sp.csr_matrix | None:
        # Set default method if none provided
        if method is None:
            method = 'overlay' if (self.left.is_linear and self.right.is_linear) else 'intersect'
        # Validate method
        if method == 'equal_groups':
            return self._get_equal_groups_data(**kwargs)
        elif method == 'overlay':
            if self.left.is_point or self.right.is_point:
                raise LRSConfigurationError(
                    "Overlay method cannot be used with point events. "
                    "Use 'intersect' method instead.")
            return self._get_overlay_data(axis=axis, **kwargs)
        elif method == 'intersect':
            return self._get_intersect_data(**kwargs)
        else:
            raise ValueError(
                f"Invalid method provided ({method}). Must be one of "
                f"{self._valid_relate_agg_methods}.")
        
    def _get_equal_groups_data(self, **kwargs) -> sp.csr_matrix | None:
        if self.equal_groups_data is not None:
            if all(
                self._equal_groups_kwargs.get(key, None) == value
                for key, value in kwargs.items()
            ):
                return self.equal_groups_data
        return self.equal_groups(**kwargs)
        
    def _get_intersect_data(self, **kwargs) -> sp.csr_matrix | None:
        if self.intersect_data is not None:
            if all(
                self._intersect_kwargs.get(key, None) == value
                for key, value in kwargs.items()
            ):
                return self.intersect_data
        return self.intersect(**kwargs)
    
    def _get_overlay_data(self, axis=1, **kwargs) -> sp.csr_matrix | None:
        # Adjust kwargs for axis
        if not 'norm_by' in kwargs:
            kwargs['norm_by'] = 'right' if axis == 1 else 'left'
        if 'profile' not in kwargs:
            kwargs['profile'] = None
        if self.overlay_data is not None:
            if all(
                self._overlay_kwargs.get(key, None) == value
                for key, value in kwargs.items()
            ):
                return self.overlay_data
        return self.overlay(**kwargs)

    def copy(self, deep: bool = False) -> EventsRelation:
        """
        Create an exact copy of the object instance.
        
        Parameters
        ----------
        deep : bool, default False
            Whether the created copy should be a deep copy.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)
    
    def reset_cache(self) -> None:
        """
        Reset all cached data from computations on the relationships between 
        the left and right events, including equal groups, intersections,
        and overlays.
        """
        self._equal_groups_data = None
        self._equal_groups_kwargs = None
        self._overlay_data = None
        self._overlay_kwargs = None
        self._intersect_data = None
        self._intersect_kwargs = None

    def reset_selector(self) -> None:
        """
        Reset the selector used to select data from either the left or right
        dataframe during aggregation operations.
        """
        return self._set_selector(None, inplace=True)

    def _set_selector(self, selector, inplace=False) -> None:
        """
        Set the column label or slice to use to select data from either the 
        left or right dataframe when performing aggregation operations.  
        Whichdataframe the selector applies to will depend on the axis of 
        theaggregation operation. During aggregation, if axis=0, the selector 
        will be applied to the left dataframe, and if axis=1, the selector 
        will be applied to the right dataframe.

        Parameters
        ----------
        selector : str, list, slice, or None
            The column label, list of column labels, or slice to use to select
            data from the left or right dataframe during aggregation operations.
        inplace : bool, default False
            Whether to perform the operation inplace or return a copy of the 
            object with the selector set.
        """
        if isinstance(selector, (str, list, slice, type(None))):
            obj = self if inplace else self.copy()
            obj._selector = selector
            return obj if not inplace else None
        else:
            raise TypeError(
                "The 'selector' parameter must be a string, list, slice, or "
                "None.")
        
    def _get_selected_data(self, axis) -> np.ndarray:
        """
        Get the selected data from the left or right dataframe based on the
        current selector and axis.
        """
        # Determine index and dataframe to pull from
        if axis == 0:
            df = self.left_df
        elif axis == 1:
            df = self.right_df
        else:
            raise ValueError("Invalid axis provided. Must be 0 or 1.")
        # Confirm that dataframes are set
        if df is None:
            raise ValueError(
                "No dataframe set on the events relationship.")
        # Select data
        if self._selector is None:
            raise ValueError("No selector set. Set a selector with indexing.")
        if isinstance(self._selector, str):
            try:
                return df.loc[:, self._selector].values
            except KeyError:
                raise KeyError(
                    f"Invalid selector provided. Column label '{self._selector}' "
                    f"not found in the {'left' if axis==0 else 'right'} dataframe.")
        elif isinstance(self._selector, list):
            try:
                return df.loc[:, self._selector].values
            except KeyError:
                raise KeyError(
                    f"Invalid selector provided. Column labels '{self._selector}' "
                    f"not found in the {'left' if axis==0 else 'right'} dataframe.")
        elif isinstance(self._selector, slice):
            try:
                return df.loc[:, self._selector].values
            except KeyError:
                raise KeyError(
                    f"Invalid selector provided. Column slice '{self._selector}' "
                    f"not found in the {'left' if axis==0 else 'right'} dataframe.")
        else:
            raise ValueError(
                "Invalid selector provided. Must be a string, list, or slice.")
        
    def equal_groups(self, chunksize: int = 1000, grouped: bool = True) -> sp.csr_matrix:
        """
        Compute a boolean matrix indicating whether left and right events
        belong to the same group.

        Parameters
        ----------
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
            The sparse boolean matrix of group equality results. The shape of 
            the matrix will be (m, n), where m is the number of events in the 
            left dataframe and n is the number of events in the right dataframe.
        """
        # Perform equal groups check
        arr = equal_groups(
            self.left,
            self.right,
            chunksize=chunksize,
            grouped=grouped
        )
        # Cache results
        if self._cache:
            self._equal_groups_data = arr
            self._equal_groups_kwargs = {
                'chunksize': chunksize,
                'grouped': grouped
            }
        return arr

    def overlay(self, normalize: bool = True, norm_by: str = 'right', profile=None, chunksize: int = 1000, grouped: bool = True) -> sp.csr_matrix:
        """
        Compute the overlay of the left and right events.
        
        Parameters
        ----------
        normalize : bool, default True
            Whether overlapping lengths should be normalized to give a 
            proportional result with a float value between 0 and 1.
        norm_by : str, default 'right'
            How overlapping lengths should be normalized. Only applied if
            `normalize` is True.
            - 'right' : Normalize by the length of the right events.
            - 'left' : Normalize by the length of the left events.
        profile : str, EventProfile, or None, default None
            An event profile defining how value is distributed along the
            event's length. When provided, overlay weights reflect the
            proportion of the event's profiled value that is overlapped
            rather than simple length proportion.
            - None : No profiling (default, current behavior).
            - str : Name of a built-in profile ('uniform', 'triangular',
              'parabolic', 'trapezoidal').
            - EventProfile : A profile instance for custom parameterization.
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
            profile=profile,
            chunksize=chunksize,
            grouped=grouped
        )
        # Cache results
        if self._cache:
            self._overlay_data = arr
            self._overlay_kwargs = {
                'normalize': normalize,
                'norm_by': norm_by,
                'profile': profile,
                'chunksize': chunksize,
                'grouped': grouped
            }
        return arr
    
    def intersect(self, enforce_edges: bool = True, chunksize: int = 1000, grouped: bool = True) -> sp.csr_matrix:
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
            ).T
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
    def count(self, axis: int = 1, **kwargs) -> np.ndarray:
        """
        Count the number of intersections along the specified axis.

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
        arr = self._get_intersect_data(**kwargs)

        # Perform aggregation
        return arr.sum(axis=axis)
    
    @_get_selector_data_wrapper
    @_require_agg_data
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def single(self, index: int = 0, data: np.ndarray | pd.Series | pd.DataFrame | None = None, axis: int = 1, squeeze: bool = True, **kwargs) -> np.ndarray:
        """
        Aggregate all input data values along the specified axis of the events
        relation against the other axis, returning the first value for
        intersecting events.

        Parameters
        ----------
        index : int, default 0
            The index of the value to return for intersecting events. If the 
            index is out of bounds for a given event, NaNs will be returned for
            that event.
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
            where m is the number of events in the right dataframe if axis=0
            or the number of events in the left dataframe if axis=1.
        """
        # Validate index
        if not isinstance(index, int):
            raise ValueError("Index must be an integer.")
        
        # Check for cached data
        arr = self._get_intersect_data(**kwargs)
        arr = (arr.T if axis == 0 else arr).tocsr()

        # Iterate over sparse rows
        output = []
        for row in arr:
            # Get data values
            values = data[row.indices]
            try:
                values = values[index, :]
            except IndexError:
                values = np.empty((1, values.shape[1]))
                values.fill(np.nan)
            # Log values
            output.append(values)
        
        # Convert to numpy array of lists
        output_array = np.vstack(output)
        return output_array
    
    def first(self, data: np.ndarray | pd.Series | pd.DataFrame | None = None, axis: int = 1, squeeze: bool = True, **kwargs) -> np.ndarray:
        """
        Aggregate all input data values along the specified axis of the events
        relation against the other axis, returning the first value for
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
            where m is the number of events in the right dataframe if axis=0
            or the number of events in the left dataframe if axis=1.
        """
        return self.single(data=data, index=0, axis=axis, squeeze=squeeze, **kwargs)
    
    def last(self, data: np.ndarray | pd.Series | pd.DataFrame | None = None, axis: int = 1, squeeze: bool = True, **kwargs) -> np.ndarray:
        """
        Aggregate all input data values along the specified axis of the events
        relation against the other axis, returning the last value for
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
            where m is the number of events in the right dataframe if axis=0
            or the number of events in the left dataframe if axis=1.
        """
        return self.single(data=data, index=-1, axis=axis, squeeze=squeeze, **kwargs)
    
    @_get_selector_data_wrapper
    @_require_agg_data
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def list(self, data: np.ndarray | pd.Series | pd.DataFrame | None = None, axis: int = 1, squeeze: bool = True, **kwargs) -> np.ndarray:
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
        arr = (arr if axis == 1 else arr.T).tocsr()

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
    
    def set(self, data: np.ndarray | pd.Series | pd.DataFrame | None = None, axis: int = 1, squeeze: bool = True, **kwargs) -> np.ndarray:
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
    
    @_get_selector_data_wrapper
    @_require_agg_data
    @_validate_agg_1d_data_wrapper
    def value_counts(self, data: np.ndarray | pd.Series | None = None, axis: int = 1, **kwargs) -> pd.DataFrame:
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
        
        # Convert to CSR format for efficient row iteration
        arr = arr.tocsr()

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
    
    @_get_selector_data_wrapper
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def sum(self, data: np.ndarray | pd.Series | pd.DataFrame | None = None, method: str | None = None, axis: int = 1, squeeze: bool = True, **kwargs) -> np.ndarray:
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
        method : {'intersect', 'overlay'}, optional
            The method to use for the events relationship data being 
            multiplied.
            If not provided, the 'intersect' method will be used if point 
            events are being aggregated, otherwise the 'overlay' method will be
            used. If 'overlay' is selected but one or both event datasets are
            point events, an LRSConfigurationError will be raised.
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
        arr = self._get_method_data(method, axis, **kwargs)
        
        # Perform aggregation
        aggregated = []
        for column in data.T:
            # Reshape column for broadcasting
            column = column.reshape(-1, 1) if axis == 0 else column.reshape(1, -1)
            aggregated.append(arr.multiply(column).sum(axis=axis))
        
        # Concatenate results
        return np.vstack(aggregated).T
    
    @_get_selector_data_wrapper
    @_require_agg_data
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def mean(self, data: np.ndarray | pd.Series | pd.DataFrame | None = None, method: str | None = None, axis: int = 1, squeeze: bool = True, **kwargs) -> np.ndarray:
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
        method : {'intersect', 'overlay'}, optional
            The method to use for the events relationship data being 
            aggregated. If 'overlay', the overlay data will be used, producing
            a length-weighted mean. If 'intersect', the intersection data will
            be used, producing a count-weighted mean.
            If not provided, the 'intersect' method will be used if point 
            events are being aggregated, otherwise the 'overlay' method will be
            used. If 'overlay' is selected but one or both event datasets are
            point events, an LRSConfigurationError will be raised.
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
        arr = self._get_method_data(method, axis, **kwargs)
        
        # Perform aggregation
        aggregated = []
        for column in data.T:
            # Reshape column for broadcasting
            column = column.reshape(-1, 1) if axis == 0 else column.reshape(1, -1)
            numerator = arr.multiply(column).sum(axis=axis)
            denominator = arr.sum(axis=axis)
            # Use np.nan for division by zero (no matches) instead of 0
            result = np.full_like(numerator, np.nan, dtype=float)
            aggregated.append(np.divide(
                numerator,
                denominator,
                out=result,
                where=denominator!=0
            ))
        
        # Concatenate results
        return np.vstack(aggregated).T

    @_get_selector_data_wrapper
    @_require_agg_data
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def mode(self, data: np.ndarray | pd.Series | pd.DataFrame | None = None, axis: int = 1, method: str | None = None, squeeze: bool = True, **kwargs) -> np.ndarray:
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
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        method : {'intersect', 'overlay'}, optional
            The method to use for the events relationship data being 
            aggregated. If 'overlay', the overlay data will be used, producing
            a length-weighted mode. If 'intersect', the intersection data will
            be used, producing a count-weighted mode.
            If not provided, the 'intersect' method will be used if point 
            events are being aggregated, otherwise the 'overlay' method will be
            used. If 'overlay' is selected but one or both event datasets are
            point events, an LRSConfigurationError will be raised.
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
        arr = self._get_method_data(method, axis, **kwargs).tocsr() # Indexing requires CSR
        arr = arr if axis == 0 else arr.T
        
        # Perform aggregation
        aggregated = []
        for column_index, column in enumerate(data.T):
            # Sort unique values and identify splits in the weights data
            try:
                sorter = np.argsort(column)
            except TypeError:
                raise TypeError(
                    f"Selected data column index {column_index} contains non-"
                    "sortable values (e.g., mixed types or strings and null "
                    "values). Mode aggregation requires sortable data types."
                )
            unique, splitter = np.unique(column[sorter], return_index=True)
            splitter = np.append(splitter, len(column))
            # Compute weighted scores for each unique value
            # Use grouping matrix approach to avoid memory spike from fancy indexing.
            # The original approach arr[sorter] causes scipy to create large temporary
            # allocations. Instead, we build a sparse grouping matrix G where G[i,j]=1
            # if row i belongs to group j, then compute G.T @ arr to sum rows by group.
            # NOTE: If scipy adds more performant support for sparse fancy indexing, 
            # this can be simplified.
            n_unique = len(unique)
            n_rows = arr.shape[0]
            
            # Create mapping: which group does each row belong to?
            row_to_group = np.empty(n_rows, dtype=np.int32)
            for idx, (i, j) in enumerate(zip(splitter[:-1], splitter[1:])):
                row_to_group[sorter[i:j]] = idx
            
            # Build sparse grouping matrix: G[row, group] = 1 if row belongs to group
            row_indices = np.arange(n_rows)
            group_matrix = sp.csr_matrix(
                (np.ones(n_rows), (row_indices, row_to_group)),
                shape=(n_rows, n_unique)
            )
            
            # Vectorized computation: sum rows by group (G.T @ arr)
            # Keep result sparse to avoid memory issues with large column counts
            scores_sparse = group_matrix.T @ arr
            
            # Compute mode efficiently from sparse matrix
            # Get row sums to check which columns have data
            scores_sum = np.asarray(scores_sparse.sum(axis=0)).ravel()
            
            # Convert to CSC for efficient column access and find mode per column
            scores_csc = scores_sparse.tocsc()
            
            # Initialize mode array with appropriate dtype and fill value
            # For numeric types, use np.nan; for object/string types, use None
            if np.issubdtype(unique.dtype, np.number):
                mode = np.full(scores_sparse.shape[1], np.nan, dtype=float)
            else:
                mode = np.full(scores_sparse.shape[1], None, dtype=object)
            
            # Process only non-empty columns
            non_empty = scores_sum > 0
            for col_idx in np.where(non_empty)[0]:
                col_start = scores_csc.indptr[col_idx]
                col_end = scores_csc.indptr[col_idx + 1]
                if col_end > col_start:
                    # Find row with max value in this column
                    max_idx = np.argmax(scores_csc.data[col_start:col_end])
                    mode_row = scores_csc.indices[col_start + max_idx]
                    mode[col_idx] = unique[mode_row]
            aggregated.append(mode)

        # Concatenate results
        return np.vstack(aggregated).T
    
    @_get_selector_data_wrapper
    @_validate_agg_2d_data_wrapper
    @_squeeze_output_wrapper
    def distribute(
        self,
        data: np.array = None,
        axis: int = 1,
        method: str = None,
        decay_size: int = 0,
        decay_func: str = 'linear',
        direction: str = 'both',
        length_normalize: bool = True,
        chunksize: int = 1000,
        squeeze: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Distribute the share of intersecting events along neighboring events
        based on a decay function. The share of each intersecting event is
        distributed to neighboring events up to the specified decay size,
        with the share decreasing according to the specified decay function.
        Results are normalized to sum to 1.0 for each intersecting event and 
        can be length-normalized to favor longer events.

        Results of this method can be interpreted as a smoothed overlay of the
        events relationship, where intersecting events contribute to neighboring
        events based on distance and decay. This operates similar to a sliding 
        window analysis, where the window size is determined by the decay size
        and the aggregation of windows is determined by the decay function.

        The resulting distributed shares can be multiplied by input data to 
        weight the distribution by specific values. For example, if the input 
        data represents crash frequencies, the distributed shares can be used 
        to estimate a smoothed crash frequency profile along the linear events.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) or (n, x) where n is the number of events
            in the left dataframe if axis=0 or the number of events in the right
            dataframe if axis=1. This will result in an output shape of (m,) or 
            (m, x), respectively, where m is the number of events in the 
            opposite dataset.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        method : {'intersect', 'overlay'}, optional
            The method to use for the events relationship data being 
            aggregated. If 'overlay', the overlay data will be used, 
            distributing shares based on length of overlap. If 'intersect', 
            the intersection data will be used, distributing shares based on
            count of overlapping events.
            If not provided, the 'intersect' method will be used if point 
            events are being aggregated, otherwise the 'overlay' method will be
            used. If 'overlay' is selected but one or both event datasets are
            point events, an LRSConfigurationError will be raised.
        decay_size : int, default 0
            The number of neighboring events to which the share of intersecting
            events should be distributed. A decay size of 0 means no distribution
            will occur and intersecting events will retain their full share, 
            producing results similar to the `count` aggregator.
        decay_func : {'linear', 'exponential', 'gaussian', 'flat'} or DecayFunction, default 'linear'
            The decay function to use for distributing shares to neighboring
            events. This can be a string indicating a predefined decay function
            or an instance of a custom DecayFunction subclass.
            - 'linear' : Linearly decreases share with distance.
            - 'exponential' : Exponentially decreases share with distance.
            - 'gaussian' : Decreases share based on a Gaussian distribution.
            - 'flat' : No decrease in share with distance (equal distribution).
        direction : {'both', 'forward', 'backward'}, default 'both'
            The direction in which to distribute shares to neighboring events.
            - 'both' : Distribute shares to both forward and backward neighboring
              events.
            - 'forward' : Distribute shares only to forward neighboring events.
            - 'backward' : Distribute shares only to backward neighboring events.
        length_normalize : bool, default True
            Whether to length-normalize the distributed shares by multiplying
            by the lengths of the events. This assigns a greater share to longer
            events.
        chunksize : int or None, default 1000
            The maximum number of source event columns to process in a single
            chunk. Controls peak memory usage by avoiding materializing the
            full distributed matrix at once. Set to None to process all columns
            in a single pass (original behavior). This does not affect actual
            results, only computation.
        squeeze : bool, default True
            Whether to squeeze the output array to a 1D array if possible.
        """
        # Validate decay function
        if isinstance(decay_func, str):
            if decay_func in ['linear', 'lin']:
                decay_func = LinearDecay(decay_size)
            elif decay_func in ['exponential', 'exp']:
                decay_func = ExponentialDecay(decay_size)
            elif decay_func in ['gaussian', 'gauss']:
                decay_func = GaussianDecay(decay_size)
            elif decay_func in ['flat', 'none', None]:
                decay_func = FlatDecay(decay_size)
            else:
                raise ValueError(
                    "Invalid decay function string provided. Must be one of "
                    "'linear', 'exponential', 'gaussian', or 'flat'."
                )
        elif not isinstance(decay_func, DecayFunction):
            raise ValueError(
                "Invalid decay function provided. Must be a string or an "
                "instance of DecayFunction."
            )

        # Get relational data and adjust for axis.
        # Store as CSC since we only column-slice arr inside the loop.
        arr = self._get_method_data(method, axis, **kwargs)
        if axis == 0:
            arr = arr.T
            lengths = self.right.lengths.reshape(-1, 1)
        else:
            lengths = self.left.lengths.reshape(-1, 1)
        arr = arr.tocsc()

        # Validate decay function scale
        scale = decay_func(0)
        if not scale == 1:
            raise ValueError(
                "Decay function must return a scale of 1.0 for step 0."
            )

        n_rows = arr.shape[0]
        n_cols = arr.shape[1]

        # Pre-compute group membership vectors for efficient masking.
        # Instead of materializing the full (n_rows x n_cols) equal_groups
        # matrix (which can be multi-GB for large datasets), we store only
        # integer group codes for each event and check membership on the
        # sparse non-zeros directly. This is O(nnz) per chunk.
        row_group_codes = None
        col_group_codes = None
        if self.left.is_grouped:
            all_groups = np.concatenate([
                self.left.groups_data, self.right.groups_data
            ])
            _, codes = np.unique(all_groups, return_inverse=True)
            left_codes = codes[:self.left.num_events]
            right_codes = codes[self.left.num_events:]
            if axis == 0:
                row_group_codes = right_codes
                col_group_codes = left_codes
            else:
                row_group_codes = left_codes
                col_group_codes = right_codes

        # Pre-check for simple all-ones data optimization
        data_is_ones = (
            data is not None and data.shape[1] == 1 and np.all(data == 1)
        )
        n_out_cols = (
            1 if data is None or data_is_ones else data.shape[1]
        )
        output_array = np.zeros((n_rows, n_out_cols))

        # Process source event columns in chunks to limit memory usage.
        # Each column is independent through decay, normalization, and
        # aggregation, so partial results accumulate additively.
        if chunksize is None:
            chunksize = n_cols
        for col_start in range(0, n_cols, chunksize):
            col_end = min(col_start + chunksize, n_cols)
            arr_chunk = arr[:, col_start:col_end].tocsr()

            # Distribute intersection share based on decay function
            distributed = arr_chunk.copy()
            padded = sp.vstack([
                sp.csr_matrix((decay_size, arr_chunk.shape[1])),
                arr_chunk,
                sp.csr_matrix((decay_size, arr_chunk.shape[1]))
            ], format='csr')
            for step in range(1, min(decay_size + 1, n_rows)):
                # Aggregate shifted shares
                if direction in ['forward', 'forw', 'both']:
                    distributed += padded[decay_size + step : n_rows + decay_size + step, :] * decay_func(step)
                if direction in ['backward', 'back', 'both']:
                    distributed += padded[decay_size - step : n_rows + decay_size - step, :] * decay_func(step)
            del padded

            # Enforce equal groups by zeroing out distributed shares that
            # cross group boundaries. Uses vectorized group code comparison
            # on sparse non-zeros instead of full matrix multiplication.
            if row_group_codes is not None:
                chunk_col_codes = col_group_codes[col_start:col_end]
                row_indices = np.repeat(
                    np.arange(distributed.shape[0]),
                    np.diff(distributed.indptr)
                )
                group_match = (
                    row_group_codes[row_indices]
                    == chunk_col_codes[distributed.indices]
                )
                distributed.data[~group_match] = 0
                distributed.eliminate_zeros()

            # Multiply result shares by event lengths to favor longer events
            if length_normalize:
                distributed = distributed.multiply(lengths).tocsr() # Ensure format

            # Normalize result shares to sum to 1.0
            denominator = np.asarray(distributed.sum(axis=0)).flatten()
            # Normalize non-zero values in-place via direct CSR data array
            # access. In CSR format, distributed.indices holds the column
            # index for each stored value, so np.take maps each value to its
            # column's denominator.
            nonzero_mask = denominator[distributed.indices] != 0
            distributed.data[nonzero_mask] /= \
                denominator[distributed.indices[nonzero_mask]]

            # Accumulate chunk results into output
            if data is not None and not data_is_ones:
                data_chunk = data[col_start:col_end, :]
                for k, column in enumerate(data_chunk.T):
                    output_array[:, k] += np.asarray(
                        distributed.multiply(column).sum(axis=1)
                    ).flatten()
            else:
                output_array += np.asarray(
                    distributed.sum(axis=1)
                ).reshape(n_rows, 1)

            del distributed

        return output_array

    @_get_linestring_m_data_wrapper
    @_require_agg_data
    @_validate_agg_1d_data_wrapper
    def interpolate(self, data: np.ndarray | None = None, axis: int = 1, multiple: str = 'first', **kwargs) -> np.ndarray:
        """
        Interpolate new point geometries from intersecting events based on the 
        location of the target events, returning an array of interpolated
        geometries.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) where n is the number of events in the left 
            dataframe if axis=0 or the number of events in the right dataframe if 
            axis=1. This will result in an output shape of (m,), where m is the 
            number of events in the opposite dataframe.

            Data must contain LineStringM objects.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        multiple : {'first', 'last', 'merge', 'list', 'raise'}, default 'first'
            The strategy to use when multiple geometries intersect.
            - 'first' : Use the first intersecting geometry only.
            - 'last' : Use the last intersecting geometry only.
            - 'list' : Return a list of all interpolated geometries.
            - 'raise' : Raise an error if multiple geometries intersect.
        **kwargs
            Additional keyword arguments to pass to the intersection method
            if it has not been previously computed and cached.

        Returns
        -------
        arr : numpy.ndarray
            An array containing the interpolated geometries for all 
            intersecting events. The shape of the array will be (m,), where m 
            is the number of events in the right dataframe if axis=0 or the 
            number of events in the left dataframe if axis=1.
        """
        # Ensure both linear and point datasets contain relevant features
        if axis == 1 and not (self.left.is_point and self.right.is_linear):
            raise ValueError(
                "Left events must be point and right events must be linear "
                "for 'interpolate' aggregation with axis=1."
            )
        if axis == 0 and not (self.left.is_linear and self.right.is_point):
            raise ValueError(
                "Left events must be linear and right events must be point "
                "for 'interpolate' aggregation with axis=0."
            )
        
        # Check for cached data
        arr = self._get_intersect_data(**kwargs)
        arr = (arr if axis == 1 else arr.T).tocsr()

        # Pull event locations
        locs = self.left.locs if axis == 1 else self.right.locs

        # Iterate over sparse rows
        output = []
        for row, loc in zip(arr, locs):
            # Get data values
            geoms = data[row.indices]
            # Determine approach to interpolate multiple geometries
            if multiple == 'first':
                geoms = geoms[:1]
            elif multiple == 'last':
                geoms = geoms[-1:]
            elif multiple == 'raise':
                if len(geoms) > 1:
                    raise ValueError(
                        "Multiple intersecting geometries found when 'raise' "
                        "option is set for 'interpolate' aggregation."
                    )
            # Interpolate new geometries from intersecting geometries
            interp_geoms = [
                geom.interpolate(loc, normalized=False, m=True, snap=True)
                for geom in geoms
            ]
            # Determine approach to post-processing multiple geometries
            if multiple == 'list':
                pass  # Keep as list
            else:  # 'first' or 'last'
                interp_geoms = interp_geoms[0] if interp_geoms else None
            # Log values
            output.append(interp_geoms)

        # Convert to numpy array of lists if needed
        if multiple == 'list':
            output_array = np.empty((len(output), data.shape[0]), dtype=object)
            output_array[:] = output
        else:
            output_array = np.array(output, dtype=object)
        return output_array
    
    @_get_linestring_m_data_wrapper
    @_require_agg_data
    @_validate_agg_1d_data_wrapper
    def cut(self, data: np.ndarray | None = None, axis: int = 1, multiple: str = 'first', **kwargs) -> np.ndarray:
        """
        Cut new linear geometries from intersecting events based on the begin 
        and end bounds of the target events, returning an array of cut 
        geometries.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) where n is the number of events in the left 
            dataframe if axis=0 or the number of events in the right dataframe if 
            axis=1. This will result in an output shape of (m,), where m is the 
            number of events in the opposite dataframe.

            Data must contain LineStringM objects.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        multiple : {'first', 'last', 'merge', 'list', 'raise'}, default 'first'
            The strategy to use when multiple geometries intersect.
            - 'first' : Use the first intersecting geometry only.
            - 'last' : Use the last intersecting geometry only.
            - 'merge' : Attempt to merge all intersecting geometries into a 
                        single M-enabled geometry.
            - 'list' : Return a list of all cut intersecting geometries.
            - 'raise' : Raise an error if multiple geometries intersect.
        **kwargs
            Additional keyword arguments to pass to the intersection method
            if it has not been previously computed and cached.

        Returns
        -------
        arr : numpy.ndarray
            An array containing the cut geometries for all intersecting events. 
            The shape of the array will be (m,), where m is the number of 
            events in the right dataframe if axis=0 or the number of 
            events in the left dataframe if axis=1.
        """
        # Ensure both datasets are linear and contain relevant features
        if not (self.left.is_linear and self.right.is_linear):
            raise ValueError(
                "Both left and right events must be linear for 'cut' aggregation."
            )
        
        # Check for cached data
        kwargs['enforce_edges'] = False # Edges not needed for cutting
        arr = self._get_intersect_data(**kwargs)
        arr = (arr if axis == 1 else arr.T).tocsr()

        # Pull event bounds
        begs = self.left.begs if axis == 1 else self.right.begs
        ends = self.left.ends if axis == 1 else self.right.ends

        # Iterate over sparse rows
        output = []
        for row, beg, end in zip(arr, begs, ends):
            # Get data values
            geoms = data[row.indices]
            # Determine approach to cut multiple geometries
            if multiple == 'first':
                geoms = geoms[:1]
            elif multiple == 'last':
                geoms = geoms[-1:]
            elif multiple == 'raise':
                if len(geoms) > 1:
                    raise ValueError(
                        "Multiple intersecting geometries found when 'raise' "
                        "option is set for 'cut' aggregation."
                    )
            # Cut new geometries from intersecting geometries
            cut_geoms = [
                geom.cut(beg, end, normalized=False, m=True, snap=True)
                for geom in geoms
            ]
            # Determine approach to post-processing multiple geometries
            if multiple == 'merge':
                cut_geoms = geometry.line_merge_m(
                    cut_geoms,
                    allow_multiple=False,
                    allow_mismatch=False,
                    squeeze=True,
                    cast_geom=True,
                )
            elif multiple == 'list':
                pass  # Keep as list
            else:  # 'first' or 'last'
                cut_geoms = cut_geoms[0] if cut_geoms else None
            # Log values
            output.append(cut_geoms)
        
        # Convert to numpy array of lists if needed
        if multiple == 'list':
            output_array = np.empty((len(output), data.shape[0]), dtype=object)
            output_array[:] = output
        else:
            output_array = np.array(output, dtype=object)
        return output_array

    @_get_selector_data_wrapper
    @_require_agg_data
    @_validate_agg_1d_data_wrapper
    def line_merge_m(self, data: np.ndarray | None = None, axis: int = 1, **kwargs) -> pd.Series:
        """
        Aggregate all input data values along the specified axis of the events
        relationship against the other axis, returning a pandas Series with
        the linemerged values for intersecting events.

        Parameters
        ----------
        data : array-like or None, default None
            The data to aggregate along the axis of the events relationship. Data 
            must have a shape of (n,) where n is the number of events in the left 
            dataframe if axis=0 or the number of events in the right dataframe if 
            axis=1. This will result in an output shape of (m,), where m is the 
            number of events in the opposite dataframe.

            Data must contain LineString or LineStringM objects.
        axis : int, default 1
            The axis along which to aggregate the events relationship.
            - 0 : Aggregate left events onto the right events index.
            - 1 : Aggregate right events onto the left events index.
        **kwargs
            Additional keyword arguments to pass to the intersection method
            if it has not been previously computed and cached.

        Returns
        -------
        ser : pandas.Series
            A Series containing the merged linear geometries for all
            intersecting events. The index of the Series will be the index 
            of the events in the opposite dataframe.
        """
        # Check for cached data
        kwargs['enforce_edges'] = False # Edges not needed for merging
        arr = self._get_intersect_data(**kwargs)
        arr = (arr if axis == 1 else arr.T).tocsr()

        # Iterate over sparse rows
        output = []
        for i, row in enumerate(arr):
            # Get data values
            geoms = data[row.indices]
            try:
                merged = geometry.line_merge_m(
                    geoms,
                    allow_multiple=False,
                    allow_mismatch=False,
                    squeeze=True,
                    cast_geom=True,
                )
            except Exception as e:
                raise type(e)(
                    f"Error occurred during merge on row number {i}. "
                    f"{e}"
                ) from e
            # Log values
            output.append(merged)
        
        # Convert to pandas Series
        index = self.left.index if axis == 1 else self.right.index
        return pd.Series(output, index=index)


# -----------------------------------------------------------------------------
# Events relation core methods
# -----------------------------------------------------------------------------

def _grouped_operation_wrapper(func) -> callable:
    """
    Decorator for wrapping functions that operate on grouped data.
    
    Performance optimization: Uses sorted group iteration with boundary lookups
    instead of repeated np.isin calls for significant speedup without memory overhead.
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
            # Sort both datasets and get group boundaries
            # This avoids repeated np.isin calls while keeping memory usage low
            left_sorted = left.reset_index().sort_standard(inplace=False)
            right_sorted = right.reset_index().sort_standard(inplace=False)
            
            # Get unique groups and their boundaries for both datasets
            left_unique, left_splits = np.unique(left_sorted.groups, return_index=True)
            right_unique, right_splits = np.unique(right_sorted.groups, return_index=True)
            
            # Append end positions for slicing
            left_splits = np.append(left_splits, len(left_sorted.groups))
            right_splits = np.append(right_splits, len(right_sorted.groups))
            
            # Get all unique groups across both datasets
            unique_groups = np.unique(np.concatenate([left.groups, right.groups]))
            
            # Create searchsorted-based lookup for O(log n) group boundary access
            # This avoids dictionary key hashing issues with structured dtypes
            left_positions = np.searchsorted(left_unique, unique_groups)
            right_positions = np.searchsorted(right_unique, unique_groups)
            
            # Iterate over groups
            left_index = []
            right_index = []
            res = []
            
            for i, group in enumerate(unique_groups):
                # Check if group exists in left using searchsorted result
                left_pos = left_positions[i]
                if left_pos < len(left_unique) and np.array_equal(left_unique[left_pos], group):
                    left_slice = slice(left_splits[left_pos], left_splits[left_pos + 1])
                    left_group = left_sorted.select(left_slice, inplace=False)
                    left_group.ungroup(inplace=True)
                else:
                    left_group = base.EventsData(
                        begs=np.array([]) if left.is_linear else None,
                        ends=np.array([]) if left.is_linear else None,
                        locs=np.array([]) if left.is_point else None
                    )
                
                # Check if group exists in right using searchsorted result
                right_pos = right_positions[i]
                if right_pos < len(right_unique) and np.array_equal(right_unique[right_pos], group):
                    right_slice = slice(right_splits[right_pos], right_splits[right_pos + 1])
                    right_group = right_sorted.select(right_slice, inplace=False)
                    right_group.ungroup(inplace=True)
                else:
                    right_group = base.EventsData(
                        begs=np.array([]) if right.is_linear else None,
                        ends=np.array([]) if right.is_linear else None,
                        locs=np.array([]) if right.is_point else None
                    )
                
                # Log indices
                left_index.append(left_group.index)
                right_index.append(right_group.index)
                
                # Don't compute if either group is empty, instead set empty array
                if left_group.num_events == 0:
                    group_res = np.array([]).reshape(0, right_group.num_events)
                elif right_group.num_events == 0:
                    group_res = np.array([]).reshape(left_group.num_events, 0)
                # Compute operation on group
                else:
                    group_res = func(left_group, right_group, *args, **kwargs)
                
                # Log resulting array and indices
                res.append(group_res)
            
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
def overlay(left, right, normalize=True, norm_by='right', profile=None, chunksize=None) -> sp.csr_matrix:
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
    profile : str, EventProfile, or None, default None
        An event profile defining how value is distributed along the
        event's length. When provided, overlay weights reflect the
        proportion of the event's profiled value that is overlapped
        rather than simple length proportion. Only applied when
        `normalize` is True.
        - None : No profiling (default, simple length-based normalization).
        - str : Name of a built-in profile ('uniform', 'triangular',
          'parabolic', 'trapezoidal').
        - EventProfile : A profile instance for custom parameterization.
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

    # Resolve profile
    profile = resolve_profile(profile)

    # Early exit for completely disjoint event collections
    if left.begs.min() >= right.ends.max() or right.begs.min() >= left.ends.max():
        # No possible overlaps - return zero matrix
        return np.zeros((left.num_events, right.num_events))

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

    # Apply group masking if necessary
    if left.is_grouped:
        # Identify matching groups
        mask = left.groups.reshape(-1, 1) == right.groups.reshape(1, -1)
        np.multiply(overlap, mask, out=overlap)

    # Normalize if necessary
    if normalize:
        if profile is not None:
            # Profile-based normalization: compute the integral of the profile
            # over the normalized overlap region within the norm_by event.
            # Determine overlap start/end in absolute coordinates
            overlap_start = np.maximum(
                left.begs.reshape(-1, 1), right.begs.reshape(1, -1)
            )
            overlap_end = np.minimum(
                left.ends.reshape(-1, 1), right.ends.reshape(1, -1)
            )
            # Compute has-overlap mask before in-place normalization
            has_overlap = overlap_end > overlap_start
            # Normalize positions to [0, 1] within the norm_by event
            if norm_by == 'right':
                event_begs = right.begs.reshape(1, -1)
                event_lengths = right.lengths.reshape(1, -1)
            elif norm_by == 'left':
                event_begs = left.begs.reshape(-1, 1)
                event_lengths = left.lengths.reshape(-1, 1)
            else:
                raise ValueError(
                    f"Invalid 'norm_by' parameter value provided ({norm_by}). Must be one "
                    f"of {_norm_by_options}.")
            # Normalize overlap positions in-place (reuse overlap_start/end)
            safe_lengths = np.where(event_lengths == 0, np.inf, event_lengths)
            np.subtract(overlap_start, event_begs, out=overlap_start)
            np.divide(overlap_start, safe_lengths, out=overlap_start)
            np.subtract(overlap_end, event_begs, out=overlap_end)
            np.divide(overlap_end, safe_lengths, out=overlap_end)
            np.clip(overlap_start, 0, 1, out=overlap_start)
            np.clip(overlap_end, 0, 1, out=overlap_end)
            # Compute profiled weights
            overlap = profile.integral(overlap_start, overlap_end)
            # Zero out where there's no actual overlap
            np.multiply(overlap, has_overlap, out=overlap)
            # Re-apply group mask
            if left.is_grouped:
                np.multiply(overlap, mask, out=overlap)
        else:
            # Standard length-based normalization
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
    
    return overlap

@_grouped_operation_wrapper
@_chunked_operation_wrapper
def equal_groups(left, right, chunksize=None) -> sp.csr_matrix:
    """
    Identify equal groups between two collections of events.
    """
    # Validate inputs
    if not isinstance(left, base.EventsData) or not isinstance(right, base.EventsData):
        raise TypeError("Input objects must be EventsData class instances.")
    if left.is_grouped != right.is_grouped:
        raise ValueError("Input collections must have the same grouping status.")

    # If not grouped, return all True array
    if not left.is_grouped:
        res = np.ones((left.num_events, right.num_events), dtype=bool)
    else:
        # Reshape arrays for broadcasting
        left_groups = left.groups.reshape(-1, 1)
        right_groups = right.groups.reshape(1, -1)
        # Test for equality of groups
        res = np.equal(left_groups, right_groups)
    
    return res

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

    # Early exit for completely disjoint event collections
    if left.begs.min() >= right.ends.max() or right.begs.min() >= left.ends.max():
        # No possible intersections - return zero matrix
        return np.zeros((left.num_events, right.num_events), dtype=bool)

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


# -----------------------------------------------------------------------------
# Decay functions for distribution
# -----------------------------------------------------------------------------


class DecayFunction:
    """
    Base class for decay functions used in distribution. To define a custom
    decay function, subclass this class and implement the `decay` method.
    All decay functions must take a distance (float) as input and return a 
    decay weight (float) between 0 and 1.
    """
    def __init__(self, decay_size: float):
        self.decay_size = decay_size
    
    def __call__(self, distance: float) -> float:
        # Check decay size
        if self.decay_size == 0:
            return 1.0
        # Validate distance
        if distance < 0:
            raise ValueError("Distance must be non-negative.")
        elif distance == 0:
            return 1.0
        # Apply decay function
        return self.decay(distance)
    
    @property
    def decay_size(self) -> float:
        return self._decay_size
    
    @decay_size.setter
    def decay_size(self, value: float):
        if value < 0:
            raise ValueError("Decay size must be non-negative.")
        self._decay_size = value

    @property
    def decay_cap(self) -> float:
        return self.decay_size + 1
        
    def decay(self, distance: float) -> float:
        """
        Compute the decay weight for a given distance.

        Parameters
        ----------
        distance : float
            The distance from the target event.

        Returns
        -------
        weight : float
            The decay weight between 0 and 1.
        """
        raise NotImplementedError("DecayFunction is an abstract base class.")


class LinearDecay(DecayFunction):
    """
    Linear decay function.
    """
    
    def decay(self, distance: float) -> float:
        return 1.0 - (distance / self.decay_cap)
        

class ExponentialDecay(DecayFunction):
    """
    Exponential decay function.
    """
    
    def __init__(self, decay_size: float, rate: float = 1.0):
        super().__init__(decay_size)
        self.rate = rate

    def decay(self, distance: float) -> float:
        return np.exp(-5 * (distance / self.decay_cap))
        
    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float):
        if value < 0:
            raise ValueError("Rate must be non-negative.")
        self._rate = value


class GaussianDecay(DecayFunction):
    """
    Gaussian decay function, producing values between 1.0 at distance 0 and 
    approximately 0.01 at distance equal to decay_size, following a normal
    distribution curve.
    """
    
    def __init__(self, decay_size: float):
        super().__init__(decay_size)
        self.denominator = norm.pdf(0, loc=0, scale=1)

    def decay(self, distance: float) -> float:
        return norm.pdf(distance / self.decay_size * 3, loc=0, scale=1) / self.denominator
        

class FlatDecay(DecayFunction):
    """
    Flat decay function.
    """

    def decay(self, distance: float) -> float:
        return 1.0