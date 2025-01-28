import numpy as np
from linref.events import base
from scipy import sparse as sp


class EventsRelation(object):

    def __init__(self, left, right, cache=True):
        self.cache = cache
        self._left = left
        self._right = right
        self._validate_events()
        self.reset_cache()

    @property
    def left(self):
        return self._left
    
    @left.setter
    def left(self, value):
        self._left = value
        self._validate_events()
        self.reset_cache()

    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self, value):
        self._right = value
        self._validate_events()
        self.reset_cache()

    @property
    def cache(self):
        return self._cache
    
    @cache.setter
    def cache(self, value):
        if not isinstance(value, bool):
            raise TypeError("The 'cache' parameter must be a boolean.")
        self._cache = value
        if not value:
            self.reset_cache()

    @property
    def shape(self):
        return self.left.num_events, self.right.num_events
    
    @property
    def overlay_data(self):
        return self._overlay_data
        
    @property
    def intersect_data(self):
        return self._intersect_data
    
    def _validate_events(self):
        """
        Validate the input events.
        """
        if not isinstance(self.left, base.Rangel) or not isinstance(self.right, base.Rangel):
            raise TypeError("Input objects must be Rangel class instances.")
        if self.left.is_grouped != self.right.is_grouped:
            raise ValueError("Input objects must have the same grouping status.")

    def reset_cache(self):
        self._overlay_data = None
        self._intersect_data = None
        
    def overlay(self, normalize=True, norm_by='right', chunksize=1000, grouped=True):
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
        chunksize : int or None, default 1000
            The maximum number of events to process in a single chunk.
            Input chunksize will affect the memory usage and performance of
            the function. This does not affect actual results, only 
            computation.
        grouped : bool, default True
            Whether to process the overlay operation for each group separately.
            This will affect the memory usage and performance of the function. 
            This does not affect actual results, only computation.
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
        return arr
    
    def intersect(self, enforce_edges=True, chunksize=1000, grouped=True):
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
        return arr

def _grouped_operation_wrapper(func):
    """
    Decorator for wrapping functions that operate on grouped data.
    """
    def wrapper(left, right, *args, **kwargs):
        # Validate inputs
        if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
            raise TypeError("Input objects must be Rangel class instances.")
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

def _chunked_operation_wrapper(func):
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
def overlay(left, right, normalize=True, norm_by='right', chunksize=None):
    """
    Compute the overlay of two collections of events.

    Parameters
    ----------
    left, right : Rangel
        Input Rangel instances to overlay.
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
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
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
def intersect_point_point(left, right, chunksize=None):
    """
    Identify intersections between two collections of point events.
    """
    # Validate inputs
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
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
def intersect_point_linear(left, right, enforce_edges=True, chunksize=None):
    """
    Identify intersections between a collection of point events and a collection 
    of linear events.
    """
    # Validate inputs
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
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
def intersect_linear_linear(left, right, enforce_edges=True, chunksize=None):
    """
    Identify intersections between two collections of linear events.
    """
    # Validate inputs
    if not isinstance(left, base.Rangel) or not isinstance(right, base.Rangel):
        raise TypeError("Input objects must be Rangel class instances.")
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
